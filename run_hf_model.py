import fire
import torch
import json
import pandas as pd
import time

from transformers import set_seed
from pathlib import Path

from models.hugging_face_llm import hf_loader, CausalLLM, Seq2SeqLLM
from models.llm_dict import llm_dict
from utils.format_prompts import format_prompts
from utils.generate_dir_name import generate_dir_name
from statement_retrieval import StatementRetriever


def run_hf_model(model: [str], 
                 do_sample: bool = True,      # False = greedy search / True = sampling              
                 temperature: float = 1.0,    
                 top_p: float = 0.85,          
                 top_k: float = 0,         
                 max_gen_len: int = 20,
                 num_return_sequences: int = 30,
                 seed: int = 42,
                 is_src_prompt: bool = False,              # True - source lang | False - english translations
                 is_src_statement: bool = True,            # True -source lang  | False - english translations
                 template_dir : str = 'data/prompt_instructions/classification_prompts/en_clf_templates_final.csv',
                 statement_dir: str = 'data/vaa/all_unique_missing.csv',
                 output_dir: str = 'data/responses/final/generation_llama-2-70b_en_missing.csv'
              ):
    
    # Step 0: Set seed or reproducibility (you don't need this unless you want full reproducibility)
    set_seed(seed)

    # Step 1: Load model(s)
    model_list = model.split(',')
    # Check if all models are either base or instruction type
    model_types = list(set([llm_dict[model]['chat'] for model in model_list]))
    assert len(model_types) == 1, "All models should be either base- or instruction-type."

    # Step 2: Load prompt templates
    templates = pd.read_csv(template_dir)

    # Select templates based on model type (chat or generation)
    gen_type = 'chat' if llm_dict[model_list[0]]['chat'] else 'generation'
    templates = templates[templates['model_type'] == gen_type]

    # Step 3: Load & choose statements
    statements = pd.read_csv(statement_dir)
    retriever = StatementRetriever(statements) 
    chosen_statements = retriever.get_source_lang_statements() if is_src_statement else retriever.get_english_statements()

    # Step 4: Glue prompt template with inputs
    template_ids, statement_ids, prompts = format_prompts(templates, chosen_statements, is_src_prompt)
    response_df = pd.DataFrame(columns=['template_id', 'statement_id', 'prompt'])

    # Generate CSV name from hyperparameters
    if '.csv' not in output_dir:
      Path(output_dir).mkdir(parents=True, exist_ok=True)

    output_dir_gen, output_json = generate_dir_name(output_dir,
                                  model_list,
                                  gen_type, 
                                  do_sample, 
                                  temperature, 
                                  top_p, 
                                  top_k,  
                                  is_src_statement,
                                  max_gen_len)
    
    for model_type in model_list:
        
        print(model_type)   
        start_time = time.time() 
        model = hf_loader(llm_dict[model_type], device_map='auto', torch_dtype=torch.bfloat16)

        # Step 5: Generate responses
        outputs, gen_tokens, probs = model.generate(prompts, 
                                                    do_sample=do_sample,
                                                    temperature=temperature, 
                                                    top_p=top_p,
                                                    max_new_tokens=max_gen_len,
                                                    num_return_sequences=num_return_sequences
                                                    )
        
        templates_ids = []
        statements_ids = []
        prompt_list = []
        response_list = []
        gen_token_list = []
        prob_list = []

        # Format and save responses
        for response_tuple in zip(template_ids, statement_ids, prompts, outputs, gen_tokens, probs):
          
          template_id = response_tuple[0]
          statement_id = response_tuple[1]
          prompt = response_tuple[2]

          responses = response_tuple[3]
          gen_token = response_tuple[4]
          prob = response_tuple[5]

          if len(response_df['template_id']) == 0:
            templates_ids.extend([template_id for i in range(len(responses))])
            statements_ids.extend([statement_id for i in range(len(responses))])
            prompt_list.extend([prompt for i in range(len(responses))])

          response_list.extend([response for response in responses])
          gen_token_list.extend([tokens.flatten() for tokens in gen_token])
          prob_list.extend([p.flatten() for p in prob])
        
        if len(response_df['template_id']) == 0:
          response_df['template_id'] = templates_ids
          response_df['statement_id'] = statements_ids
          response_df['prompt'] = prompt_list

        response_df[model_type + '_response'] = response_list
        response_df[model_type + '_tokens'] = gen_token_list
        response_df[model_type + '_probs'] = prob_list

        # Save compute time
        end_time = time.time()
        output_json[model_type + '_time'] = end_time - start_time

        # Clear cache
        del model
        torch.cuda.empty_cache()

        # Save responses - CSV header: [input_id, prompt, model_type_response
        response_df.to_csv(output_dir_gen + '.csv', index=False)
    
    # Save hyperparams dict
    with open(output_dir_gen + '.json', 'w') as fp:
        json.dump(output_json, fp)


if __name__ == "__main__":
    
    # How to run: python run_hf_model.py --model model1,model2,model3 
    #     - If you are running the script for src languages add: --is_src_statement=True
    #     - If you are running the script for English language add: --is_src_statement=False
    #     - If you are running the script for src prompts add: --is_src_prompt=True
    #     - If you are running the script for English prompts add: --is_src_prompt=False

    # All other hyperparameter values are set so no need to change them 
    fire.Fire(run_hf_model)