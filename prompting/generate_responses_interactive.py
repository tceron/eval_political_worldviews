import fire
import torch
import pandas as pd

from prompting.huggingface_llm_wrapper import hf_loader, CausalLLM, Seq2SeqLLM
from prompting.llm_dict import llm_dict

from itertools import product

def generate_responses_interacitive(model_type: str, #llama-2-13b
              temperature: float = 1.0,
              top_p: float = 1.0,
              max_gen_len: int = 20,
              seed: int = 42,
              instruction_path : str = 'data/prompt_instructions/classification_prompts/en_chat_impersonal_default.txt',
              statement_path : str = 'data/vaa/all_unique_sampled.csv'
              ):
    
    model = hf_loader(llm_dict[model_type], device_map='auto', torch_dtype=torch.bfloat16)

    # Load prompt instructions from .txt file 
    if '.txt' in statement_path:
        with open(statement_path) as file:
            statements = [line.rstrip() for line in file]
            print(statements)
    else:
        statements = pd.read_csv(statement_path)['statement'].tolist()

    running = True
    while running:

        stop_sig = input('Press Enter to continue or stop to quit: ')
        if stop_sig == 'stop':
            running = False
            break

        # Load prompt instructions from .txt file 
        with open(instruction_path) as file:
            instructions = [line.rstrip() for line in file]
            print(instructions)

        prompts = []
        for i, s in product(instructions, statements):
            prompt = i % s
            print(prompt)
            prompts.append(prompt)

        # Step 5: Generate responses
        outputs, tokens, probs = model.generate(prompts,
                                 do_sample=True,
                                 temperature=temperature, 
                                 top_p=top_p,
                                 max_new_tokens=max_gen_len,
                                 num_return_sequences=10
                                )
        
        for output in outputs:
            for o in output:
                print(o)
                print('##############################################################################')
            print('------------------------------------------------------------------------------')

    print('Done!')


if __name__ == "__main__":
    '''
    Usage example: torchrun --nproc_per_node 1 models/generate_responses_interacitive.py --model_type 7B --option option_value

    --model_type:
        - LLama:        7B  / 13B / 30B / 65B
        - LLama 2:      llama-2-7b / llama-2-13b / llama-2-70b
        - LLama 2 Chat: llama-2-7b-chat / llama-2-13b-chat / llama-2-70b-chat

    -- nproc_per_node param changes depending on the model size:
        - 7B -> 1
        - 13B -> 2
        - 30B -> 4
        - 65B/70B -> 8

    '''
    fire.Fire(generate_responses_interacitive)