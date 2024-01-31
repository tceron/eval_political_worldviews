import openai
import fire
import json
import pandas as pd
import time
from pathlib import Path

from statement_retrieval import StatementRetriever
from itertools import product
from utils.check_chatgpt_token_count import num_tokens_from_messages, total_num_tokens_from_messages

def format_openai_prompt(templates: pd.DataFrame, statements: pd.DataFrame, num_return_sequences: int):

    templates = templates[templates['lang'] == 'en']

    responses = pd.DataFrame(columns=['template_id', 'statement_id', 'prompt'])
    templates_ids = []
    statement_ids = []
    prompts = []

    chat_gpt_prompts = []

    total_prompts = len(list(product(templates['template_id'], statements['full_statement_id'])))
    total_data = total_prompts * num_return_sequences

    print('translations: ', len(templates), '|', len(statements), '|', total_prompts,'|', total_data )
    
    for zipped in product(templates['template_id'], statements['full_statement_id']):

        template_id = templates[templates['template_id'] == zipped[0]]['template_id'].values[0]
        template = templates[templates['template_id'] == zipped[0]]['template'].values[0]

        statement = statements[statements['full_statement_id'] == zipped[1]]['statement'].values[0]

        prompt = template + ' ' + statement

        templates_ids.append(template_id)
        statement_ids.append(zipped[1])
        prompts.append(prompt)

        chat_gpt_prompts.append([{"role": "system", "content": template},
                                 {"role": "user", "content": statement}
                          ])

    responses['template_id'] = templates_ids
    responses['statement_id'] = statement_ids
    responses['prompt'] = prompts

    return chat_gpt_prompts, responses

def run_chat_gpt(temperature: float = 1.0,    
                 top_p: float = 0.85,          
                 top_k: float = 0,         
                 max_gen_len: int = 20,
                 num_return_sequences: int = 30,
                 seed: int = 42,
                 is_src_prompt: bool = False,              # True - source lang | False - english translations
                 is_src_statement: bool = False,           # True - source lang  | False - english translations
                 template_dir : str = 'data/prompt_instructions/templates_src/en_chatgpt_templates_final.csv',
                 statement_dir: str = 'data/vaa/all_unique.csv',
                 output_dir: str = 'data/responses/final_new/',
                 token_limit: int = 60000,
                 rpd_limit: int = 10000,
                 rpm_limit: int = 500):
    
    script_starting_time = time.time()

    # Read and set OpenAI API key
    with open('openai_api_key.txt') as f:
        api_key = f.read()

    openai.api_key = api_key

    # Set the model
    MODEL = "gpt-3.5-turbo-0613"

    # Load prompt templates
    templates = pd.read_csv(template_dir)
    templates = templates[(templates['lang'] == 'en') & (templates['model_type'] == 'chat')][:3]

    # Load statements
    statements = pd.read_csv(statement_dir)
    retriever = StatementRetriever(statements) 
    chosen_statements = retriever.get_english_statements()[:3]

    # Merge templates with statements
    chatgpt_prompts, prompts_df = format_openai_prompt(templates, chosen_statements, num_return_sequences)

    # Approximate ChatGPT cost
    total_num_tokens_from_messages(chatgpt_prompts, MODEL, num_return_sequences, max_gen_len)

    # Generate responses

    all_responses_df = pd.DataFrame(columns=['template_id', 'statement_id', 'prompt', MODEL+'_response']) 

    total_request_num = 0

    for i in range(len(chatgpt_prompts)):

        token_counter = 0
        request_counter = 0

        # Save data
        responses = []
        templates_ids = [prompts_df['template_id'].iloc[i]for j in range(num_return_sequences) ]
        statements_ids = [prompts_df['statement_id'].iloc[i] for j in range(num_return_sequences)]
        prompt_list = [prompts_df['prompt'].iloc[i] for j in range(num_return_sequences)]

        # Generate responses
        prompt = chatgpt_prompts[i]

        # Check if rate limit is surpassed
        token_num = num_tokens_from_messages(prompt) * num_return_sequences
        token_counter += token_num
        request_counter += num_return_sequences
        total_request_num += num_return_sequences

        script_time_checkpoint = time.time()
        script_minutes_passed = (script_time_checkpoint - script_starting_time) / 60

        if script_minutes_passed <= 1439.0:
        # Stop the script if the total number of requests exceed rpd limit
            if (total_request_num >= rpd_limit):
                print(i-1)
                exit()

        start_time = time.time()

        response = openai.chat.completions.create(model=MODEL,
                                                  messages=prompt,
                                                  n = num_return_sequences,
                                                  temperature=temperature,
                                                  top_p = top_p,
                                                  max_tokens = max_gen_len                     
                                                )
        
        for choice in response.choices:
            responses.append(choice.message.content)

    
        response_df = pd.DataFrame({'template_id': templates_ids,
                                     'statement_id': statements_ids,
                                     'prompt': prompt_list,
                                     MODEL+'_response': responses})
        
        all_responses_df = pd.concat([all_responses_df, response_df], ignore_index=True)
        
        if i % 100 == 0:
            all_responses_df.to_csv(f'data/responses/chatgpt/{MODEL}_en.csv', index=False)
            print(f'Response checkpoint: {i}')

        # Check rate limits
        end_time = time.time()
        minutes_passed = (end_time - start_time) / 60

        if (minutes_passed <= 1.0):
            # If token limit or request limit is reached then reset the counter and sleep for 1 minute
            if (token_counter >= token_limit) or (request_counter >= rpm_limit):
                token_counter = 0
                request_counter = 0
                time.sleep(60.1)

    all_responses_df.to_csv(f'data/responses/chatgpt/{MODEL}_en.csv', index=False)

if __name__ == '__main__':
    fire.Fire(run_chat_gpt)
