import fire
from models.llama_class import *
from prompting.prompts import *
from pathlib import Path
from collections import defaultdict

MODEL_PATH = '/mount/arbeitsdaten32/projekte/mardy/baricaa/models/'

VAA = {'cz': 'data/cz_unique.csv',
       'de': 'data/vaa/de_unique.csv',
       'es': 'data/vaa/es_unique.csv',
       'nl': 'data/vaa/nl_unique.csv',
       'pl': 'data/vaa/pl_unique.csv',
       'it': 'data/vaa/it_questions.csv'
       }


def run_model(model_type: str,  # llama-2-13b
              temperature: float = 0.6,
              top_p: float = 0.8,
              max_seq_len: int = 4096,
              max_gen_len: int = 50,
              max_batch_size: int = 10,
              seed: int = 1,
              lang: str = 'es',
              interactive: bool = False):
    instruction_path: str = lang + '_instructions.txt'
    output_path: str = VAA[lang].split("/")[-1]
    # Load model
    model = LLamaLoader(model_dir=MODEL_PATH,
                        model_type=model_type,
                        max_seq_len=max_seq_len,
                        max_batch_size=max_batch_size,
                        seed=seed)

    # Load data - VAA
    statements = pd.read_csv(VAA[lang])

    # Merge instructions + VAA
    prompt_format = StatementPrompt()

    responses_df = pd.DataFrame()
    responses = defaultdict(list)

    if interactive:
        # Insert instruction from the command line.
        running = True

        while running:
            instruction = input('Input instruction or STOP to quit. Mark statement placeholder with %s:')

            if instruction == 'STOP':
                running == False
                break

            # Generate prompts
            for sents in ["statement", "negation", "opposite"]:
                prompt = prompt_format.generate_prompt_batch(instruction, statements[sents].tolist())

                # Generate responses
                responses[sents].extend(model.generate_responses(prompt,
                                                                 temperature=temperature,
                                                                 top_p=top_p,
                                                                 max_gen_len=max_gen_len))

            responses["ids"].extend(statements['statementID'].tolist())
            responses["prompts"].extend([instruction] * len(statements['statementID'].tolist()))


    else:
        # Load prompt instructions from .txt file 
        with open("./data/prompt_instructions/" + instruction_path) as file:
            instructions = [line.rstrip() for line in file]

        for instruction in instructions:
            # Generate prompts
            for sents in ["statement", "negation", "opposite"]:
                prompt = prompt_format.generate_prompt_batch(instruction, statements[sents].tolist())

                # Generate responses
                responses[sents].extend(model.generate_responses(prompt,
                                                                 temperature=temperature,
                                                                 top_p=top_p,
                                                                 max_gen_len=max_gen_len))

            responses["ids"].extend(statements['statementID'].tolist())
            responses["prompts"].extend([instruction] * len(statements['statementID'].tolist()))

    for response in responses:
        print(response)
        print('-' * 150)

    responses_df['statementID'] = responses["ids"]
    responses_df['prompt'] = responses["prompts"]
    responses_df['statement'] = responses["statement"]
    responses_df['negation'] = responses["negation"]
    responses_df['opposite'] = responses["opposite"]

    Path("./data/responses").mkdir(parents=True, exist_ok=True)
    responses_df.to_csv(f'data/responses/{output_path}', index=False)
    print('Done!')


if __name__ == "__main__":
    '''
    Usage example: torchrun --nproc_per_node 1 models/run_model.py --model_type 7B --option option_value

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
    fire.Fire(run_model)
