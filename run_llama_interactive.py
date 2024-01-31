import fire
from models.llama_class import *
from prompting.prompts import *

MODEL_PATH = '/mount/arbeitsdaten32/projekte/mardy/baricaa/models/'

def run_model(model_type: str, #llama-2-13b
              temperature: float = 0.6,
              top_p: float = 0.8,
              max_seq_len: int = 4096,
              max_gen_len: int = 50,
              max_batch_size: int = 10,
              seed: int = 1,
              instruction_path : str = 'data/prompt_instructions/es_instructions.txt'
              ):

    # Load model
    model = LLamaLoader(model_dir = MODEL_PATH,
                        model_type=model_type, 
                        max_seq_len=max_seq_len,
                        max_batch_size=max_batch_size, 
                        seed=seed)
    
    running = True
    while running:

        stop_sig = input('Press Enter to continue or stop to quit: ')
        if stop_sig == 'stop':
            running = False
            break

        # Load prompt instructions from .txt file 
        with open(instruction_path) as file:
            instructions = [line.rstrip() for line in file]
        
        # Generate responses
        responses = model.generate_responses(instructions,
                                            temperature=temperature,
                                            top_p = top_p,
                                            max_gen_len = max_gen_len)
        
        for response in responses:
            print(response)
            print()

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