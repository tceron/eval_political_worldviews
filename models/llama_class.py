import fire
import os
import sys
import torch
import json
import pandas as pd

from tqdm import tqdm
from models.llama2.generation import Llama, Dialog
from models.llama.llama import ModelArgs, Transformer, Tokenizer, LLaMA
from typing import List, Optional
from utils.split2batch import split2batch
from abc import ABC, abstractmethod
from typing import Tuple
from pathlib import Path
from fairscale.nn.model_parallel.initialize import initialize_model_parallel

MODEL_DIR = '/mount/arbeitsdaten32/projekte/mardy/baricaa/models/'


class LLM(ABC):
    def __init__(self):
        pass

    def generate(self):
        pass

class LlamaClass(LLM):
    def __init__(self,
                ckpt_dir: str,
                tokenizer_path: str,
                max_seq_len: int = 512,
                max_batch_size: int = 3,
                seed: int = 1):
        
        local_rank, world_size = self.setup_model_parallel(seed)
        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))

        assert world_size == len(checkpoints), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
        
        ckpt_path = checkpoints[local_rank]

        checkpoint = torch.load(ckpt_path, map_location="cpu")

        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params)

        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = Transformer(model_args)
        torch.set_default_tensor_type(torch.FloatTensor)
        model.load_state_dict(checkpoint, strict=False)

        self.generator = LLaMA(model, tokenizer)
        self.batch_size = max_batch_size

    def setup_model_parallel(self, seed) -> Tuple[int, int]:
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        world_size = int(os.environ.get("WORLD_SIZE", -1))

        torch.distributed.init_process_group("nccl")
        initialize_model_parallel(world_size)
        torch.cuda.set_device(local_rank)

        # seed must be the same in all processes
        torch.manual_seed(seed)
        return local_rank, world_size

    def generate(self, prompts: List[str], 
                 temperature: float = 0.6, 
                 top_p: float = 0.9, 
                 max_gen_len: int = 100):

        # Get batch indices
        batch_ids = split2batch(self.batch_size, len(prompts))

        # Init results list
        results = []

        # Iterate over batches
        for (begin, end) in tqdm(batch_ids):
            prompt_batch = prompts[begin:end]

            batch_results = self.generator.generate(prompt_batch,
                                                    max_gen_len=max_gen_len,
                                                    temperature=temperature,
                                                    top_p=top_p)
            
            results.extend([x for x in batch_results])

        return results


class LLama2Class(LLM):
    def __init__(self, 
            ckpt_dir: str, 
            tokenizer_path: str, 
            max_seq_len: int = 2096, 
            max_batch_size: int = 4,
            seed : int = 1
            ):
        
        self.generator = Llama.build(ckpt_dir=ckpt_dir,
                                     tokenizer_path=tokenizer_path,
                                     max_seq_len=max_seq_len,
                                     max_batch_size=max_batch_size,
                                     seed = seed
                                ) 
        
        self.batch_size = max_batch_size

    @abstractmethod
    def generate(self):
        pass

class LLama2TxtGen(LLama2Class):      

    def generate(self, 
                 prompts: List[str], 
                 temperature: float = 0.6, 
                 top_p: float = 0.9, 
                 max_gen_len: int = 5) -> List[str]:

            # Get batch indices
            batch_ids = split2batch(self.batch_size, len(prompts))

            # Init results list
            results = []

            # Iterate over batches
            for (begin, end) in tqdm(batch_ids):
                prompt_batch = prompts[begin:end]

                batch_results = self.generator.text_completion(prompt_batch, 
                                                        max_gen_len=max_gen_len, 
                                                        temperature=temperature, 
                                                        top_p=top_p)
                
                results.extend([x['generation'] for x in batch_results])

            return results
    
    
class LLama2Chat(LLama2Class):  

    def generate(self, 
                prompts: List[Dialog], 
                temperature: float = 0.6, 
                top_p: float = 0.9, 
                max_gen_len: Optional[int] = 50) -> List[str]:
        
        # Get batch indices
        batch_ids = split2batch(self.batch_size, len(prompts))

        # Init results list
        results = []

        # Iterate over batches
        for (begin, end) in tqdm(batch_ids):
            dialog_batch = prompts[begin:end]

            batch_results = self.generator.chat_completion(dialog_batch, 
                                                    max_gen_len=max_gen_len, 
                                                    temperature=temperature, 
                                                    top_p=top_p
                                                    )
            
            results.append(batch_results)

        # Transform to List[str]
        #results = [result['generation']['content'] for result in results]

        return results
    
class LLamaLoader:
    def __init__(self, model_dir, model_type, **kwargs):
        self.model_dir = model_dir
        self.model = self.load_model(model_type=model_type, **kwargs)
    
    def get_model_path(self, model_type):

        if 'llama-2' in model_type:
            model_path = self.model_dir + 'llama2/' 

        else:
            model_path = self.model_dir + 'llama/'
        
        tokenizer_path = model_path + 'tokenizer.model'
        model_path = model_path + model_type

        return model_path, tokenizer_path
    
    def load_model(self, model_type, **kwargs):
        model_path, tokenizer_path = self.get_model_path(model_type)

        if 'llama-2' in model_path:
            if 'chat' in model_path:
                model = LLama2Chat(ckpt_dir=model_path, tokenizer_path=tokenizer_path, **kwargs)
            else:
                model = LLama2TxtGen(ckpt_dir=model_path, tokenizer_path=tokenizer_path, **kwargs)
        else:
            model = LlamaClass(ckpt_dir=model_path, tokenizer_path=tokenizer_path, **kwargs)

        return model

    def generate_responses(self, prompts, **kwargs):
        return self.model.generate(prompts, **kwargs)
    

if __name__ == '__main__':

    # Usage example:

    # Model and tokenizer path
    model_type = 'llama-2-7b'

    # Init model context
    model = LLamaLoader(model_type, max_seq_len=200)

    running = True
    while running:
        # Load data
        prompt = input('Insert your prompt to run the model or STOP to quit:')

        if prompt == 'STOP':
            running == False
            break

        prompts = [prompt]

        # Run model inference
        responses = model.generate_responses(prompts, temperature=0.9)

        print(responses)
    
    print('Done!')