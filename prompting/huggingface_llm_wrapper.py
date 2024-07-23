from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import transformers
import torch
from tqdm import tqdm
import numpy as np

from optimum.bettertransformer import BetterTransformer

class HuggingFaceLLM:
    def __init__(self, model: str, **kwargs):
        pass
        
    def generate(self, prompts: [str], **kwargs):

        # Generated textual response
        responses = []

        # Token IDs of the generated textual response (they are needed for probability mapping)
        generated_tokens = []
        
        # Probabilities of each generated token
        probs = []
        
        for prompt in tqdm(prompts, total=len(prompts)):
            inputs = self.tokenizer(prompt, return_tensors='pt') # padding=True
            inputs = {k: v.cuda() for k, v in inputs.items()}

            # Generate response
            outputs = self.model.generate(**inputs, **kwargs, return_dict_in_generate=True, output_scores=True)
            

            # Decode outputs
            input_length = 1 if self.model.config.is_encoder_decoder else inputs['input_ids'].shape[1]
            
            generated_token = outputs.sequences[:, input_length:]
            all_tokens = outputs.sequences[:, :]
            
            tokenized_output = self.tokenizer.batch_decode(all_tokens, skip_special_tokens=True)
            
            # Get probabilities on a token level, dim(prob) = (num_return_sequences, max_new_tokens)
            prob = self.model.compute_transition_scores(outputs.sequences, 
                                                        outputs.scores, 
                                                        normalize_logits=True
                                                        )
            
            # Save responses, probs and token IDs
            probs.append(np.exp(prob.cpu().float()))
            responses.append(tokenized_output)
            generated_tokens.append(generated_token.cpu())

        return responses, generated_tokens, probs
    
    
class CausalLLM(HuggingFaceLLM):
    def __init__(self, model: str, **kwargs):
        # padding_side = 'left' for decoder-only (causal) models

        self.model = AutoModelForCausalLM.from_pretrained(model, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model)               # padding_side='left'
        
        # Most LLMs don't have a pad token by default
        #self.tokenizer.pad_token = self.tokenizer.eos_token  

    
class Seq2SeqLLM(HuggingFaceLLM):
    def __init__(self, model: str, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model, **kwargs)


def hf_loader(model_dict, **kwargs):
    model_name = model_dict['name']
    model_class = model_dict['type']

    return model_class(model_name, **kwargs)



        

