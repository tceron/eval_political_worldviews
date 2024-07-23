from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import transformers
import torch

class HuggingFaceLLM:
    def __init__(self, model: str, **kwargs):
        pass
        
    def generate(self, prompts: [str], **kwargs):
        
        responses = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors='pt') # padding=True
            inputs = {k: v.cuda() for k, v in inputs.items()}

            outputs = self.model.generate(**inputs, **kwargs)
            output = self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True)
            
            responses.extend(output)

        return responses
    
class CausalLLM(HuggingFaceLLM):
    def __init__(self, model: str, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model)           
        self.model = AutoModelForCausalLM.from_pretrained(model, **kwargs)
    
class Seq2SeqLLM(HuggingFaceLLM):
    def __init__(self, model: str, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model, **kwargs)


def hf_loader(model_dict, **kwargs):
    model_name = model_dict['name']
    model_class = model_dict['type']

    return model_class(model_name, **kwargs)



        

