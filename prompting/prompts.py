import re
import pandas as pd
from utils.string_formatting import *


class Prompt:
    def __init__(self):
        pass

    def generate_prompt(self):
        pass

    def generate_prompt_batch(self):
        pass

    def format_prompt(self):
        pass

    def insert_input(self, prompt, input_demo):
        return prompt.format(input_demo)


class StatementPrompt(Prompt):
    def __init__(self):
        pass

    def generate_prompt(self, instruction=None, statement=''):
        if instruction:
            return instruction % statement

        else:
            return statement

    def generate_prompt_batch(self, instruction, statements):
        prompts = []
        for s in statements:
            prompts.append(self.generate_prompt(instruction, s))

        return prompts

    def format_prompt(self):
        # TODO
        pass


class FewShotPrompt(Prompt):
    def __init__(self):
        super().__init__()

    def generate_prompt(self, instruction=None, demos=[]):

        prompt = f''

        if instruction:
            prompt += '### Instruction: ' + instruction + '\n\n'

        for demo in demos:
            prompt += '### Input: ' + demo[0] + '\n'
            prompt += '### Response: ' + demo[1] + '\n\n'

        prompt += '### Input: {}\n'
        #prompt += '### Response: '

        return prompt
    
    # def generate_prompt_batch(self, demos: [[str]],
    #                           responses: [[str]],
    #                           input_demos: [str],
    #                           input_ids: [int]=None,
    #                           instruction=None):
    #
    #     """
    #     Generates batch prompts.
    #
    #     Returns: [(input_id, prompt)]
    #     """
    #
    #     demo_list = list(zip(demos, responses))
    #     prompts = []
    #
    #     for i in range(len(input_demos)):
    #         prompt = self.insert_input(self.generate_prompt(instruction, demo_list[i]), input_demos[i])
    #         prompts.append(prompt)
    #
    #
    #     prompts = list(zip(input_ids, prompts))
    #
    #     return prompts
    
    
    def format_response(responses: [str]):

        formatted_responses = []

        for response in responses:
            p = remove_multiple_whitespace(remove_newline(response))

            if '### Response: ' in p:
                p = p.split('### Response: ')[1]

            p = remove_char(p, '#')

            if 'Input' in p:
                p = p.split('Input')[0]
            

            if len(p) <= 1:
                p = ' '

            formatted_responses.append(p)
        
        return formatted_responses

class DialogPrompt(Prompt):
    def __init__(self):
        super().__init__()
    
    def generate_prompt(self):
        pass

    def generate_prompt_batch(self):
        pass

    def format_prompt(self):
        pass
