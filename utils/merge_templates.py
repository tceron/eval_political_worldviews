import glob
import pandas as pd

def merge_templates(dir_path:str):
    all_formats = glob.glob(dir_path + '*.txt')

    prompt_formats = pd.DataFrame(columns=['template', 'lang', 'model_type', 'prompt_type'])

    for prompt_path in all_formats:
        
        with open(prompt_path) as f:
            prompts = f.read()

        prompts = [prompt for prompt in prompts.split('\n')]

        prompt_property = prompt_path.split('data/prompt_instructions/')[1].split('.txt')[0]
        
        lang = prompt_property[:2]
        lang = [lang for i in range(len(prompts))]

        model_type = 'chat' if 'chat' in prompt_property else 'generation'
        model_type = [model_type for i in range(len(prompts))]

        prompt_type = '_'.join(prompt_property.split('_')[2:])
        prompt_type = [prompt_type for i in range(len(prompts))]

        temp = pd.DataFrame(list(zip(prompts, lang, model_type, prompt_type)), 
                        columns=['template', 'lang', 'model_type', 'prompt_type'])
        
        prompt_formats = pd.concat([prompt_formats, temp], ignore_index=True)
        

    prompt_formats['template_id'] = prompt_formats.index
    prompt_formats = prompt_formats[['template_id', 'template', 'lang', 'model_type', 'prompt_type']]
    
    prompt_formats.to_csv(dir_path+'all_templates.csv', index=False)
    