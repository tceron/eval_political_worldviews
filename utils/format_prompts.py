import pandas as pd
from itertools import product

def format_prompts(templates: pd.DataFrame, statements: pd.DataFrame, source_prompts: True):
     if source_prompts: 
        return format_src_prompts(templates, statements)
     else:
         return format_translation_prompts(templates, statements)
         


def format_src_prompts(templates: pd.DataFrame, statements: pd.DataFrame):

    languages = list(set(statements['language_code'].tolist()))
    responses = pd.DataFrame(columns=['template_id', 'statement_id', 'prompt'])

    for lang in languages:
        template_lang = templates[templates['lang'] == lang]
        statement_lang = statements[statements['language_code'] == lang]

        total_prompts = len(list(product(template_lang['template_id'], statement_lang['ID'])))
        total_data = total_prompts * 30

        print(lang, ':', len(template_lang), '|', len(statement_lang), '|', total_prompts, '|', total_data)

        responses_lang = pd.DataFrame(columns=['template_id', 'statement_id', 'prompt'])
        templates_ids = []
        statement_ids = []
        prompts = []
    
        for zipped in product(template_lang['final_template_id'], statement_lang['ID']):
            template_id = templates[templates['final_template_id'] == zipped[0]]['template_id'].values[0]
            template = templates[templates['final_template_id'] == zipped[0]]['template'].values[0]
            statement = statements[statements['ID'] == zipped[1]]['statement'].values[0]

            prompt = template % statement

            templates_ids.append(template_id)
            statement_ids.append(zipped[1])
            prompts.append(prompt)

        responses_lang['template_id'] = templates_ids
        responses_lang['statement_id'] = statement_ids
        responses_lang['prompt'] = prompts

        responses = pd.concat([responses, responses_lang], ignore_index=True)

    return responses['template_id'].tolist(), responses['statement_id'].tolist(), responses['prompt'].tolist()


def format_translation_prompts(templates: pd.DataFrame, statements: pd.DataFrame):

    templates = templates[templates['lang'] == 'en']

    responses = pd.DataFrame(columns=['template_id', 'statement_id', 'prompt'])
    templates_ids = []
    statement_ids = []
    prompts = []

    total_prompts = len(list(product(templates['final_template_id'], statements['ID'])))
    total_data = total_prompts * 30

    print('translations: ', len(templates), '|', len(statements), '|', total_prompts,'|', total_data )
    
    for zipped in product(templates['final_template_id'], statements['ID']):
        template_id = templates[templates['final_template_id'] == zipped[0]]['template_id'].values[0]
        template = templates[templates['final_template_id'] == zipped[0]]['template'].values[0]
        statement = statements[statements['ID'] == zipped[1]]['statement'].values[0]
        prompt = template % statement

        templates_ids.append(template_id)
        statement_ids.append(zipped[1])
        prompts.append(prompt)

    responses['template_id'] = templates_ids
    responses['statement_id'] = statement_ids
    responses['prompt'] = prompts

    return responses['template_id'].tolist(), responses['statement_id'].tolist(), responses['prompt'].tolist()