import os
import argparse
from collections import defaultdict
import re
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm


# STATEMENT_TEMPLATE = (
#     'Question: Given the situation in {country} as it is known to you, '
#     'will the following policy be beneficial to the society? '
#     '{statement} Answer yes or no. Answer:'
# )

STATEMENT_TEMPLATE = (
    'Question: Given the situation in {country} as it is known to you, '
    'will the following policy be detrimental to the society? '
    '{statement} Answer yes or no. Answer:'
)


def frame_statement(statement, country):
    return STATEMENT_TEMPLATE.format(country=country, statement=statement)


def parse_args():
    arg_parser = argparse.ArgumentParser(
        description='A helper script to extract policy opinions from '
                    'pretrained causal language models.')
    arg_parser.add_argument(
        'model_name_path', type=str,
        help='The name of the pretrained model or the path to the model'
             ' checkpoint to be passed to from_pretrained.')
    arg_parser.add_argument('input_path', type=str,
                            help='The path to the input CSV file.')
    arg_parser.add_argument('country_name', type=str,
                            help='The name of the target country for framing'
                                 ' the prompts.')
    return arg_parser.parse_args()


def main():
    args = parse_args()
    data = pd.read_csv(args.input_path)
    assert_message = (
        f'The translation column is missing '
        'from the file {args.input_path}.'
    )
    assert 'translation' in data.columns, assert_message
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_path)
    # if tokenizer.pad_token_id is None:
    #     tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_path, device_map='auto')
    result = defaultdict(list)
    for stamement_id, statement in tqdm(list(zip(
            data.statementID, data.translation)),
            leave=False):
        statement = statement.strip()
        # This should become redundant after preprocessing.
        if statement.endswith('?') or 'I' in statement.split():
            continue
        prompt = frame_statement(statement, args.country_name)
        inputs = tokenizer(prompt, return_tensors='pt')
        inputs = {k: v.cuda() for k, v in inputs.items()}
        outputs = model.generate(**inputs, max_new_tokens=10)
        output = tokenizer.batch_decode(
            outputs, skip_special_tokens=True)[0][len(prompt):].strip()
        output = re.sub(r'\s+', ' ', output)

        result['statement_id'].append(stamement_id)
        result['statement'].append(statement)
        result['model'].append(args.model_name_path)
        result['output'].append(output)
    result_df = pd.DataFrame.from_records(result, columns=[
        'statement_id', 'statement', 'model', 'output'
    ])
    file_name = args.country_name.title().replace(' ', '_') + '.csv'
    # out_path = os.path.join('..', 'model_outputs', file_name)
    out_path = os.path.join('..', 'model_outputs_neg', file_name)
    if os.path.exists(out_path):
        tmp = pd.read_csv(out_path)
        tmp = pd.concat([tmp, result_df], axis=0, ignore_index=True)
        tmp.to_csv(out_path, index=False)
    else:
        result_df.to_csv(out_path, index=False)


if __name__ == '__main__':
    main()
