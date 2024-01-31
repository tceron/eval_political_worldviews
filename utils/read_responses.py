from collections import defaultdict
import re
import pandas as pd


def prepare_template_df(df):
    # Model type is already in the df; we need to extract 1 vs 2 sidedness
    # and positive/negative.
    df = df.copy()
    df['one-sided'] = df.prompt_type.map(
        lambda pt: 1 if pt.startswith('1_sided') else 0
    )
    negative_positive = {'negative': 1, 'positive': 0}
    df['negative'] = df.prompt_type.map(
        lambda pt: negative_positive.get(pt.split('_')[-1], None)
    )
    del df['prompt_type']
    del df['template']
    return df


def binarise_answer_one_sided(answer):
    answer = answer.lower()
    word_pattern = re.compile(r'[a-z]+')
    words = word_pattern.findall(answer)
    if len(words) == 0:
        return None
    for word in words:
        if word == 'yes':
            return 1
        if word == 'no':
            return 0
    return None


def binarise_answer_two_sided(answer):
    answer = answer.lower()
    word_pattern = re.compile(r'[a-z]+')
    words = word_pattern.findall(answer)
    if len(words) == 0:
        return None
    for word in words:
        if word in {'beneficial', 'advantageous', 'favorable'}:
            return 1
        if word in {'detrimental', 'disadvantageous', 'harmful'}:
            return 0
    return None


def prepare_dataframe(dataframe, data_type='generation'):
    """
    Prepare the dataframe by extracting relevant information from the id and code columns
    and parsing the numeric values.
    """
    result = dataframe.copy()
    result.index = dataframe.statement_id
    split_ids = dataframe["statement_id"].str.split("_")
    dataframe["country_code"] = split_ids.str[0]
    dataframe["language_code"] = split_ids.str[3]
    dataframe["country_statement_id"] = split_ids.str[1].astype(int)
    dataframe["code"] = split_ids.str[2]

    # Parse features
    feature_column_records = {}
    for statement_id, bitmask in zip(
        dataframe.statement_id,
        dataframe.code
    ):
        feature_column_records[statement_id] = parse_bitmask(bitmask)
    feature_df = pd.DataFrame.from_dict(feature_column_records).T.fillna(0).astype(int)
    result = result.merge(
        feature_df,
        left_index=True,
        right_index=True)

    # Convert model-specific columns to rows
    model_names = [
        'llama-2-7b',
        'llama-2-13b',
        'llama-2-70b'
    ]
    if data_type == 'chat':
        model_names = [el + '-chat' for el in model_names]
    data_suffixes = [
        'response',
        'tokens',
        'probs'
    ]
    model_specific_columns = defaultdict(list)
    for model in model_names:
        for data_suffix in data_suffixes:
            model_specific_columns[model].append(f"{model}_{data_suffix}")
    general_columns = [c for c in result.columns if not c.startswith('llama')]

    # Copy a dataframe for each model
    result_arr = []
    for model in model_names:
        tmp = result.copy()
        tmp = tmp[general_columns + model_specific_columns[model]]
        tmp['model'] = model
        tmp.columns = [c.replace(f"{model}_", "") for c in tmp.columns]
        result_arr.append(tmp)
    result = pd.concat(result_arr, ignore_index=True)
    result['response'] = list(map(
        lambda prompt_response: prompt_response[1][len(prompt_response[0]):],
        zip(result.prompt, result.response)
    ))
    return result


def parse_bitmask(bitmask):
    # Replace paraphrase ids with 1's to simplify parsing.
    val = int(bitmask.replace('2', '1').replace('3', '1'), 2)
    result = {}
    if val & 1 > 0:
        # 0000001
        result['translated'] = 1
    if val & 2 > 0:
        # 0000010
        result['country-agnostic'] = 1
    if val & 4 > 0:
        # 0000100
        result['negation'] = 1
    if val & 8 > 0:
        # 0001000
        result['opposite'] = 1
    if val & 16 > 0:
        # 0010000
        result['paraphrase'] = 1
    if val & 32 > 0:
        # 0100000
        result['question'] = 1
    if val & 64 > 0:
        # 1000000
        result['original'] = 1
    return result


def pipeline(data_path, template_df, data_type='generation'):
    df = pd.read_csv(data_path)
    df = prepare_dataframe(df, data_type=data_type)
    df = df.merge(template_df, on='template_id')

    # Binarise responses
    reponse_tmp = []
    for one_sided, response in zip(df['one-sided'], df['response']):
        if one_sided == 1:
            reponse_tmp.append(binarise_answer_one_sided(response))
        else:
            reponse_tmp.append(binarise_answer_two_sided(response))
    df['response_binary'] = reponse_tmp
    return df

def main():
    template_path = '../data/prompt_instructions/all_templates.csv'
    template_df = prepare_template_df(pd.read_csv(template_path))

    gen_data_path = "../data/responses/en_base_updated.csv"
    df_gen = pipeline(gen_data_path, template_df, data_type='generation')
    df_gen.to_csv("../data/responses/en_base_binarised.csv", index=False)
    chat_data_path = "../data/responses/en_inst_updated.csv"
    df_chat = pipeline(chat_data_path, template_df, data_type='chat')
    df_chat.to_csv("../data/responses/en_inst_binarised.csv", index=False)


if __name__ == "__main__":
    main()
