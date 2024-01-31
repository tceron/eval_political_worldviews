import os
from glob import glob
from collections import defaultdict, Counter
import pandas as pd


def get_two_col_dict(df, col1, col2):
    "Assumes that all elements in col1 are distinct."
    return dict(zip(df[col1], df[col2]))


def load_df(path):
    country_name = os.path.split(path)[-1].split('.')[0]
    tmp_df = pd.read_csv(path)
    tmp_df['statement_id'] = tmp_df.statement_id.map(lambda sid: f'{country_name}_{sid}')
    return tmp_df


path_to_csv_pos = os.path.join('..', 'model_outputs_combined', '*.csv')
table_paths_pos = glob(path_to_csv_pos)
path_to_csv_neg = os.path.join('..', 'model_outputs_neg_combined', '*.csv')
table_paths_neg = glob(path_to_csv_neg)

df_arr_pos = []
for path in table_paths_pos:
    df_arr_pos.append(load_df(path))
all_data_pos = pd.concat(df_arr_pos, axis=0, ignore_index=True)
    
df_arr_neg = []
for path in table_paths_neg:
    df_arr_neg.append(load_df(path))
all_data_neg = pd.concat(df_arr_neg, axis=0, ignore_index=True)

print(all_data_pos.shape, all_data_neg.shape)

model_names_pos = all_data_pos.columns[2:]
model_names_neg = all_data_neg.columns[2:]
assert (model_names_neg == model_names_pos).all()

results = defaultdict(Counter)
for model in model_names_neg:
    model_dict_pos = get_two_col_dict(all_data_pos, 'statement_id', model)
    model_dict_neg = get_two_col_dict(all_data_neg, 'statement_id', model)
    for key in model_dict_pos:
        pos_answer = model_dict_pos[key]
        neg_answer = model_dict_neg[key]
        if pd.isnull(pos_answer) or pd.isnull(neg_answer):
            continue
        if pos_answer == neg_answer:
            results[model]['miss'] += 1
        else:
            results[model]['hit'] += 1

print(pd.DataFrame.from_records([
    (model, stats['hit'], stats['miss'],
          round(stats['hit'] / (stats['hit'] + stats['miss']), 3) * 100)
    for model, stats in results.items()
], columns=['Model', 'DifferentAnswer', 'SameAnswer', 'DifferentAnswerPercentage']))
