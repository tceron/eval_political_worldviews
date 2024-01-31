import os
from glob import glob
from collections import defaultdict
import pandas as pd
from tqdm.auto import tqdm


def binarise_answer(answer: str):
    answer = answer.lower().strip()
    if answer.startswith('yes'):
        return 1
    elif answer.startswith('no'):
        return 0
    else:
        return pd.NA


# datapath = os.path.join('..', 'model_outputs', '*.csv')
datapath = os.path.join('..', 'model_outputs_neg', '*.csv')
file_paths = glob(datapath)
# outpath = os.path.join('..', 'model_outputs_combined')
outpath = os.path.join('..', 'model_outputs_neg_combined')

for path in tqdm(file_paths):
    filename = os.path.split(path)[-1]
    data = pd.read_csv(path)
    # We create a new dataframe where answers by different models are
    # binarised and collected into rows indexed by prompts.
    result = defaultdict(dict)
    for row in data.itertuples():
        answer_binary = binarise_answer(row.output)
        if row.statement_id not in result['statement']:
            result['statement_id'][row.statement_id] = row.statement_id
            result['statement'][row.statement_id] = row.statement
        result[row.model][row.statement_id] = answer_binary
    result_df = pd.DataFrame.from_dict(result)
    result_df.to_csv(os.path.join(outpath, filename), index=False)
