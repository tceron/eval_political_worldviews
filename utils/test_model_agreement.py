import os
from glob import glob
import numpy as np
import pandas as pd


path_to_csv = os.path.join('..', 'model_outputs_combined', '*.csv')
table_paths = glob(path_to_csv)
df_arr = []
for path in table_paths:
    df_arr.append(pd.read_csv(path))
all_data = pd.concat(df_arr, axis=0, ignore_index=True)
print('Before removing NAs:', all_data.shape[0], 'rows.')
all_data = all_data.iloc[:, 2:].dropna().astype(int)
all_data.columns = list(map(lambda colname: colname.split('/')[1], 
                            all_data.columns))
print('After removing NAs:', all_data.shape[0], 'rows.')
print(all_data.corr().round(2))
