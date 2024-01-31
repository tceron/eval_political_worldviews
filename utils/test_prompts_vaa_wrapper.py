import os
from itertools import product
import subprocess
from tqdm.auto import tqdm

country_names = [
    'the Czech Republic',
    'Germany',
    'Spain',
    'Italy',
    'the Netherlands',
    'Poland'
]

filenames = [
    'cz_unique.csv',
    'de_unique.csv',
    'es_unique.csv',
    'it_unique.csv',
    'nl_unique.csv',
    'pl_unique.csv'
]

path_to_csvs = os.path.join('..', 'data', 'vaa')

model_names = [
    'facebook/opt-1.3b',
    'facebook/opt-2.7b',
    'facebook/opt-6.7b',
    'facebook/opt-13b',
    'facebook/opt-30b'
]

files_and_countries = list(zip(filenames, country_names))
all_params = list(product(model_names, files_and_countries))
for model_name, (filename, country_name) in tqdm(all_params):
    print(model_name, filename, country_name)
    subprocess.run([
        'python', 
        'test_prompts_vaa.py',
        model_name, 
        os.path.join(path_to_csvs, filename), 
        country_name
    ])
