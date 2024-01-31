import pandas as pd
import tabulate
import matplotlib.pyplot as plt

def modify_binary_answer(x):
    if x == "1":
        return 1
    elif x == "0":
        return -1
    else:
        return 0

def compute_final_answer(df):
    """ Compute final answer from the 30 responses of the same template-statement-inverted pair. """
    # df = df[df['binary_answer'] != "not_applied"]
    df['binary_answer'] = df['binary_answer'].apply(lambda x: modify_binary_answer(x))
    # calculate final answer as the mean of the 30 responses from the same template-statement-inverted pair
    df['final_answer'] = df.groupby(['template_id', 'statement_id', 'inverted', "model_name", "template_prompt"])['binary_answer'].transform('mean')
    # df['std'] = df.groupby(['template_id', 'statement_id', 'inverted',  "model_name", "template_prompt"])['binary_answer'].transform('std')
    df = df.drop_duplicates(subset=['template_id', 'statement_id', 'inverted', "model_name", "template_prompt"]).drop(columns = ['binary_answer', "answer"])

    for model in set(df.model_name.tolist()):
        tmp1 = df[(df['inverted']==0)&(df['model_name']==model)]
        tmp2 = df[(df['inverted'] == 1) & (df['model_name'] == model)]
        # print(tabulate.tabulate(tmp[:10], headers='keys', tablefmt='psql'))

        # plot the distrubtions in two subplots
        plt.subplot(1, 2, 1)
        tmp1['final_answer'].hist()
        plt.title(f"{model} - non_inverted")
        plt.subplot(1, 2, 2)
        tmp2['final_answer'].hist()
        plt.title("inverted")
        plt.show()

df = pd.read_csv('../data/responses/final_merged/answers_binary.csv')
compute_final_answer(df)

def pos_neg(l):
    pos = len([i for i in l if i == 1])
    neg = len([i for i in l if i == 0])
    if neg == 0:
        return 0
    else:
        return pos/(pos+neg)

def compute_final_answer_only_positive_negative(df):
    """ Compute final answer from the 30 responses of the same template-statement-inverted pair. """
    df = df[df['binary_answer'] != "not_applied"]
    df["binary_answer"] = df["binary_answer"].astype(int)
    # calculate final answer as the sum of 1s divided by the sum of 1s plus sum of 0s of the 30 responses from the same template-statement-inverted pair
    df['final_answer'] = df.groupby(['template_id', 'statement_id', 'inverted'])['binary_answer'].transform(pos_neg)
    df = df.drop_duplicates(subset=['template_id', 'statement_id', 'inverted']).drop(columns = ['binary_answer', "answer"])
    print(tabulate.tabulate(df, headers='keys', tablefmt='psql'))
    # plot the distribution of the final answers
    df['final_answer'].hist()
    plt.show()

# compute_final_answer_only_positive_negative(df)
