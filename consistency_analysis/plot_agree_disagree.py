import pandas as pd
import glob
from utils import sorter_models, models_renamed, colors_agree_disagree
import matplotlib.pyplot as plt
import numpy as np
import argparse

def number_agree_disagree_models():
    """ Create dataframe with number of times models have agreed or disagreed per policy domain."""

    files = glob.glob("./data/responses/updated/*.csv")
    results = []

    for f in files:
        model_name = models_renamed[f.split("/")[-1].split("_")[0]]
        template_id = int(f.split("/")[-1].split("_")[1])
        df = pd.read_csv(f)
        if args.passed_test and (model_name not in ['alwaysDISagree', 'alwaysAgree', 'RANDOM']):
            df = df[df[args.passed_test] == True]
        df["answer"]=df.apply(lambda x: -1 if x["string_answers"][10] == "n" else int(x["string_answers"][10]), axis=1)

        model_answers = dict(zip(df["unique_id"].tolist(), df.answer.tolist()))
        for unique_id in model_answers.keys():
            # for policy in id2policies[unique_id]:
            if model_answers[unique_id] == 1:
                results.append((model_name, template_id, "agree", len(model_answers.keys())))
            elif model_answers[unique_id] == 0:
                results.append((model_name, template_id, "disagree", len(model_answers.keys())))
    df = pd.DataFrame(results, columns=["model", "template_id", "stance", "n_unique_ids"])
    return df

def plot_number_agree_disagree_models():
    """ Generate plot with number of times models have agreed or disagreed"""

    df = number_agree_disagree_models()
    df["counts"] = df.groupby(["model", "template_id", "n_unique_ids", "stance"])["stance"].transform("count")
    df = df.drop_duplicates(subset=["model", "template_id", "stance", "n_unique_ids"])
    df["norm_count"] = df.apply(lambda x: 100*(x.counts/x.n_unique_ids), axis=1)  # divide by the total number of unique ids = 239

    df = df.set_index("model")
    df = df.loc[sorting]
    df = df.reset_index()

    # If only one test is present, axes is not a list, so we convert it into a list for consistency.
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))

    offset = 0.2
    for model in sorting:
        tmp_df = df[df['model'] == model]

        for case in ['agree', 'disagree']:

            mean_value = np.mean(tmp_df[tmp_df["stance"]==case].norm_count.tolist())
            std_value = np.std(tmp_df[tmp_df["stance"]==case].norm_count.tolist())
            model_index = sorting.index(model)

            if case == 'agree':
                position = model_index
            else:
                position = model_index - offset

            print(model, case, round(mean_value, 2))

            ax.errorbar(mean_value, position, xerr=std_value, fmt='o',
                        color=colors_agree_disagree[case], ecolor='lightgray', elinewidth=2, capsize=3, label=case)

    ax.set_yticks(np.arange(len(sorting)) - offset / 2)  # Adjust the ticks to be in between the two cases
    ax.set_yticklabels(sorting)
    # ax.set_xlim(0, 200)
    ax.invert_yaxis()  ## O Invert y-axis to match the provided plot
    if args.passed_test:
        ax.set_title(f'Agree/Disagree - reliable statements')
    else:
        ax.set_title(f'Agree/Disagree - all statements')

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='lower center', bbox_to_anchor=(-0.2, -0.16), fontsize=9,
          fancybox=True, shadow=True) #borderaxespad=-0.55
    ax.set_xlabel('% answer type')

    plt.tight_layout()
    # Save figure
    plt.savefig(f'data/responses/plots_paperv2/num_agrees_disagrees_{args.passed_test}.jpeg', dpi=300)
    plt.show()
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--passed_test", type=str, default=False)
    parser.add_argument("--simulation", action='store_true')
    args = parser.parse_args()

    if args.passed_test and not args.simulation:
        sorting = sorter_models['only_models']
    if not args.passed_test and not args.simulation:
        sorting = sorter_models['only_models']
    if args.simulation:
        sorting = sorter_models['simulations']

    plot_number_agree_disagree_models()