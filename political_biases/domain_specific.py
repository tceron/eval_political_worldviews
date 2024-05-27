import pandas as pd
import glob
from utils import sorter_models, models_renamed, policy_names, policy2count, id2idx, policy2idx, color_models, colors_agree_disagree, test_names
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import argparse


def position_per_statement():
    df = pd.read_csv("../data/human_annotations/annotations_spiderweb_gold.csv").fillna(0)
    df["ID"] = df.ID.apply(lambda x: "".join(x.split("_")[:2]))

    m = np.zeros([len(df.columns[3:]), len(df.ID.tolist())])

    for pol in policy2idx.keys():
        dic = dict(zip(df.ID.tolist(), df[pol].tolist()))
        for id in id2idx.keys():
            if dic[id] == 1:
                m[policy2idx[pol], id2idx[id]] = 1
            elif dic[id] == -1:
                m[policy2idx[pol], id2idx[id]] = -1
    return m

def id_to_policy_stance():
    df = pd.read_csv("../data/human_annotations/annotations_spiderweb_gold.csv").fillna(0)
    df["ID"] = df.ID.apply(lambda x: "".join(x.split("_")[:2]))
    id2policies = defaultdict(list)

    for id in df.ID.tolist():
        for col in df.columns[3:]:
            if df[col][df.ID == id].values[0] == 1:
                id2policies[id].append(col)
            elif df[col][df.ID == id].values[0] == -1:
                id2policies[id].append(col)
            else:
                pass
    return id2policies, df.columns[3:]

def positioning_per_model():

    m_annotation = position_per_statement()
    files = glob.glob("../data/responses/dataframes_withpasses/*.csv")

    id2policies, policies = id_to_policy_stance()

    matrices = {model: {} for model in sorting}

    for f in files:
        model_name = models_renamed[f.split("/")[-1].split("_")[0]]
        if model_name in sorting:
            template_id = int(f.split("/")[-1].split("_")[1])
            df = pd.read_csv(f)

            if args.passed_tests and (model_name not in ['alwaysDISagree', 'alwaysAgree', 'RANDOM']):
                df = df[df[args.passed_tests].all(axis=1)]
            df["answer"]=df.apply(lambda x: -1 if x["string_answers"][10] == "n" else int(x["string_answers"][10]), axis=1)

            model_answers = dict(zip(df["unique_id"].tolist(), df.answer.tolist()))
            m_model = {i: np.zeros([len(policy2idx.keys())]) for i in ["agree", "disagree"]}

            for unique_id in model_answers.keys():
                for policy in id2policies[unique_id]:
                    if m_annotation[policy2idx[policy], id2idx[unique_id]] == 1 and model_answers[unique_id] == 1:
                        m_model["agree"][policy2idx[policy]] += 1
                    elif m_annotation[policy2idx[policy], id2idx[unique_id]] == -1 and model_answers[unique_id] == 0:
                        m_model["disagree"][policy2idx[policy]] += 1
            matrices[model_name][template_id] = m_model
    return matrices

def stance_per_statement():
    df = pd.read_csv("../data/human_annotations/annotations_spiderweb_gold.csv").fillna(0)
    df["ID"] = df.ID.apply(lambda x: "".join(x.split("_")[:2]))
    id2policystance = {k: {} for k in df.ID.tolist()}

    for id in df.ID.tolist():
        for col in df.columns[3:]:
            if df[col][df.ID == id].values[0] == 1:
                id2policystance[id][col]=1
            elif df[col][df.ID == id].values[0] == -1:
                id2policystance[id][col]=-1
            else:
                pass
    return id2policystance, df.columns[3:]

def dic_category_annotated():
    df = pd.read_csv("../data/human_annotations/annotations_spiderweb_gold.csv").fillna(0)
    results = {}
    for col in df.columns[3:]:
        dic = dict(zip(df.ID.tolist(), df[col].tolist()))
        m = {"agree": 0, "disagree": 0}
        for id in dic.keys():
            if dic[id] == 1:
                m["agree"] += 1
            elif dic[id] == -1:
                m["disagree"] += 1
        results[col] = m
    return results

def matrix_statement_categories_per_model():

    """ This function creates a matrix with the number of times a model has answered a statement in a certain category."""

    id2policies, policies = id_to_policy_stance()
    files = glob.glob("../data/responses/dataframes_withpasses/*.csv")

    matrices = {model: {} for model in sorting}

    for f in files:
        model_name = models_renamed[f.split("/")[-1].split("_")[0]]
        if model_name in sorting:
            template_id = int(f.split("/")[-1].split("_")[1])
            df = pd.read_csv(f)

            if args.passed_tests and (model_name not in ['alwaysDISagree', 'alwaysAgree', 'RANDOM']):
                df = df[df[args.passed_tests].all(axis=1)]

            df["answer"]=df.apply(lambda x: -1 if x["string_answers"][10] == "n" else int(x["string_answers"][10]), axis=1)

            model_answers = dict(zip(df["unique_id"].tolist(), df.answer.tolist()))
            print(len(model_answers))

            m = np.zeros([len(policies)])

            for unique_id in model_answers.keys():
                for policy in id2policies[unique_id]:
                    m[policy2idx[policy]] += 1  # we don't need to know if positive or negative, only if it is present
            matrices[model_name][template_id] = m

    return matrices

def plot_positioning_per_model():

    m_model= positioning_per_model()
    m_count = matrix_statement_categories_per_model()  # total number of answers per policy domain per model
    annotated_categories = dic_category_annotated()

    # If only one test is present, axes is not a list, so we convert it into a list for consistency.
    if len(sorting) == 6:
        n_rows, n_cols = 2, 3
    elif len(sorting) == 3:
        n_rows, n_cols = 1, 3
    else:
        n_rows, n_cols = 3, 3
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(13, 7), subplot_kw=dict(polar=True))
    if len(sorting) == 1:
        axes = [axes]

    axes = axes.flatten()

    handles, labels = [], []

    for enu, (ax, model) in enumerate(zip(axes, sorting)):

        agreement = defaultdict(list)

        for policy in policy2idx.keys():
            for template_id in m_model[model].keys():

                n_agree = m_model[model][template_id]["agree"][policy2idx[policy]]
                n_disagree = m_model[model][template_id]["disagree"][policy2idx[policy]]

                if n_disagree+n_agree == 0:
                    agreement[policy].append(0)
                    print("policy domain with 0 stances", policy, model)
                else:
                    print(model, policy, n_agree, annotated_categories[policy]["agree"], n_disagree, annotated_categories[policy]["disagree"])
                    agreement[policy].append((n_agree/annotated_categories[policy]["agree"])-(n_disagree/annotated_categories[policy]["disagree"]))

        temp_df = pd.DataFrame(agreement).T
        # take mean and std of rows
        temp_df["mean"] = temp_df.mean(axis=1)
        temp_df["std"] = temp_df.std(axis=1)
        # print(temp_df)

        # Sort always by the same order
        temp_df = temp_df.loc[policy_names.keys()]
        temp_df = temp_df.reset_index().rename(columns={'index': 'policy_stance'})

        means = np.array(temp_df["mean"].tolist())
        std_devs = np.array(temp_df["std"].tolist())

        policy_stances = temp_df.policy_stance.unique()
        # Calculate angle for each axis
        angles = np.linspace(0, 2 * np.pi, len(policy_stances), endpoint=False).tolist()

        # Plot the data with error bars representing standard deviation
        ax.fill_between(angles, means - std_devs, means + std_devs, color=color_models[model], alpha=0.3)
        ax.plot(angles, means, color=color_models[model], linewidth=2, label=model)

        ax.fill_between(np.linspace(0, 2 * np.pi, len(policy_stances), endpoint=False), -0.9, 0, color='gray', alpha=0.3)

        min_stances = 6
        for i, mean in enumerate(means):
            m = np.mean([m_count[model][i] for i in m_count[model].keys()], axis=0)
            if m[policy2idx[policy_stances[i]]] < min_stances :
                ax.plot(angles[i], means[i], color=color_models[model], marker="o", markersize=5)

        # Set labels for each axis
        if enu != 0:
            ax.set_thetagrids(np.degrees(angles), labels=[policy_names[p] for p in temp_df.policy_stance.tolist()])
        else:
            ax.set_thetagrids(np.degrees(angles), labels=["\n".join([policy_names[p], "("+str(policy2count[p])+")"]) for p in temp_df.policy_stance.tolist()])

        ax.set_ylim(ymin=-0.9, ymax=1)
        rticks = [round(i, 1) for i in np.arange(-1, 1.1, 0.5)]
        ax.set_rticks(rticks) #

        # Plot only a few labels of the radial ticks
        rtick_labels = [round(i, 1) if i in [-1, -0.5, 0, 0.5, 1] else '' for i in rticks]
        ax.set_yticklabels(rtick_labels, fontsize=9)

        ax.set_theta_offset(np.pi / 4)
        # Set the radial axis label
        ax.set_rlabel_position(-4)

        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    # Create a dictionary mapping labels to handles
    by_label = dict(zip(labels, handles))

    # Create a legend for the entire plot
    fig.legend(by_label.values(), by_label.keys(), loc='lower center', ncol=len(by_label), fontsize=10, borderaxespad=-0.55)

    plt.tight_layout()
    if args.simulation:
        plt.savefig(f'../data/responses/plots/policy_stance_simulation.jpeg', dpi=300)
        fig.suptitle('Simulations', fontsize=14, y=1)  # Increase y value here
    else:
        fig.suptitle(test_names[test], fontsize=14, y=1)  # And here
        plt.savefig(f'../data/responses/plots/policy_stance_{test}.jpeg', dpi=300)

    plt.show()

def number_agree_disagree_models():
    """ Create dataframe with number of times models have agreed or disagreed per policy domain."""

    files = glob.glob("../data/responses/dataframes_withpasses/*.csv")
    results = []

    for f in files:
        model_name = models_renamed[f.split("/")[-1].split("_")[0]]
        template_id = int(f.split("/")[-1].split("_")[1])
        df = pd.read_csv(f)
        if args.passed_tests and (model_name not in ['alwaysDISagree', 'alwaysAgree', 'RANDOM']):
            df = df[df[args.passed_tests].all(axis=1)]
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
    ax.invert_yaxis()  ## O Invert y-axis to match the provided plot
    ax.set_title(f'Relative number of answer type')

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='lower center', bbox_to_anchor=(-0.2, -0.16), fontsize=9,
          fancybox=True, shadow=True) #borderaxespad=-0.55
    ax.set_xlabel('% answer type')

    plt.tight_layout()
    # Save figure
    pathlib.Path("../data/responses/plots").mkdir(parents=True, exist_ok=True)
    plt.savefig(f'../data/responses/plots/num_agrees_disagrees.jpeg', dpi=300)
    plt.show()
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--passed_tests", type=str, default=False, help="Hyphen between tests. Filter models that passed specific tests or none of the tests if not called.")
    parser.add_argument("--simulation", action='store_true', help="Plot for random answers, always agree and always disagree.")
    args = parser.parse_args()

    if not args.simulation:
        for test in list(test_names.keys()):
            if test:
                args.passed_tests = test.split("-")
            else:
                args.passed_tests = False
            if args.passed_tests and not args.simulation:
                sorting = sorter_models['only_models']
            if not args.passed_tests and not args.simulation:
                sorting = sorter_models['only_models']
            plot_positioning_per_model()
    else:
        sorting = sorter_models['simulations']
        plot_positioning_per_model()

    # plot_number_agree_disagree_models()