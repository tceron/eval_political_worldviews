import pandas as pd
import glob
from utils import parties_ches, convert_unique_id, sorter_models, models_renamed, positions, policy_colors, policy_names, color_models, policy2count, id2idx, policy2idx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import matplotlib.gridspec as gridspec

def most_biased_domains(country):
    df = pd.read_csv("./data/annotations/spiderweb/annotations_vaas_gold.csv").fillna(0)
    df["ID"] = df.ID.apply(lambda x: int(x.split("_")[1]))
    id2policy_domain = dict(zip(df.ID.tolist(), df.policy_domain.tolist()))
    files = glob.glob("./data/responses/dataframes_withpasses/*.csv")
    results = []

    for f in files:
        model_name = models_renamed[f.split("/")[-1].split("_")[0]]
        template_id = f.split("/")[-1].split("_")[1]
        df = pd.read_csv(f)
        tmp = df[df["country_code"] == country]

        tmp["unique_id"] = tmp.unique_id.apply(lambda x: convert_unique_id(x))
        tmp["answer"]=tmp.apply(lambda x: -1 if x["hard_pass"] == False else int(x["string_answers"][10]), axis=1) # -1 if it has not passed all models.
        unique_ids = tmp.unique_id.unique().tolist()
        tmp2 = tmp[tmp["hard_pass"] == True]

        model_answers = dict(zip(tmp2["unique_id"].tolist(), tmp2.answer.tolist()))

        for unique_id in model_answers.keys():
            answer = model_answers[unique_id]
            results.append((country, model_name, template_id, id2policy_domain[unique_id], answer, len(unique_ids)))
    df = pd.DataFrame(results, columns=["country", "model", "template_id", "policy_domain", "answer", "n_unique_ids"])

    return df


def matches_domain_specific():
    df = pd.DataFrame()
    for c in list(parties_ches.keys()):
        tmp = most_biased_domains(c)
        df = pd.concat([df, tmp], ignore_index=True)

    sum_ids = sum([v for v in dict(zip(df.country.tolist(), df.n_unique_ids.tolist())).values()])

    # Get the counts, including zero counts
    counts = df.groupby(["template_id", "model", "policy_domain"]).size()

    # Create a DataFrame with all possible categories and fill NaN with 0
    df = counts.unstack(fill_value=0).stack().reset_index(name='n_stats_pol')

    df = df.drop_duplicates(subset=["template_id", "model", "policy_domain"])
    df["norm_count"] = df.apply(lambda x: 100*(x.n_stats_pol/sum_ids), axis=1)  # divide by the total number of unique ids = 239

    df = df.set_index("model")
    df = df.loc[sorter_models]
    df = df.reset_index()

    # If only one test is present, axes is not a list, so we convert it into a list for consistency.
    fig, ax = plt.subplots() #figsize=(5, 8)

    for (model, case), group in df.groupby(['model', 'policy_domain']):
        print(group)

        mean_value = group['norm_count'].mean()
        std_value = group['norm_count'].std()
        model_index = sorter_models.index(model)

        ax.errorbar(mean_value,  model_index + positions[case], xerr=std_value, fmt='o',
                    color=policy_colors[case], ecolor='lightgray', elinewidth=2, capsize=3, label=case)

    ax.set_yticks(np.arange(len(sorter_models)) - positions[case] / 2)  # Adjust the ticks to be in between the two cases
    ax.set_yticklabels(sorter_models)
    ax.invert_yaxis()  ## O Invert y-axis to match the provided plot
    # ax.grid(True)
    ax.set_title(f'Relative number of policy domains across templates')
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='center left',bbox_to_anchor=(1.0, 0.5))
    ax.set_xlabel('Mean percentage of statements across templates')

    plt.tight_layout()
    # Save figure
    plt.savefig(f'data/responses/policy_domain_norm.jpeg', dpi=600)
    plt.show()
    plt.close()

# matches_domain_specific()


def stance_count():
    df = pd.read_csv("./data/annotations/spiderweb/annotations_vaas_gold.csv").fillna(0)
    df["ID"] = df.ID.apply(lambda x: "".join(x.split("_")[:2]))

    position2ids = defaultdict(list)

    for col in df.columns[3:]:
        dic = dict(zip(df.ID.tolist(), df[col].tolist()))
        for id in dic.keys():
            if dic[id] == 1:
                position2ids[col].append(id)
            elif dic[id] == -1:
                position2ids[col].append(id)
    print({k:len(v) for k, v in position2ids.items()})

def position_per_statement():
    df = pd.read_csv("./data/annotations/spiderweb/annotations_vaas_gold.csv").fillna(0)
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
    df = pd.read_csv("./data/annotations/spiderweb/annotations_vaas_gold.csv").fillna(0)
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
    files = glob.glob("./data/responses/dataframes_withpasses/*.csv")

    id2policies, policies = id_to_policy_stance()

    matrices = {model: {} for model in sorter_models}

    for f in files:
        model_name = models_renamed[f.split("/")[-1].split("_")[0]]
        template_id = int(f.split("/")[-1].split("_")[1])
        df = pd.read_csv(f)
        df = df[df["hard_pass"] == True]
        df["answer"] = df.apply(lambda x: int(x["string_answers"][10]), axis=1)  # -1 if it has not passed all models.

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
    df = pd.read_csv("./data/annotations/spiderweb/annotations_vaas_gold.csv").fillna(0)
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


def num_category_annotated():
    df = pd.read_csv("./data/annotations/spiderweb/annotations_vaas_gold.csv").fillna(0)
    results = {}
    for col in df.columns[3:]:
        dic = dict(zip(df.ID.tolist(), df[col].tolist()))
        m = {"agree": 0, "disagree": 0}
        for id in dic.keys():
            if dic[id] == 1:
                m["agree"]+=1
            elif dic[id] == -1:
                m["disagree"]+=1
        results[col] = m
    for k, v in results.items():
        print(k, v)
    return results

# num_category_annotated()

def matrix_statement_categories_per_model():

    """ This function creates a matrix with the number of times a model has answered a statement in a certain category."""

    id2policies, policies = id_to_policy_stance()
    files = glob.glob("./data/responses/dataframes_withpasses/*.csv")

    matrices = {model: {} for model in sorter_models}

    for f in files:
        model_name = models_renamed[f.split("/")[-1].split("_")[0]]
        template_id = int(f.split("/")[-1].split("_")[1])
        df = pd.read_csv(f)
        df = df[df["hard_pass"] == True]

        df["answer"] = df.apply(lambda x: -1 if x["hard_pass"] == False else int(x["string_answers"][10]),
                                axis=1)  # -1 if it has not passed all models.
        model_answers = dict(zip(df["unique_id"].tolist(), df.answer.tolist()))

        m = np.zeros([len(policies)])

        for unique_id in model_answers.keys():
            for policy in id2policies[unique_id]:
                m[policy2idx[policy]] += 1  # we don't need to know if positive or negative, only if it is present
        matrices[model_name][template_id] = m

    return matrices


def plot_positioning_per_model():
    # df = positioning_per_model()
    # df = pd.read_csv("./data/responses/policy_stances.csv")

    m_model= positioning_per_model()
    m_count = matrix_statement_categories_per_model()  # total number of answers per policy domain per model

    # If only one test is present, axes is not a list, so we convert it into a list for consistency.
    fig, axes = plt.subplots(nrows=2, ncols=int(len(sorter_models)/2), figsize=(12, 6), subplot_kw=dict(polar=True))
    if len(sorter_models) == 1:
        axes = [axes]

    axes = axes.flatten()

    handles, labels = [], []

    for enu, (ax, model) in enumerate(zip(axes, sorter_models)):

        agreement = defaultdict(list)

        for policy in policy2idx.keys():
            for template_id in m_model[model].keys():

                n_total_model = m_count[model][template_id][policy2idx[policy]]
                n_agree = m_model[model][template_id]["agree"][policy2idx[policy]]
                n_disagree = m_model[model][template_id]["disagree"][policy2idx[policy]]
                print(n_agree, n_disagree, "\t total", n_total_model)

                if n_total_model == 0:
                    agreement[policy].append(0)
                    print("policy domain with 0 stances", policy, model)
                else:
                    agreement[policy].append((n_agree / n_total_model) - (n_disagree / n_total_model))

        temp_df = pd.DataFrame(agreement).T
        # take mean and std of rows
        temp_df["mean"] = temp_df.mean(axis=1)
        temp_df["std"] = temp_df.std(axis=1)
        print(temp_df)

        # Sort always by the same order
        temp_df = temp_df.loc[policy_names.keys()]
        temp_df = temp_df.reset_index().rename(columns={'index': 'policy_stance'})

        # print(temp_df)

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

    # plt.legend(by_label.values(), by_label.keys(), loc='center left', fontsize=8)

    plt.tight_layout()
    plt.savefig(f'data/responses/policy_stance_positioning1.jpeg', dpi=300)
    plt.show()

plot_positioning_per_model()

def number_agree_disagree_models():

    files = glob.glob("./data/responses/dataframes_withpasses/*.csv")
    id2policies, policies = id_to_policy_stance()
    results = []

    for f in files:
        model_name = models_renamed[f.split("/")[-1].split("_")[0]]
        template_id = int(f.split("/")[-1].split("_")[1])
        df = pd.read_csv(f)
        df = df[df["hard_pass"] == True]
        df["answer"] = df.apply(lambda x: int(x["string_answers"][10]), axis=1)  # -1 if it has not passed all models.

        model_answers = dict(zip(df["unique_id"].tolist(), df.answer.tolist()))
        for unique_id in model_answers.keys():
            # for policy in id2policies[unique_id]:
            if model_answers[unique_id] == 1:
                results.append((model_name, template_id, "agree", len(model_answers.keys())))
            elif model_answers[unique_id] == 0:
                results.append((model_name, template_id, "disagree", len(model_answers.keys())))
    df = pd.DataFrame(results, columns=["model", "template_id", "stance", "n_unique_ids"])
    return df

colors_agree_disagree = {"agree": "skyblue", "disagree": "lightcoral"}
def plot_number_agree_disagree_models():

    df = number_agree_disagree_models()
    df["counts"] = df.groupby(["model", "template_id", "n_unique_ids", "stance"])["stance"].transform("count")
    df = df.drop_duplicates(subset=["model", "template_id", "stance", "n_unique_ids"])
    df["norm_count"] = df.apply(lambda x: 100*(x.counts/x.n_unique_ids), axis=1)  # divide by the total number of unique ids = 239

    df = df.set_index("model")
    df = df.loc[sorter_models]
    df = df.reset_index()

    # If only one test is present, axes is not a list, so we convert it into a list for consistency.
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))

    offset = 0.2
    for model in sorter_models:
        tmp_df = df[df['model'] == model]

        for case in ['agree', 'disagree']:

            mean_value = np.mean(tmp_df[tmp_df["stance"]==case].norm_count.tolist())
            std_value = np.std(tmp_df[tmp_df["stance"]==case].norm_count.tolist())
            model_index = sorter_models.index(model)

            if case == 'agree':
                position = model_index
            else:
                position = model_index - offset

            print(model, case, round(mean_value, 2))

            ax.errorbar(mean_value, position, xerr=std_value, fmt='o',
                        color=colors_agree_disagree[case], ecolor='lightgray', elinewidth=2, capsize=3, label=case)

    ax.set_yticks(np.arange(len(sorter_models)) - offset / 2)  # Adjust the ticks to be in between the two cases
    ax.set_yticklabels(sorter_models)
    # ax.set_xlim(0, 200)
    ax.invert_yaxis()  ## O Invert y-axis to match the provided plot
    ax.set_title(f'Relative number of answer type')

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='lower center', bbox_to_anchor=(-0.2, -0.16), fontsize=9,
          fancybox=True, shadow=True) #borderaxespad=-0.55
    ax.set_xlabel('% answer type')

    plt.tight_layout()
    # Save figure
    plt.savefig(f'data/responses/num_agrees_disagrees.jpeg', dpi=300)
    plt.show()
    plt.close()


# plot_number_agree_disagree_models()