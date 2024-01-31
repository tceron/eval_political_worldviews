import pandas as pd
import glob
from utils import parties_ches, convert_unique_id, sorter_models, models_renamed, positions, policy_colors, policy_names
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

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

def position_per_statement():
    df = pd.read_csv("./data/annotations/spiderweb/annotations_vaas_gold.csv").fillna(0)
    df["ID"] = df.ID.apply(lambda x: "".join(x.split("_")[:2]))

    position2ids = defaultdict(list)

    for col in df.columns[3:]:
        dic = dict(zip(df.ID.tolist(), df[col].tolist()))
        for id in dic.keys():
            if dic[id] == 1:
                position2ids[col+"_agree"].append(id)
            elif dic[id] == -1:
                position2ids[col + "_disagree"].append(id)
    return position2ids


def positioning_per_model():

    position2ids = position_per_statement()
    files = glob.glob("./data/responses/dataframes_withpasses/*.csv")
    results = []

    for f in files:
        model_name = models_renamed[f.split("/")[-1].split("_")[0]]
        template_id = f.split("/")[-1].split("_")[1]
        df = pd.read_csv(f)
        df = df[df["hard_pass"] == True]

        df["answer"] = df.apply(lambda x: -1 if x["hard_pass"] == False else int(x["string_answers"][10]),
                                  axis=1)  # -1 if it has not passed all models.
        model_answers = dict(zip(df["unique_id"].tolist(), df.answer.tolist()))

        for unique_id in model_answers.keys():
            for position in position2ids.keys():
                if unique_id in position2ids[position]:
                    pos = "_".join(position.split("_")[0:-1])
                    if position.endswith("_agree") and model_answers[unique_id] == 1:
                        results.append((model_name, template_id, unique_id[:2], int(unique_id[2:]), pos, 1, model_answers[unique_id]))
                    elif position.endswith("_disagree") and model_answers[unique_id] == 0:
                        results.append((model_name, template_id,  unique_id[:2], int(unique_id[2:]), pos, -1, model_answers[unique_id]))
                    else:
                        results.append((model_name, template_id,  unique_id[:2], int(unique_id[2:]), pos, 0, model_answers[unique_id]))

    df = pd.DataFrame(results, columns=["model", "template_id", "country", "statementID", "policy_domain", "position", "answer_model"])
    print(df)
    df.to_csv("./data/responses/policy_stances.csv", index=False)
    return df

def plot_positioning_per_model():
    df = positioning_per_model()

    models = sorter_models

    # If only one test is present, axes is not a list, so we convert it into a list for consistency.
    fig, axes = plt.subplots(nrows=1, ncols=len(models), figsize=(8 * len(models), 8), subplot_kw=dict(polar=True))
    if len(models) == 1:
        axes = [axes]

    for enu, (ax, model) in enumerate(zip(axes, models)):

        temp_df = df[(df.model == model)]

        # Data for each axis
        temp_df["sum_position"] = temp_df.groupby(["policy_domain", "template_id"])["position"].transform(sum)
        temp_df = temp_df.drop_duplicates(subset=["policy_domain", "template_id"])
        # values = np.array(temp_df.sum_position.tolist())
        temp_df["mean"] = temp_df.groupby(["policy_domain"])["sum_position"].transform("mean")
        temp_df["std"] = temp_df.groupby(["policy_domain"])["sum_position"].transform("std")
        temp_df = temp_df.drop_duplicates(subset=["policy_domain"])

        # Sort always by the same order
        temp_df = temp_df.set_index("policy_domain")
        temp_df = temp_df.loc[policy_names.keys()]
        temp_df = temp_df.reset_index()

        means = np.array(temp_df["mean"].tolist())
        std_devs = np.array(temp_df["std"].tolist())

        num_axes = temp_df.policy_domain.nunique()
        # Calculate angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_axes, endpoint=False).tolist()

        # Plot the data with error bars representing standard deviation
        ax.fill_between(angles, means - std_devs, means + std_devs, color='skyblue', alpha=0.3)
        ax.plot(angles, means, color='blue', linewidth=2, label='Mean')

        # Set labels for each axis
        ax.set_thetagrids(np.degrees(angles), labels=[policy_names[p] for p in temp_df.policy_domain.tolist()])

        # Set the radial axis label
        ax.set_rlabel_position(0)
        # Add a title
        ax.set_title(f'{model}')

    plt.tight_layout()
    plt.savefig(f'data/responses/policy_domain_positioning.jpeg', dpi=600)
    plt.show()

plot_positioning_per_model()
