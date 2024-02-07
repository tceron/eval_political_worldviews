import pandas as pd
pd.options.mode.chained_assignment = None
import glob
from collections import defaultdict
from itertools import combinations
from utils import run_pca, convert_answers, parties_ches, convert_unique_id, country2ches, ch_ches, models_renamed, sorter_models, country_color, retrieve_template_type, party2ches
import tabulate
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import cityblock, cosine
import matplotlib.pyplot as plt
import numpy as np

def proximity_passed_all_tests(country):
    """Only the ones that passed all models, no neutral statements. """

    party2answers = get_answers_parties_vaa(country)
    files = glob.glob("./data/responses/dataframes_withpasses/*.csv")
    results = []
    for f in files:
        model_name = models_renamed[f.split("/")[-1].split("_")[0]]
        template_id = f.split("/")[-1].split("_")[1]
        df = pd.read_csv(f)
        tmp = df[df["country_code"] == country]
        tmp["unique_id"] = tmp.unique_id.apply(lambda x: convert_unique_id(x))
        tmp["answer"]=tmp.apply(lambda x: -1 if x["hard_pass"] == False else int(x["string_answers"][10]), axis=1)
        tmp = tmp[tmp["hard_pass"] == True]
        model_answers = dict(zip(tmp["unique_id"].tolist(), tmp.answer.tolist()))

        for p in party2answers.keys():
            party_answers = {k:v for k,v in party2answers[p].items() if v != -1}
            set_ids = set(model_answers.keys()).intersection(set(party_answers.keys()))
            spearman = spearmanr([model_answers[i] for i in set_ids], [party_answers[i] for i in set_ids])
            if str(spearman[0])!="nan":
                r, pval = round(spearman[0], 3), round(spearman[1], 3)
            else:
                r, pval = 0, 0
            results.append((model_name, "full_success", p, template_id, r, pval, len(set_ids)))
    df = pd.DataFrame(results, columns=["model", "test", "party", "template_id", "r", "pval", "n_statements"])
    return df

def proximity_to_parties(country):
    """ Non-consistent statements become neutral. """
    party2answers = get_answers_parties_vaa(country)
    files = glob.glob("./data/responses/dataframes_withpasses/*.csv")
    results = []
    for f in files:
        model_name = models_renamed[f.split("/")[-1].split("_")[0]]
        template_id = f.split("/")[-1].split("_")[1]
        df = pd.read_csv(f)
        tmp = df[df["country_code"] == country]

        tmp["unique_id"] = tmp.unique_id.apply(lambda x: convert_unique_id(x))
        tmp["answer"]=tmp.apply(lambda x: -1 if x["hard_pass"] == False else int(x["string_answers"][10]), axis=1) # -1 if it has not passed all models.
        # tmp = tmp[tmp["hard_pass"] == True]

        model_answers = dict(zip(tmp["unique_id"].tolist(), tmp.answer.tolist()))

        for p in party2answers.keys():
            party_answers = {k:v for k,v in party2answers[p].items() if v != -1}
            set_ids = set(model_answers.keys()).intersection(set(party_answers.keys()))
            matches = round(100*(len([i for i in set_ids if model_answers[i] == party_answers[i]])/len(set_ids)),2) if len(set_ids) > 0 else 0
            mismatches = round(100*(len([i for i in set_ids if model_answers[i] != party_answers[i]])/len(set_ids)), 2) if len(set_ids) > 0 else 0
            results.append((model_name, country, p, int(template_id), len(model_answers.values()), len(tmp[tmp["hard_pass"] == True]), matches, mismatches, len(set_ids)))
    df = pd.DataFrame(results, columns=["model", "country", "party", "template_id", "n_statements", "n_stats_answered", "matches", "mismatches", "n_stat_matches"])
    return df


def proximity_to_parties_country_agnostic(country):
    """ Non-consistent statements become neutral. """
    party2answers = get_answers_parties_vaa(country)
    files = glob.glob("./data/responses/dataframes_withpasses/*.csv")
    results = []
    for f in files:
        model_name = models_renamed[f.split("/")[-1].split("_")[0]]
        template_id = f.split("/")[-1].split("_")[1]
        df = pd.read_csv(f)

        for agn in df.country_agnostic.unique():
            tmp = df[(df["country_code"] == country)&(df.country_agnostic == agn)]
            tmp["unique_id"] = tmp.unique_id.apply(lambda x: convert_unique_id(x))
            tmp["answer"]=tmp.apply(lambda x: -1 if x["hard_pass"] == False else int(x["string_answers"][10]), axis=1) # -1 if it has not passed all models.
            # tmp = tmp[tmp["hard_pass"] == True]
            model_answers = dict(zip(tmp["unique_id"].tolist(), tmp.answer.tolist()))

            for p in party2answers.keys():
                party_answers = {k:v for k,v in party2answers[p].items() if v != -1}
                set_ids = set(model_answers.keys()).intersection(set(party_answers.keys()))
                matches = round(100*(len([i for i in set_ids if model_answers[i] == party_answers[i]])/len(set_ids)),2) if len(set_ids) > 0 else 0
                mismatches = round(100*(len([i for i in set_ids if model_answers[i] != party_answers[i]])/len(set_ids)), 2) if len(set_ids) > 0 else 0
                results.append((model_name, country, p, int(template_id), len(model_answers.values()), len(tmp[tmp["hard_pass"] == True]), matches, mismatches, agn))

    df = pd.DataFrame(results, columns=["model", "country", "party", "template_id", "n_statements", "n_stats_answered",  "matches", "mismatches", "country_agnostic"])

    return df

def order_ches(df):
    """ Order parties according to their position in the political spectrum of CHES. """

    # df_ches = pd.read_csv("data/1999-2019_CHES_dataset_means(v3).csv")
    # df_ches = df_ches[(df_ches.year == 2019) & (df_ches.country.isin(country2ches.values()))]
    # party2ches = dict(zip([i.lower() for i in df_ches.party.tolist()], df_ches.lrgen.tolist()))
    parties = df.party.unique()
    # party2ches.update(ch_ches)
    # order parties list according to their position in the political spectrum of party2ches
    sorter = sorted(parties, key=lambda x: party2ches[x.lower()], reverse=False)

    df = df.set_index("party")
    df = df.loc[sorter]
    df = df.reset_index()

    return df

def plot_proximity_to_parties(df, file_name, title):
    df["mean"] = df.groupby(["party", "model"])['matches'].transform('mean')
    df["std"] = df.groupby(["party", "model"])['matches'].transform('std')
    df["mean_stats"] = df.groupby(["party", "model"])['n_stats_answered'].transform('mean')
    df["std_stats"] = df.groupby(["party", "model"])['n_stats_answered'].transform('std')

    df = df.drop_duplicates(subset=['party', 'model'])

    models =  sorter_models

    # If only one test is present, axes is not a list, so we convert it into a list for consistency.
    fig, axes = plt.subplots(nrows=2, ncols=int(len(models)/2), figsize=(14 , 8))
    # Flatten axes array for easy iteration (if len(tests) is 6)
    axes = axes.flatten()

    if len(models) == 1:
        axes = [axes]

    for enu, (ax, model) in enumerate(zip(axes, models)):
        # Filter the dataframe for the current test
        model_df = df[df['model'] == model]
        model_df = order_ches(model_df)

        # plot errorbar per row of the dataframe
        for i, row in model_df.iterrows():
            ax.errorbar(row['mean'], row['party'], xerr=row['std'], fmt='o', color=country_color[row["country"]], ecolor='gray',
                    elinewidth=2, capsize=3, label=row["country"])

        ax.set_title(f'{model} {title}')
        ax.set_xlim(0, 70)
        # ax.set_xlabel('Mean % of matches across templates')
        # ax.set_ylabel('Model')
        ax.invert_yaxis()  # Invert y-axis to match the provided plot
        ax.grid(True)

        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(range(0, len(model_df), 1), [round(i, 1) for i in model_df["mean_stats"].tolist()])

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="lower right")

    fig.text(0.5, 0.01, 'Mean % of matches per party across templates', ha='center', va='center',
             fontsize=10)

    # Adjust layout
    plt.tight_layout()
    # Save figure
    plt.savefig(f'data/responses/{file_name}.jpeg', dpi=300)
    plt.show()
    plt.close()


def run_plot_agnostic_specific():
    df = pd.DataFrame()
    for c in list(parties_ches.keys()):
        tmp = proximity_to_parties_country_agnostic(c)
        df = pd.concat([df, tmp], ignore_index=True)
    for t in df.country_agnostic.unique():
        tmp = df[df.country_agnostic == t]
        if t == 1:
            title = "country-agnostic"
        else:
            title = "country-specific"
        plot_proximity_to_parties(tmp, f"matches_{t}", title)

def run_plot_across_country():
    df = pd.DataFrame()
    for c in list(parties_ches.keys()):
        tmp = proximity_to_parties(c)
        df = pd.concat([df, tmp], ignore_index=True)
    plot_proximity_to_parties(df, "matches_across_countries", "")

def run_plot_across_country_personal_impersonal():
    template2type, template2type_gpt = retrieve_template_type()
    df = pd.DataFrame()
    for c in list(parties_ches.keys()):
        tmp = proximity_to_parties(c)
        df = pd.concat([df, tmp], ignore_index=True)
    print(df)
    df["prompt_type"] = df.apply(lambda x: template2type[x.template_id] if x.model != "GPT3.5-turbo-20b" else template2type_gpt[x.template_id], axis=1)
    for t in df.prompt_type.unique():
        tmp = df[df.prompt_type == t]
        plot_proximity_to_parties(tmp, f"matches_{t}", t)


def get_answers_all_models(country):
    files = glob.glob("./data/responses/dataframes_withpasses/*.csv")
    models_answers = {}
    for f in files:
        model_name = models_renamed[f.split("/")[-1].split("_")[0]]
        template_id = f.split("/")[-1].split("_")[1]

        if template_id not in models_answers.keys():
            models_answers[template_id] = {}

        df = pd.read_csv(f)
        tmp = df[df["country_code"] == country]
        tmp["unique_id"] = tmp.unique_id.apply(lambda x: convert_unique_id(x))
        tmp["answer"]=tmp.apply(lambda x: -1 if x["hard_pass"] == False else int(x["string_answers"][10]), axis=1) # -1 if it has not passed all models.

        models_answers[template_id][model_name] = dict(zip(tmp["unique_id"].tolist(), tmp.answer.tolist()))
    return models_answers

def get_answers_stats_vaa(country):
    stats2answers = {}
    vaa = pd.read_csv(f"./PolBiases/data/vaa/{country}.csv")
    vaa["position"] = vaa.position.apply(lambda x: convert_answers(x))
    vaa["party"] = vaa.party.apply(lambda x: parties_ches[country][x] if x in parties_ches[country].keys() else False)
    vaa = vaa[vaa.party != False]
    for st in vaa.statementID.unique():
        stats2answers[st] = dict(zip([i.lower() for i in vaa[vaa.statementID == st].party.tolist()], vaa[vaa.statementID == st].position.tolist()))

    return stats2answers


def save_party_answers(country):
    stats2answers = {}
    vaa = pd.read_csv(f"./PolBiases/data/vaa/{country}.csv")
    vaa["position"] = vaa.position.apply(lambda x: convert_answers(x))
    vaa["party"] = vaa.party.apply(lambda x: parties_ches[country][x] if x in parties_ches[country].keys() else False)
    vaa = vaa[vaa.party != False]
    for st in vaa.statementID.unique():
        stats2answers[st] = dict(zip([i.lower() for i in vaa[vaa.statementID == st].party.tolist()], vaa[vaa.statementID == st].position.tolist()))

    return stats2answers

def get_answers_parties_vaa(country):
    party2answers = {}
    vaa = pd.read_csv(f"./PolBiases/data/vaa/{country}.csv")
    vaa["position"] = vaa.position.apply(lambda x: convert_answers(x))
    vaa["party"] = vaa.party.apply(lambda x: parties_ches[country][x] if x in parties_ches[country].keys() else False)
    vaa = vaa[vaa.party != False]
    for p in vaa.party.unique():
        party2answers[p.lower()] = dict(zip(vaa[vaa.party == p].statementID.tolist(), vaa[vaa.party == p].position.tolist()))
    return party2answers

def add_scale(df):
    # df_ches = pd.read_csv("data/1999-2019_CHES_dataset_means(v3).csv")
    # df_ches = df_ches[(df_ches.year == 2019) & (df_ches.country.isin(country2ches.values()))]
    # party2ches = dict(zip([i.lower() for i in df_ches.party.tolist()], df_ches.lrgen.tolist()))
    # party2ches.update(ch_ches)

    df["rile"] = df.party.apply(lambda x:map_lrgen(party2ches[x.lower()]))

    return df

def map_lrgen(x):
    if x < 4:
        return "left"
    elif x > 6:
        return "right"
    else:
        return "center"

rile_color = {"left": "purple", "center": "green", "right": "blue"}

def num_parties_leaning():
    # df_ches = pd.read_csv("data/1999-2019_CHES_dataset_means(v3).csv")
    # df_ches = df_ches[(df_ches.year == 2019) & (df_ches.country.isin(country2ches.values()))]
    # party2ches = dict(zip([i.lower() for i in df_ches.party.tolist()], df_ches.lrgen.tolist()))
    # party2ches.update(ch_ches)
    results= {}
    for c in list(parties_ches.keys()):
        for p in parties_ches[c].keys():
            position = map_lrgen(party2ches[parties_ches[c][p].lower()])
            if position not in results.keys():
                results[position] = 0
            results[position]+=1
    return results


def validity_leaning():
    data = pd.read_csv('./data/responses/leaning_across_templates.csv')
    results = []
    for model, group_model in data.groupby("model"):
        for (temp, rile), group_rile in group_model.groupby(["template_id", "rile"]):
            validity = round(group_rile.answered_stats_model.sum() / group_rile.total_stats.sum() * 100)
            leaning = round(group_rile.n_matches.sum() / group_model.answered_stats_party_and_model.sum() * 100)
            # print(f'\t{camp.title()}: {validity=}, {leaning=}')
            results.append((model, temp, rile, validity, leaning))
    df = pd.DataFrame(results, columns=["model", "template_id", "rile", "validity", "leaning"])
    return df


# def similarity_answers_models_and_parties(country):
#     """ Non-consistent statements become neutral. """
#     df_final = pd.DataFrame()
#     for country in list(parties_ches.keys()):
#         stats2answers = get_answers_stats_vaa(country)
#         party2answers = get_answers_parties_vaa(country)
#         files = glob.glob("./data/responses/dataframes_withpasses/*.csv")
#         results = []
#         for f in files:
#             model_name = models_renamed[f.split("/")[-1].split("_")[0]]
#             template_id = f.split("/")[-1].split("_")[1]
#             df = pd.read_csv(f)
#             tmp = df[df["country_code"] == country]
#             tmp["unique_id"] = tmp.unique_id.apply(lambda x: convert_unique_id(x))
#             tmp["answer"]=tmp.apply(lambda x: -1 if x["hard_pass"] == False else int(x["string_answers"][10]), axis=1) # -1 if it has not passed all models.
#
#             tmp2 = tmp[tmp["hard_pass"] == True]
#
#             model_answers = dict(zip(tmp2["unique_id"].tolist(), tmp2.answer.tolist()))
#
#             for id in model_answers.keys():
#                 for p in stats2answers[id].keys():
#                     num_answer_party = [k for k,v in party2answers[p].items() if v != -1]
#                     common_ids = list(set(num_answer_party).intersection(model_answers.keys()))
#                     if model_answers[id] == stats2answers[id][p]:
#                         results.append((model_name, country, template_id, p, 1, len(model_answers.keys()),  len(tmp["unique_id"].unique()), len(num_answer_party), len(common_ids)))
#                     else:
#                         results.append((model_name, country, template_id, p, 0, len(model_answers.keys()), len(tmp["unique_id"].unique()), len(num_answer_party), len(common_ids)))
#         df = pd.DataFrame(results, columns=["model", "country", "template_id", "party", "match", "answered_stats_model", "total_stats", "answered_stats_party", "answered_stats_party_and_model"])
#         df = add_scale(df)
#         df["n_matches"] = df.groupby(["model", "template_id", "rile", "country", "party"])['match'].transform('sum')
#         df = df.drop_duplicates(subset=["model", "template_id", "rile", "country", "party"])
#         df_final = pd.concat([df_final, df], ignore_index=True)
#     df_final = df_final.drop(columns=["match"])
#     df_final.to_csv("./data/responses/leaning_across_templates.csv", index=False)
#     return df_final


def agreement_normalized():
    df = pd.read_csv('./data/responses/leaning_across_templates.csv')
    results = []

    count_per_category = df.drop_duplicates(subset="party").groupby('rile').size().reset_index(name='n_party')
    dic_counts = dict(zip(count_per_category.rile.tolist(), count_per_category.n_party.tolist()))

    df["matches_rile"] = df.groupby(["model", "template_id", "rile"])["n_matches"].transform("sum")
    df = df.drop_duplicates(subset=["model", "template_id", "rile"])
    df["norm_matches_rile"] = df.apply(lambda x: round(x["matches_rile"] / dic_counts[x["rile"]] * 100), axis=1)

    return df

def similarity_answers_models_and_parties(country):
    """ Non-consistent statements become neutral. """
    stats2answers = get_answers_stats_vaa(country)
    party2answers = get_answers_parties_vaa(country)
    files = glob.glob("./data/responses/dataframes_withpasses/*.csv")
    results = []
    for f in files:
        model_name = models_renamed[f.split("/")[-1].split("_")[0]]
        template_id = f.split("/")[-1].split("_")[1]
        df = pd.read_csv(f)
        tmp = df[df["country_code"] == country]
        tmp["unique_id"] = tmp.unique_id.apply(lambda x: convert_unique_id(x))
        tmp["answer"]=tmp.apply(lambda x: -1 if x["hard_pass"] == False else int(x["string_answers"][10]), axis=1) # -1 if it has not passed all models.

        tmp2 = tmp[tmp["hard_pass"] == True]

        model_answers = dict(zip(tmp2["unique_id"].tolist(), tmp2.answer.tolist()))

        for id in model_answers.keys():
            for p in stats2answers[id].keys():
                num_answer_party = [k for k,v in party2answers[p].items() if v != -1]
                common_ids = list(set(num_answer_party).intersection(model_answers.keys()))
                if model_answers[id] == stats2answers[id][p]:
                    results.append((model_name, country, template_id, p, 1, len(model_answers.keys()),  len(tmp["unique_id"].unique()), len(num_answer_party), len(common_ids)))
                else:
                    results.append((model_name, country, template_id, p, 0, len(model_answers.keys()), len(tmp["unique_id"].unique()), len(num_answer_party), len(common_ids)))
    df = pd.DataFrame(results, columns=["model", "country", "template_id", "party", "match", "answered_stats_model", "total_stats", "answered_stats_party", "answered_stats_party_and_model"])
    df = add_scale(df)
    # n_parties = num_parties_leaning()
    df["n_matches"] = df.groupby(["model", "template_id", "rile", "country", "party"])['match'].transform('sum')
    df = df.drop_duplicates(subset=["model", "template_id", "rile", "country", "party"])
    df["relative_matches_model"] = df.apply(lambda x: 100*(x.n_matches/x.answered_stats_model), axis=1)
    df["relative_matches_stats"] = df.apply(lambda x: 100 * (x.n_matches / x.total_stats), axis=1)

    return df

def save_df():
    df = pd.DataFrame()
    for c in list(parties_ches.keys()):
        tmp = similarity_answers_models_and_parties(c)
        df = pd.concat([df, tmp], ignore_index=True)
    df = df.drop(columns=["match"])
    df.to_csv(f"./data/responses/scaling_across_templates_relative.csv", index=False)

save_df()

def binned_answers2ches_plots_together():

    df = pd.read_csv('./data/responses/scaling_across_templates_relative.csv')
    df = df.set_index("model")
    df = df.loc[sorter_models]
    df = df.reset_index()

    # If only one test is present, axes is not a list, so we convert it into a list for consistency.
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    offset = 0.2

    for i, ax in zip(["relative_matches_stats", "relative_matches_model"], axes):
        models_num_stats = []
        for model in sorter_models:
            tmp_df = df[df['model'] == model]
            tmp_df["mean_models_answers"] = tmp_df.relative_matches_model.mean()

            tmp_df["mean_stats_model"] = tmp_df.groupby(["template_id"])["answered_stats_model"].transform("mean")
            models_num_stats.append(tmp_df["mean_stats_model"].unique().sum())

            for case in ['center', 'right', 'left']:
                means = []
                for temp in tmp_df.template_id.unique():
                    means.append(tmp_df[(tmp_df['rile'] == case)&(tmp_df.template_id == temp)][i].mean())

                mean_value = np.mean(means)
                std_value = np.std(means)
                model_index = sorter_models.index(model)

                if case == 'center':
                    position = model_index
                elif case == 'right':
                    position = model_index - offset
                else:  # 'Specific'
                    position = model_index + offset

                ax.errorbar(mean_value,  position, xerr=std_value, fmt='o',
                            color=rile_color[case], ecolor='lightgray', elinewidth=2, capsize=3, label=case)

        if i == "relative_matches_model":
            ax.set_xlim(0, 100)
            ax.invert_yaxis()  ## O Invert y-axis to match the provided plot
            ax.set_title(f'Political leaning')
            ax.set_yticks([])
            # handles, labels = ax.get_legend_handles_labels()
            # by_label = dict(zip(labels, handles))
            # ax.legend(by_label.values(), by_label.keys(), loc="upper right")
            ax.set_xlabel('% matches norm. by length of consistent stats')
            ax2 = ax.twinx()
            ax2.set_ylim(ax.get_ylim())
            ax2.set_yticks(np.arange(len(sorter_models)) - offset / 2)  # Adjust the ticks to be in between the two cases
            ax2.set_yticklabels([round(i, 1) for i in models_num_stats])
            ax2.set_ylabel('Number of consistent statements', rotation=270, labelpad=14)
        else:
            ax.set_yticks(np.arange(len(sorter_models)) - offset / 2)  # Adjust the ticks to be in between the two cases
            ax.set_yticklabels(sorter_models)
            ax.set_xlim(0, 50)
            ax.invert_yaxis()  ## O Invert y-axis to match the provided plot
            ax.set_title(f'Validity of the leaning')

            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc="upper right")
            ax.set_xlabel('% matches norm. by length of VAA')

    # xlabel = plt.xlabel('Mean of normalized number of matches per leaning across contries')
    # fig.text(0.5, 0.01, 'Norm. number of matches across templates', ha='center', va='center', fontsize=10)

    plt.tight_layout()
    # Save figure
    plt.savefig(f'data/responses/scaling_across_templates_norm.jpeg', dpi=300)
    plt.show()
    plt.close()

# run_plot_agnostic_specific()
# run_plot_across_country()
# run_plot_across_country_personal_impersonal()
binned_answers2ches_plots_together()

