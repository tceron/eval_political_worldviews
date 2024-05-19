import pandas as pd
pd.options.mode.chained_assignment = None
import glob
from utils import convert_answers, parties_ches, convert_unique_id, models_renamed, sorter_models,  party2ches, rile_color
import matplotlib.pyplot as plt
import numpy as np
import argparse

def get_answers_stats_vaa(country):
    """ Get the answer of parties per VAA per country given in the parameter
    :return: dict with statementID as key and a dictionary with parties as keys and answers as values.
     """
    stats2answers = {}
    vaa = pd.read_csv(f"./PolBiases/data/vaa/{country}.csv")
    vaa["position"] = vaa.position.apply(lambda x: convert_answers(x))
    vaa["party"] = vaa.party.apply(lambda x: parties_ches[country][x] if x in parties_ches[country].keys() else False)
    vaa = vaa[vaa.party != False]
    for st in vaa.statementID.unique():
        stats2answers[st] = dict(zip([i.lower() for i in vaa[vaa.statementID == st].party.tolist()], vaa[vaa.statementID == st].position.tolist()))

    return stats2answers

def get_answers_parties_vaa(country):
    """ Get the answer of parties per VAA per country given in the parameter
    :return: dict with party as key and a dictionary with statementID as keys and answers as values.
    """
    party2answers = {}
    vaa = pd.read_csv(f"./PolBiases/data/vaa/{country}.csv")
    vaa["position"] = vaa.position.apply(lambda x: convert_answers(x))
    vaa["party"] = vaa.party.apply(lambda x: parties_ches[country][x] if x in parties_ches[country].keys() else False)
    vaa = vaa[vaa.party != False]
    for p in vaa.party.unique():
        party2answers[p.lower()] = dict(zip(vaa[vaa.party == p].statementID.tolist(), vaa[vaa.party == p].position.tolist()))
    return party2answers

def map_ches_scale(df):
    """ Map the leaning of the party to the dataframe according to the CHES scale."""
    df["rile"] = df.party.apply(lambda x:map_lrgen(party2ches[x.lower()]))
    return df

def map_lrgen(x):
    """ Map the CHES scale to left, right or center according to the survey's orientation. """
    if x < 4:
        return "left"
    elif x > 6:
        return "right"
    else:
        return "center"

def similarity_answers_models_and_parties(country):
    """ Get the relative number of times that the answers of the models match the answers of the parties per VAA.
    :return: DataFrame with the relative number of matches between the models and the parties per VAA.
    """

    stats2answers = get_answers_stats_vaa(country) # retrieves statements to answers
    files = glob.glob("./data/responses/updated/*.csv")
    results = []
    for f in files:
        model_name = models_renamed[f.split("/")[-1].split("_")[0]]
        template_id = f.split("/")[-1].split("_")[1]
        df = pd.read_csv(f)
        tmp = df[df["country_code"] == country]
        tmp["unique_id"] = tmp.unique_id.apply(lambda x: convert_unique_id(x))

        tmp["answer"]=tmp.apply(lambda x: -1 if x["string_answers"][10] == "n" else int(x["string_answers"][10]), axis=1)

        if args.passed_test and (model_name not in ['alwaysDISagree', 'alwaysAgree', 'RANDOM']):
            tmp2 = tmp[tmp[args.passed_test] == True]
        else:
            tmp2 = tmp

        model_answers = dict(zip(tmp2["unique_id"].tolist(), tmp2.answer.tolist())) # for a dictionary with the unique_id as key and the answer as value
        if (args.condition==0 or args.condition == 1):
            # retrieve keys and values of model_answers if value is equal to the condition passed
            model_answers = {k: v for k, v in model_answers.items() if v == args.condition}

        if len(model_answers)>0:
            for id in model_answers.keys():
                for party in stats2answers[id].keys():
                    # if their answers match, mark as 1, otherwise as 0
                    if model_answers[id] == stats2answers[id][party]:
                        results.append((model_name, country, template_id, id, party, 1, len(model_answers.keys()),
                                        len(tmp["unique_id"].unique())))
                    else:
                        results.append((model_name, country, template_id, id, party, 0, len(model_answers.keys()),
                                        len(tmp["unique_id"].unique())))
    df = pd.DataFrame(results, columns=["model", "country", "template_id", "statementID", "party", "match", "answered_stats_model", "n_vaa_stats"])
    df = map_ches_scale(df)
    df["n_matches"] = df.groupby(["model", "template_id", "rile","country", "party"])['match'].transform('sum') #"country", "party"
    df = df.drop_duplicates(subset=["model", "template_id", "rile", "country", "party"])
    df["relative_matches_model"] = df.apply(lambda x: 100*(x.n_matches/x.answered_stats_model), axis=1) # relative number of matches with a party based on the number of statements that the model has answered and passed the test
    df["relative_matches_stats"] = df.apply(lambda x: 100 * (x.n_matches / x.n_vaa_stats), axis=1) # relative number of matches with a party based on the total number of statements per VAA.
    df["relative_reliable_statements"] = df.apply(lambda x: 100 * (x.answered_stats_model / x.n_vaa_stats), axis=1)

    return df

def similarity_dataframe_per_party():
    df = pd.DataFrame()
    for c in list(parties_ches.keys()):
        tmp = similarity_answers_models_and_parties(c)
        df = pd.concat([df, tmp], ignore_index=True)
    df = df.drop(columns=["match"])
    # df.to_csv(f"./data/responses/scaling_across_templates_relative.csv", index=False)
    return df

def binned_answers2ches_plots_together():

    # df = pd.read_csv('./data/responses/scaling_across_templates_relative.csv')
    df = similarity_dataframe_per_party()
    df = df.set_index("model")
    print(sorting)
    df = df.loc[sorting]
    df = df.reset_index()

    # If only one test is present, axes is not a list, so we convert it into a list for consistency.
    if args.condition == 4:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 2.5))
    else:
        fig, ax= plt.subplots(nrows=1, ncols=1, figsize=(5, 4))

    offset = 0.2
    models_num_stats = []
    for model in sorting:
        tmp_df = df[df['model'] == model]

        tmp_df["mean_per_country"]=tmp_df.groupby(["template_id"])["relative_reliable_statements"].transform("mean")  # first it groups by the template_id because the by_country is the same
        models_num_stats.append(round(tmp_df["mean_per_country"].unique().mean(),1))  # this is the mean of reliable statements from the mean across templates per country

        for case in ['center', 'right', 'left']:

            means_same_template = []
            for temp in tmp_df.template_id.unique():
                means_same_template.append(tmp_df[(tmp_df['rile'] == case)&(tmp_df.template_id == temp)]["relative_matches_model"].mean())

            mean_value = np.mean(means_same_template)
            std_value = np.std(means_same_template)
            model_index = sorting.index(model)

            if case == 'center':
                position = model_index
            elif case == 'right':
                position = model_index - offset
            else:
                position = model_index + offset

            ax.errorbar(mean_value,  position, xerr=std_value, fmt='o',
                        color=rile_color[case], ecolor='lightgray', elinewidth=2, capsize=3, label=case)

    ax.set_yticks(np.arange(len(sorting)) - offset / 2)  # Adjust the ticks to be in between the two cases
    ax.set_yticklabels(sorting)
    if args.passed_test:
        ax.set_xlim(0, 100)
    else:
        ax.set_xlim(0, 100)
    ax.invert_yaxis()  ## O Invert y-axis to match the provided plot
    #ax.set_title(title)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='lower center', bbox_to_anchor=(-0.2, -0.16), fontsize=9,
              fancybox=True, shadow=True)  # borderaxespad=-0.55
    ax.set_xlabel('Relative matches per party')

    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(np.arange(len(sorting)) - offset / 2)  # Adjust the ticks to be in between the two cases
    ax2.set_yticklabels(models_num_stats)
    if args.condition == 0:
        ax2.set_ylabel('Rel. num of DISgrees as answers', rotation=270, labelpad=14)
    elif args.condition == 1:
        ax2.set_ylabel('Rel. num of Agrees as answers', rotation=270, labelpad=14)
    else:
        ax2.set_ylabel('Rel. num of answers', rotation=270, labelpad=14)

    plt.tight_layout()
    # Save figure
    plt.savefig(f'data/responses/plots_paperv3/scaling_{args.passed_test}_{args.condition}.jpeg', dpi=300)
    plt.show()
    plt.close()

def binned_answers2ches_plots_together_more_plots():

    # df = pd.read_csv('./data/responses/scaling_across_templates_relative.csv')
    df = similarity_dataframe_per_party()
    df = df.set_index("model")
    print(sorting)
    df = df.loc[sorting]
    df = df.reset_index()

    # If only one test is present, axes is not a list, so we convert it into a list for consistency.
    fig, axes = plt.subplots(nrows=2, ncols=int(len(df["template_id"].unique())/2), figsize=(12, 8))
    axes = axes.flatten()

    offset = 0.2
    for enu, (ax, temp_id) in enumerate(zip(axes, df["template_id"].unique())):
        models_num_stats = []

        for model in sorting:
            tmp_df = df[(df['model'] == model)&(df['template_id'] == temp_id)]

            models_num_stats.append(round(tmp_df["relative_reliable_statements"].unique().mean(),
                                          1))  # this is the mean of reliable statements from the mean across templates per country

            if len(tmp_df)>0:

                for case in ['center', 'right', 'left']:

                    mean_value = np.mean(tmp_df[(tmp_df['rile'] == case)]["relative_matches_model"])
                    std_value = np.std(tmp_df[(tmp_df['rile'] == case)]["relative_matches_model"])
                    model_index = sorting.index(model)

                    if case == 'center':
                        position = model_index
                    elif case == 'right':
                        position = model_index - offset
                    else:
                        position = model_index + offset

                    ax.errorbar(mean_value,  position, xerr=std_value, fmt='o',
                                color=rile_color[case], ecolor='lightgray', elinewidth=2, capsize=3, label=case)

        ax.set_yticks(np.arange(len(sorting)) - offset / 2)  # Adjust the ticks to be in between the two cases
        ax.set_yticklabels(sorting)
        if args.passed_test:
            ax.set_xlim(0, 100)
        else:
            ax.set_xlim(0, 100)
        ax.invert_yaxis()  ## O Invert y-axis to match the provided plot
        ax.set_title(f'{temp_id} - {title}')

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper right")
        ax.set_xlabel('Relative matches per party')

        if args.passed_test:
            ax2 = ax.twinx()
            ax2.set_ylim(ax.get_ylim())
            ax2.set_yticks(np.arange(len(sorting)) - offset / 2)  # Adjust the ticks to be in between the two cases
            ax2.set_yticklabels(models_num_stats)
            if args.condition == 0:
                ax2.set_ylabel('Rel. num of disagrees as answers', rotation=270, labelpad=14)
            elif args.condition == 1:
                ax2.set_ylabel('Rel. num of agrees as answers', rotation=270, labelpad=14)
            else:
                ax2.set_ylabel('Rel. num of answers', rotation=270, labelpad=14)

    plt.tight_layout()
    # Save figure
    plt.savefig(f'data/responses/plots_paperv3/scaling_{args.passed_test}_6templates_{args.condition}.jpeg', dpi=300)
    plt.show()
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--passed_test", type=str, default=False)
    parser.add_argument("--condition", type=int, default=2, help="If a condition is passed, the script will only consider the answers that match the condition. 1 for agree and 0 for disagree.")
    args = parser.parse_args()

    if args.condition == 3 and not args.passed_test:
        title = f'Leaning under simulations'
        sorting = sorter_models['simulations']

    if args.condition == 1 and args.passed_test:
        title = f'Leaning of agrees'
        sorting = sorter_models['agree']
    if args.condition == 0 and args.passed_test:
        title = f'Leaning of disagrees'
        sorting = sorter_models['disagree']
    if args.condition == 2 and args.passed_test:
        title = f'Passed all tests'
        sorting = sorter_models['only_models']
    if args.condition == 2 and not args.passed_test:
        title = f'All statements answered'
        sorting = sorter_models['all']
    if args.condition == 1 and not args.passed_test:
        title = f'Leaning of agrees UNconstrained'
        sorting = sorter_models['agree']
    if args.condition == 0 and not args.passed_test:
        title = f'Leaning of disagrees UNconstrained'
        sorting = sorter_models['disagree']

    binned_answers2ches_plots_together()
    # binned_answers2ches_plots_together_more_plots()