import glob
import math

import numpy as np
import pandas as pd
import tabulate
import simpledorff
from sklearn.metrics import cohen_kappa_score
from collections import Counter, defaultdict
from aggregate import add_variante_type_info
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.express as px


def remove_statements_by_sign(dataframe, sign):
    """
    Removes statements from the dataframe where the unique_id in 'original' has the specified sign. Can be used to filter
    out all statements from a dataframe, for which the original statement did not result in a significant stance.
    """
    if sign == "-":
        original_statements = dataframe[dataframe["original"] == 1]
        unique_ids_to_remove = original_statements[original_statements["sign"] == sign]["unique_id"].unique()
        return dataframe[~dataframe["unique_id"].isin(unique_ids_to_remove)]
    return dataframe


def add_model_type_chat(df):
    """Add a column which indicates whether the model is a chat model or not."""
    df["chat_model"] = df["model_name"].apply(lambda x: "chat" in x)
    df["chat_model"] = df["chat_model"].apply(
        lambda x: True if x else False)
    return df


def compute_krippendorff_alpha(df):
    """
    Given the dataframe for one specific model, for one specific template, compute the Krippendorff's alpha between
    the answers between original and all paraphrases.
    """
    # we are only interested in the rows with original == True or paraphrase == True
    df = df[df["original"] | df["paraphrase"]]
    # the answers are specified in the column "binary", the annotators in "code" (each original and paraphrase have a
    # different code), and the unique_id is the unique identifier for each statement
    kripp_alpha = simpledorff.calculate_krippendorffs_alpha_for_df(df, class_col="binary",
                                                                   annotator_col="code",
                                                                   experiment_col="unique_id")
    return kripp_alpha


def normalized_entropy(annotations, num_labels):
    # remove nan values element based and compute the normalized entropy for a list of labels
    annotations = [x for x in annotations if str(x) != 'nan']
    data_count = Counter(annotations)
    total_count = len(annotations)
    entropy = 0.0

    for count in data_count.values():
        probability = count / total_count
        entropy -= probability * math.log2(probability)

    normalized_entropy = entropy / math.log2(num_labels)

    return normalized_entropy


def get_aggregated_with_disagreement(df):
    # for one model, one template, and one inverted setting, aggregate the answers for each statement and compute the
    # entropy and whether there is "full agreement" (i.e., all variants resulted on the same binary answer)
    df = df[df["original"] | df["paraphrase"]]
    aggregated = df.groupby("unique_id").agg({"binary": list}).reset_index()
    # Check for full agreement and calculate entropy
    aggregated["full_agreement"] = aggregated["binary"].apply(lambda x: len(set(x)) == 1)
    aggregated["entropy"] = aggregated["binary"].apply(normalized_entropy, num_labels=2)
    aggregated.loc[aggregated["full_agreement"], "entropy"] = 0
    return aggregated


def compute_cohens_kappa(df, col1, col2):
    """
    Given the dataframe for one specific model, for one specific template, compute the cohens kappa between the answers
    of two variants (specified in col1 and col2). The cohens kappa can only be computed between exactly two variants.
    """
    # get the columns with the answers for col1 and col2, such that they are sorted by unique_id
    df_col1 = df[df[col1] == 1].sort_values(by="unique_id")
    df_col2 = df[df[col2] == 1].sort_values(by="unique_id")
    # get all indices where one of the two has nan in binary
    indices_to_drop = set(df_col1[df_col1["binary"].isna()].unique_id.tolist())
    indices_to_drop = indices_to_drop.union(set(df_col2[df_col2["binary"].isna()].unique_id.tolist()))
    if len(indices_to_drop) > 0:
        # print(f"Dropping {len(indices_to_drop)} statements where one of the two variants has nan in binary.")
        df_col1 = df_col1[~df_col1["unique_id"].isin(indices_to_drop)]
        df_col2 = df_col2[~df_col2["unique_id"].isin(indices_to_drop)]
    # check that the numbers are the same
    assert len(df_col1) == len(df_col2)
    # compute the cohens kappa
    return cohen_kappa_score(df_col1["binary"], df_col2["binary"])


def add_paraphrase_variants(df):
    """
    Add three columns, one for each paraphrase (p1, p2, p3), which indicate whether the statement was one of the
    three paraphrases and which one
    """
    # Extract paraphrase variant from 'code' and convert to integer
    df.loc[:, "paraphrase_variant"] = df["code"].apply(lambda x: int(x[2]) if x[2] != "0" else 0)

    # Convert paraphrase variant into three columns (p1, p2, p3) and drop column "p_0"
    df = pd.concat([df, pd.get_dummies(df["paraphrase_variant"], prefix="p")], axis=1)
    df = df.drop(columns=["p_0"])

    # Convert the dummies into boolean values using .loc
    df.loc[:, "p_1"] = df["p_1"].astype(bool)
    df.loc[:, "p_2"] = df["p_2"].astype(bool)
    df.loc[:, "p_3"] = df["p_3"].astype(bool)

    return df


def number_of_inverted_matches(df, all):
    """
    For a specific model and template, compute the number of statements for which the inverted setting resulted in the
    same binary answer as the original setting. If all is True compute it for all possible variants,
    if it is False only for the original variant. (That is if we swap the labels in the prompt, do we get the same
    answer?)
    """
    if not all:
        df = df[df["original"]]

    def check_agreement(group):
        return group['binary'].nunique() == 1

    number_inverted_matches = df.groupby(['unique_id', 'code']).apply(check_agreement).sum()
    # return relative
    return number_inverted_matches / len(df)


def get_all_agreement_scores_one_template(df):
    """
    Given a dataframe, that only contains one model and one template, compute all agreement scores for all possible
    tests.
    :param df:
    :return:
    """
    assert len(df["model_name"].unique()) == 1, "The dataframe should only contain one model."
    # 1) compute agreement between original and paraphrase variants
    original_statements = df[df["original"] == True]
    # convert inverted col into two new columns, were inverted_true == True if inverted == 1 and inverted_false = False,
    # if inverted == 0, then exactly fill with the opposite
    original_statements.loc[:, "labels_inverted"] = original_statements["inverted"].apply(lambda x: x == 1)
    original_statements.loc[:, "labels_orig_order"] = original_statements["inverted"].apply(lambda x: x == 0)

    kappa_between_labels_original_and_inverted = compute_cohens_kappa(original_statements, "labels_inverted",
                                                                      "labels_orig_order")
    # 2) compute agreement between original and paraphrase variants, compute fleiss kappa for the whole dataframe and pair-wise
    # cohens kappa for the part that was annotated by the humans
    # drop inverted == 1
    df = df[df["inverted"] == 0]
    df = add_paraphrase_variants(df)
    krippendorff_alpha_semantic_consistency = compute_krippendorff_alpha(df)
    # compute pair-wise cohens kappa for the part that was annotated by the humans
    pair_wise_kappas_hs = get_kappa_scores_human_sample(df)
    kappa_original_negation_full_df = compute_cohens_kappa(df, "original", "negation")
    kappa_original_opposite_full_df = compute_cohens_kappa(df, "original", "opposite")
    kappa_original_paraphrase_full_df = (compute_cohens_kappa(df, "original", "p_1") +
                                         compute_cohens_kappa(df, "original", "p_2") +
                                         compute_cohens_kappa(df, "original", "p_3")) / 3
    return {"kappa_between_labels_original_and_inverted": kappa_between_labels_original_and_inverted,
            "krippendorff_alpha_semantic_consistency": krippendorff_alpha_semantic_consistency,
            "hs-original-paraphrase": pair_wise_kappas_hs["original-paraphrase"],
            "hs-original-negation": pair_wise_kappas_hs["original-negation"],
            "hs-original-opposite": pair_wise_kappas_hs["original-opposite"],
            "kappa_original_negation_full_df": kappa_original_negation_full_df,
            "kappa_original_opposite_full_df": kappa_original_opposite_full_df,
            "kappa_original_paraphrase_full_df": kappa_original_paraphrase_full_df}


def get_results_one_template(result_file):
    # get all the agreement scores for one template and model.
    all_agreement_scores = get_all_agreement_scores_one_template(result_file)
    # next get the number of full passes.
    passed_statements = get_aggregated_questionnaire_hard_pass(result_file)
    total_number_of_hard_passes = sum(passed_statements.hard_pass)
    avg_relative_test_score = passed_statements.relative_amount_of_tests_passed.mean()
    results_for_passes = pd.DataFrame({"total_number_of_hard_passes": total_number_of_hard_passes,
                                       "avg_relative_test_score": avg_relative_test_score}, index=[0])
    df_agreement_scores = pd.DataFrame(all_agreement_scores, index=[0])
    return df_agreement_scores, results_for_passes, passed_statements


def get_kappa_scores_human_sample(df):
    """
    Compute the kappa scores for the human sample between the original and the negation, opposite, and paraphrase
    (same score as done for the humans)
    """
    human_annotated = pd.read_csv("data/responses/final_merged/aggregated/human_annotated_sample.csv", sep=",")
    unique_ids_humans = human_annotated["ID"].apply(lambda x: x.split("_")[0] + x.split("_")[1]).unique()
    df = df[df["unique_id"].isin(unique_ids_humans)]
    kappa_original_negation = compute_cohens_kappa(df, "original", "negation")
    kappa_original_opposite = compute_cohens_kappa(df, "original", "opposite")
    kappa_original_paraphrase = (compute_cohens_kappa(df, "original", "p_1") +
                                 compute_cohens_kappa(df, "original", "p_2") +
                                 compute_cohens_kappa(df, "original", "p_3")) / 3
    return {"original-negation": kappa_original_negation,
            "original-opposite": kappa_original_opposite,
            "original-paraphrase": kappa_original_paraphrase}


def compute_all_results(result_file, file_ending):
    # iterate over model_name and template
    agreement_results_human = []
    agreement_results_other = []
    passes_results = []
    for (model, template), df in result_file.groupby(["model_name", "template_id"]):
        # get the agreement for each test for this model and template.
        agreement, passes, passed_statements = get_results_one_template(df)
        print(model, template, passes)
        # split agreement in two dataframes, one only contains columns with "hs"
        agreement_human_sample = agreement.filter(regex="hs")
        agreement_other = agreement.drop(columns=agreement_human_sample.columns)
        # compute the mean of the absolute values of agreement
        agreement_human_sample["mean_absolute_agreement"] = agreement_human_sample.abs().mean(axis=1)
        agreement_other["mean_absolute_agreement"] = agreement_other.abs().mean(axis=1)

        # add model and template to the results
        agreement_human_sample["model"] = model
        agreement_human_sample["template"] = template
        agreement_other["model"] = model
        agreement_other["template"] = template
        passes["model"] = model
        passes["template"] = template
        # save the passes dataframe to data/responses/final_merged/results/dataframes_withpasses
        passed_statements.to_csv(
            f"data/responses/final_merged/results/dataframes_withpasses/{model}_{template}_{file_ending}.csv",
            sep=",", index=False)
        agreement_results_human.append(agreement_human_sample)
        agreement_results_other.append(agreement_other)
        passes_results.append(passes)
    passes_results = pd.concat(passes_results)
    agreement_results_human = pd.concat(agreement_results_human)
    agreement_results_other = pd.concat(agreement_results_other)
    print(tabulate.tabulate(passes_results, headers="keys", tablefmt="latex"))
    print(tabulate.tabulate(agreement_results_human, headers="keys", tablefmt="latex"))
    print(tabulate.tabulate(agreement_results_other, headers="keys", tablefmt="latex"))

    human_sample_mean_over_templates = agreement_results_human.groupby("model").mean()
    human_sample_max_mean_absolute_agreement = agreement_results_human.groupby("model").max()
    # add mean and max suffix to the column names
    human_sample_mean_over_templates.columns = [col + "_mean" for col in human_sample_mean_over_templates.columns]
    human_sample_max_mean_absolute_agreement.columns = [col + "_max" for col in
                                                        human_sample_max_mean_absolute_agreement.columns]
    # print the best template for each model

    # merge the two dataframes and drop duplicate columns
    merged = pd.merge(human_sample_mean_over_templates, human_sample_max_mean_absolute_agreement, on="model")
    merged = merged.loc[:, ~merged.columns.duplicated()]
    # drop template_mean and template_max cols
    merged = merged.drop(columns=["template_mean", "template_max"])
    # add the model name as a column
    merged["model"] = merged.index

    # then for the other sample
    other_sample_mean_over_templates = agreement_results_other.groupby("model").mean()
    other_sample_max_mean_absolute_agreement = agreement_results_other.groupby("model").max()
    # add mean and max suffix to the column names
    other_sample_mean_over_templates.columns = [col + "_mean" for col in other_sample_mean_over_templates.columns]
    other_sample_max_mean_absolute_agreement.columns = [col + "_max" for col in
                                                        other_sample_max_mean_absolute_agreement.columns]
    # merge the two dataframes and drop duplicate columns
    merged_other = pd.merge(other_sample_mean_over_templates, other_sample_max_mean_absolute_agreement, on="model")
    merged_other = merged_other.loc[:, ~merged_other.columns.duplicated()]
    # drop template_mean and template_max cols
    merged_other = merged_other.drop(columns=["template_mean", "template_max"])
    # add the model name as a column
    merged_other["model"] = merged_other.index

    mean_total_number_of_hard_passes = passes_results.groupby("model").mean()
    max_total_number_of_hard_passes = passes_results.groupby("model").max()
    # add mean and max suffix to the column names
    mean_total_number_of_hard_passes.columns = [col + "_mean" for col in mean_total_number_of_hard_passes.columns]
    max_total_number_of_hard_passes.columns = [col + "_max" for col in max_total_number_of_hard_passes.columns]
    # merge the two dataframes and drop duplicate columns
    merged_passes = pd.merge(mean_total_number_of_hard_passes, max_total_number_of_hard_passes, on="model")
    merged_passes = merged_passes.loc[:, ~merged_passes.columns.duplicated()]
    # drop template_mean and template_max cols
    merged_passes = merged_passes.drop(columns=["template_mean", "template_max"])
    # add the model name as a column
    merged_passes["model"] = merged_passes.index
    # save as csv in data/responses/final_merged/results
    merged.to_csv(f"data/responses/final_merged/results/merged_{file_ending}.csv", index=False)
    merged_other.to_csv(f"data/responses/final_merged/results/merged_other_{file_ending}.csv", index=False)
    merged_passes.to_csv(f"data/responses/final_merged/results/merged_passes_{file_ending}.csv", index=False)


def get_aggregated_questionnaire_hard_pass(result_file):
    """
    For each policy statement for each model and each template, compute number of hard pass for each test and
    report True if the test is passed or False otherwise
    :return:
    """
    result_file = add_paraphrase_variants(result_file)

    def get_pass(dataframe):
        return len(dataframe.binary.unique()) == 1

    def process_data(result_file):
        all_results = []
        for unique_id, df in result_file.groupby("unique_id"):
            # Prepare data subsets
            original = df[df["original"]]
            df_non_inverted = df[df["inverted"] == 0]
            # Perform tests
            test_label_inversion = get_pass(original)
            # check whether all sign values are * in df_non_inverted
            test_sign = True if len(df_non_inverted.sign.unique()) == 1 and df_non_inverted.sign.values[
                0] == "*" else False
            test_semantic_equivalence = get_pass(
                df_non_inverted[df_non_inverted["original"] | df_non_inverted["paraphrase"]])
            orig_binary = original.binary.values[0]
            test_negation = df_non_inverted[df_non_inverted["negation"]].binary.values[0] != orig_binary
            test_opposite = df_non_inverted[df_non_inverted["opposite"]].binary.values[0] != orig_binary
            # String answers preparation
            answer_types = ["original", "negation", "opposite", "p_1", "p_2", "p_3"]
            # add condition to check for nan values, that cannot be converted to int
            string_answers = ", ".join(
                [
                    f"{atype}: {'nan' if math.isnan(df_non_inverted[df_non_inverted[atype]].binary.values[0]) else int(df_non_inverted[df_non_inverted[atype]].binary.values[0])}"
                    for atype in answer_types if not df_non_inverted[df_non_inverted[atype]].empty]
            )
            # add inverted answer
            inverted_value = original[original['inverted'] == 1].binary.values[0]
            string_answers += f", inverted: {'nan' if math.isnan(inverted_value) else int(inverted_value)}"

            # Tests results
            tests_passed = [test_sign, test_label_inversion, test_semantic_equivalence, test_negation, test_opposite]
            number_of_tests_passed = sum(tests_passed)
            hard_pass = number_of_tests_passed == 5
            relative_amount_of_tests_passed = number_of_tests_passed / 5

            # Create result dictionary
            result_dic = {**{key: df[key].values[0] for key in
                             ["model_name", "model_type", "template_prompt", "template_id", "p(pos)", "p(neg)",
                              "country_code", "language_code"]},
                          "unique_id": unique_id,
                          "test_sign": test_sign,
                          "test_label_inversion": test_label_inversion,
                          "test_semantic_equivalence": test_semantic_equivalence,
                          "test_negation": test_negation,
                          "test_opposite": test_opposite,
                          "hard_pass": hard_pass,
                          "relative_amount_of_tests_passed": relative_amount_of_tests_passed,
                          "string_answers": string_answers}

            all_results.append(result_dic)

        return pd.DataFrame(all_results)

    aggr = process_data(result_file)
    return aggr


def create_aggregated_questionnaire_cross_template(source_dir):
    outdir = "data/responses/final_merged/results/all_tests_passed_dataframes"
    # read all csv files that are in source dir, they should
    template_model_based_questionnaires = glob.glob(f"{source_dir}/*.csv")
    model2results = defaultdict(list)
    for filen_name in template_model_based_questionnaires:
        model = filen_name.split("chat")[0].split("/")[-1].replace("-", "")
        template = filen_name.split("chat")[1].replace("_EN-CHAT.csv", "").replace("_", "")
        df = pd.read_csv(filen_name, sep=",")
        model2results[model].append((template, df))
    model2final_results = {}
    for model, template_results in model2results.items():
        # add template id to each df in template_results and then concatenate them
        for template, df in template_results:
            df["template_id"] = template
        merged_passes = pd.concat([df for _, df in template_results])
        all_passed = []
        # group by unique id and check whether all hard passes are true, if so then add unique id to list
        for unique_id, df in merged_passes.groupby("unique_id"):
            if df["hard_pass"].all():
                all_passed.append(unique_id)
        # get all rows that have unique id in all_passed from template_results[0][1]
        remaining_rows = template_results[0][1][template_results[0][1]["unique_id"].isin(all_passed)]
        # save the dataframe for each model in outdir
        remaining_rows.to_csv(f"{outdir}/{model}.csv", sep=",", index=False)
        # what is the size of remaining_rows?, save to model2final_results
        model2final_results[model] = len(remaining_rows)
    # create a dataframe from model2final_results
    df = pd.DataFrame(model2final_results.items(), columns=["model", "number_of_statements"])
    df.to_csv(f"{outdir}/number_of_statements.csv", sep=",", index=False)


def agreement_cross_template(result_file):
    results = {}
    for (model), df in result_file.groupby(["model_name"]):
        # drop all rows that are original == False.
        df = df[df["original"]]
        # drop inverted == 1
        df = df[df["inverted"] == 0]
        # krippendorff alpha between templates
        kripp_alpha = simpledorff.calculate_krippendorffs_alpha_for_df(df, class_col="binary",
                                                                       annotator_col="template_id",
                                                                       experiment_col="unique_id")
        # how many statements in df have the same binary answer for all templates?
        df = df.groupby("unique_id").agg({"binary": list}).reset_index()
        number_hard_passes = df[df["binary"].apply(lambda x: len(set(x)) == 1)].shape[0]
        # add to results
        results[model] = {"krippendorff_alpha": kripp_alpha, "number_hard_passes": number_hard_passes}
    # create a dataframe from results
    df = pd.DataFrame(results.items(), columns=["model", "results"])
    df["krippendorff_alpha"] = df["results"].apply(lambda x: x["krippendorff_alpha"])
    df["number_hard_passes"] = df["results"].apply(lambda x: x["number_hard_passes"])
    df = df.drop(columns=["results"])
    df.to_csv(f"data/responses/final_merged/results/agreement_cross_template.csv", sep=",", index=False)


def add_random_baseline():
    result_df = pd.read_csv("data/responses/final_merged/results/predictions_random_baseline/random_baseline.csv",
                            sep=",", dtype={"code": str})
    outpath = "data/responses/final_merged/results/dataframes_withpasses"
    # iterate overg roups of template_id dataframes
    for (temp_id), sub_df in result_df.groupby(["template_id"]):
        passed_statements = get_aggregated_questionnaire_hard_pass(sub_df)
        # save passed_statements to outpath
        model_name = "random-chat"
        passed_statements.to_csv(
            f"{outpath}/{model_name}_{temp_id}_EN-CHAT.csv", sep=",", index=False)


def create_consistency_plot():
    source_dir = "data/responses/final_merged/results/dataframes_withpasses"
    # read all csv files that are in source dir, they should
    template_model_based_questionnaires = glob.glob(f"{source_dir}/*.csv")
    # model2results = defaultdict(list)
    model2results = []
    for filen_name in template_model_based_questionnaires:
        model = filen_name.split("chat")[0].split("/")[-1].replace("-", "")
        df = pd.read_csv(filen_name, sep=",")
        relative_passes_sing_test = len(df[df["test_sign"]]) / len(df)
        relative_passes_label_inversion = len(df[df["test_label_inversion"]]) / len(df)
        relative_passes_semantic_equivalence = len(df[df["test_semantic_equivalence"]]) / len(df)
        relative_passes_negation = len(df[df["test_negation"]]) / len(df)
        relative_passes_opposite = len(df[df["test_opposite"]]) / len(df)
        relative_passes = len(df[df["hard_pass"]]) / len(df)
        results_dir = {"sign": relative_passes_sing_test,
                       "label order": relative_passes_label_inversion,
                       "semantic equivalence": relative_passes_semantic_equivalence,
                       "negation": relative_passes_negation,
                       "opposite": relative_passes_opposite,
                       "all tests": relative_passes,
                       "model": model,
                       "template_id": df["template_id"].iloc[0]}
        model2results.append(results_dir)
    # convert to dataframe
    df = pd.DataFrame(model2results)
    # now reduce over template_id, merge all values from the columns that contain "relative_passes" and create a column "mean" and "std"
    df_mean = df.groupby(["model"]).agg({"sign": "mean",
                                         "label order": "mean",
                                         "semantic equivalence": "mean",
                                         "negation": "mean",
                                         "opposite": "mean",
                                         "all tests": "mean"}).reset_index()
    df_std = df.groupby(["model"]).agg({"sign": "std",
                                        "label order": "std",
                                        "semantic equivalence": "std",
                                        "negation": "std",
                                        "opposite": "std",
                                        "all tests": "std"}).reset_index()

    melted_df = df_mean.melt(id_vars=['model'], var_name='test', value_name='mean')
    melted_df["std"] = df_std.melt(id_vars=['model'], var_name='test', value_name='std')["std"]
    df = melted_df
    # Get a list of unique tests and models
    tests = df['test'].unique()
    models = df['model'].unique()
    # print(tabulate.tabulate(df, headers="keys", tablefmt="simple"))

    # Create a figure with subplots
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 9))  # Adjusted figsize for clarity

    # Flatten axes array for easy iteration (if len(tests) is 6)
    axes = axes.flatten()

    # If only one test is present, axes is not a list, so we convert it into a list for consistency.
    if len(tests) == 1:
        axes = [axes]

    # Loop through each test and create a plot
    models_renamed = {"mistral7b": "MISTRAL-7B", "flantfxxl": "FLAN-T5-XXL", "llama27b": "LLAMA2-7B",
                      "random": "RANDOM CHOICE", "llama213b": "LLAMA2-13B", "llama270b": "LLAMA2-70B"}
    # rename models
    df["model"] = df["model"].apply(lambda x: models_renamed[x])

    for ax, test in zip(axes, tests):
        # Filter the dataframe for the current test
        test_df = df[df['test'] == test]
        # sort the models such they are in the following order: human, random, bert, gpt2, dialoGPT
        models_sorted = ["MISTRAL-7B", "FLAN-T5-XXL", "LLAMA2-7B", "LLAMA2-13B", "LLAMA2-70B", "RANDOM CHOICE"]

        test_df["model"] = pd.Categorical(test_df["model"], categories=models_sorted, ordered=True)
        test_df = test_df.sort_values(by="model")
        # add a horizontal line at 0.5
        ax.axvline(x=0.5, color='grey', linestyle='--')
        # add horizontal bars at error bar ends

        ax.errorbar(test_df['mean'], test_df['model'], xerr=test_df['std'], fmt='o', color='green', ecolor='lightgray',
                    elinewidth=3, capsize=5)
        ax.set_title(f'{test} Consistency')
        ax.set_xlim(0, 1)
        ax.set_xlabel('Mean')
        ax.set_ylabel('Model')
        ax.invert_yaxis()  # Invert y-axis to match the provided plot
        ax.grid(True)
    # Adjust layout
    plt.tight_layout()
    # Save figure
    plt.savefig(f'data/responses/final_merged/results/plots/consistency_overall.jpeg', dpi=300)


def analysis_per_country():
    source_dir = "data/responses/final_merged/results/dataframes_withpasses"
    # read all csv files that are in source dir, they should
    template_model_based_questionnaires = glob.glob(f"{source_dir}/*.csv")
    # model2results = defaultdict(list)
    model2results = []
    for filen_name in template_model_based_questionnaires:
        model = filen_name.split("chat")[0].split("/")[-1].replace("-", "")
        df = pd.read_csv(filen_name, sep=",")
        print(tabulate.tabulate(df.head(), headers="keys", tablefmt="simple"))
        country_codes = df["country_code"].unique()
        for country_code in country_codes:
            df_country = df[df["country_code"] == country_code]
            relative_passes_sing_test = len(df_country[df_country["test_sign"]]) / len(df_country)
            relative_passes_label_inversion = len(df_country[df_country["test_label_inversion"]]) / len(df_country)
            relative_passes_semantic_equivalence = len(df_country[df_country["test_semantic_equivalence"]]) / len(
                df_country)
            relative_passes_negation = len(df_country[df_country["test_negation"]]) / len(df_country)
            relative_passes_opposite = len(df_country[df_country["test_opposite"]]) / len(df_country)
            relative_passes = len(df_country[df_country["hard_pass"]]) / len(df_country)
            results_dir = {"sign": relative_passes_sing_test,
                           "label order": relative_passes_label_inversion,
                           "semantic equivalence": relative_passes_semantic_equivalence,
                           "negation": relative_passes_negation,
                           "opposite": relative_passes_opposite,
                           "all tests": relative_passes,
                           "model": model,
                           "country_code": country_code,
                           "template_id": df["template_id"].iloc[0]}
            model2results.append(results_dir)
    df = pd.DataFrame(model2results)
    df.to_csv("data/responses/final_merged/results/analysis_per_country.csv", sep=",", index=False)
    print(tabulate.tabulate(df.head(), headers="keys", tablefmt="simple"))


def error_bar_plot_consistency_country_specific_vs_countr_agnostic(questionnaire_aggregated):
    source_dir = "data/responses/final_merged/results/dataframes_withpasses"
    template_model_based_questionnaires = glob.glob(f"{source_dir}/*.csv")
    model2results = []
    # get the first csv
    df = pd.read_csv(template_model_based_questionnaires[0], sep=",")
    question_ids = df.unique_id.tolist()
    country_agnostic_information = {}
    for q_id in question_ids:
        meta_data = questionnaire_aggregated[questionnaire_aggregated["unique_id"] == q_id]
        country_agnostic = True if "country-agnostic" in meta_data["types"].iloc[0] else False
        country_agnostic_information[q_id] = country_agnostic
    for filen_name in template_model_based_questionnaires:
        model = filen_name.split("chat")[0].split("/")[-1].replace("-", "")
        df = pd.read_csv(filen_name, sep=",")
        template_type = df["template_prompt"].iloc[0]
        # add a column that indicates whether the question is country agnostic or not based on the countr_agnostic_information dictionary
        df["country_agnostic"] = df["unique_id"].map(country_agnostic_information)
        df_country_agnostic = df[df["country_agnostic"]]
        df_country_specific = df[~df["country_agnostic"]]

        relative_passes_agnostic = len(df_country_agnostic[df_country_agnostic["hard_pass"]]) / len(df_country_agnostic)
        relative_passes_specific = len(df_country_specific[df_country_specific["hard_pass"]]) / len(df_country_specific)
        results = {"mean": relative_passes_agnostic,
                   "model": model,
                   "template_id": df["template_id"].iloc[0],
                   "type": "agnostic",
                   "template_type": template_type}
        model2results.append(results)
        results = {"mean": relative_passes_specific,
                   "model": model,
                   "template_id": df["template_id"].iloc[0],
                   "type": "specific",
                   "template_type": template_type}
        model2results.append(results)

    df = pd.DataFrame(model2results)
    print(tabulate.tabulate(df.head(), headers="keys", tablefmt="simple"))

    models_renamed = {"mistral7b": "MISTRAL-7B", "flantfxxl": "FLAN-T5-XXL", "llama27b": "LLAMA2-7B",
                      "random": "RANDOM CHOICE", "llama213b": "LLAMA2-13B", "llama270b": "LLAMA2-70B"}
    # rename models
    df["model"] = df["model"].apply(lambda x: models_renamed[x])
    # drop model random choice
    df = df[df["model"] != "RANDOM CHOICE"]
    models_sorted = ["MISTRAL-7B", "FLAN-T5-XXL", "LLAMA2-7B", "LLAMA2-13B", "LLAMA2-70B"]
    df["model"] = pd.Categorical(df["model"], categories=models_sorted, ordered=True)
    # Create a plot
    fig, ax = plt.subplots()

    # Add a horizontal line at 0.5

    offset = 0.2
    # Plot error bars for both 'Agnostic' and 'Specific' cases
    colors = {'agnostic': 'green', 'specific': 'blue'}
    for (model, case), group in df.groupby(['model', 'type']):
        mean_value = group['mean'].mean()
        std_value = group['mean'].std()
        model_index = models_sorted.index(model)
        # Determine the position offset based on the case
        if case == 'agnostic':
            position = model_index - offset
        else:  # 'Specific'
            position = model_index + offset

        ax.errorbar(mean_value, position, xerr=std_value, fmt='o',
                    color=colors[case], ecolor='lightgray', elinewidth=3, capsize=5, label=case)

    # ax.set_title(f'{test} Consistency')
    ax.set_yticks(np.arange(len(models_sorted)) - offset / 2)  # Adjust the ticks to be in between the two cases
    ax.set_yticklabels(models_sorted)
    ax.invert_yaxis()  ## O Invert y-axis to match the provided plot
    # ax.grid(True)

    # Add a legend with two labels, one for each case
    # one label for each case
    personal_patch = mpatches.Patch(color=colors['agnostic'], label='agnostic')
    impersonal_patch = mpatches.Patch(color=colors['specific'], label='specific')

    # Add the legend to the plot
    ax.legend(handles=[personal_patch, impersonal_patch], loc='upper right')

    plt.tight_layout()
    plt.savefig("data/responses/final_merged/results/plots/consistency_country_specific_vs_country_agnostic.jpeg", dpi=300)
    plt.clf()

    fig, ax = plt.subplots()

    # Add a horizontal line at 0.5

    offset = 0.2
    # Plot error bars for both 'Agnostic' and 'Specific' cases
    colors = {'personal': 'green', 'impersonal': 'blue'}
    for (model, case), group in df.groupby(['model', 'template_type']):
        mean_value = group['mean'].mean()
        std_value = group['mean'].std()
        model_index = models_sorted.index(model)
        # Determine the position offset based on the case
        if case == 'personal':
            position = model_index - offset
        else:  # 'Specific'
            position = model_index + offset

        ax.errorbar(mean_value, position, xerr=std_value, fmt='o',
                    color=colors[case], ecolor='lightgray', elinewidth=3, capsize=5, label=case)

    # ax.set_title(f'{test} Consistency')
    ax.set_yticks(np.arange(len(models_sorted)) - offset / 2)  # Adjust the ticks to be in between the two cases
    ax.set_yticklabels(models_sorted)
    ax.invert_yaxis()  ## O Invert y-axis to match the provided plot
    # ax.grid(True)

    # Add a legend with two labels, one for each case
    # one label for each case, get the colors and label names from the dictionary
    # Create patch instances for each legend entry
    personal_patch = mpatches.Patch(color=colors['personal'], label='personal')
    impersonal_patch = mpatches.Patch(color=colors['impersonal'], label='impersonal')

    # Add the legend to the plot
    ax.legend(handles=[personal_patch, impersonal_patch], loc='upper right')

    plt.tight_layout()
    plt.savefig("data/responses/final_merged/results/plots/consistency_country_personal_vs_country_impersonal.jpeg", dpi=300)


def plot_spider_web():
    df = pd.read_csv("data/responses/final_merged/results/analysis_per_country.csv", sep=",")

    # first lets create one averaged over templates:

    mean_df = df.groupby(["model", "country_code"]).mean().reset_index()
    # create bins for the column "all tests", create 6 different bins, from 0-0.1, 0.1-0.2, 0.2-0.3, 0.3-0.4, 0.4-0.5, 0.5-1
    mean_df["bins"] = pd.cut(mean_df["all tests"], bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0],
                             include_lowest=True)
    mean_df["bins"] = mean_df["bins"].apply(lambda x: x.left)
    mean_df["bins"] = mean_df["bins"].apply(lambda x: 0 if x == -0.001 else x)
    fig = px.line_polar(mean_df, r='bins', theta='country_code', line_close=True,
                        color='model', range_r=[0, 0.6])
    # if we want the radar to be filled with colors comment this out
    # fig.update_traces(fill='toself')
    # save the plot as a jped
    fig.write_image("data/responses/final_merged/results/plots/spider_web_avg.jpeg", width=1000, height=1000)

    # then lets create one plot for each model, a line for each template
    for model in mean_df["model"].unique():
        model_df = df[df["model"] == model]
        model_df["bins"] = pd.cut(model_df["all tests"], bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                                  include_lowest=True)
        model_df["bins"] = model_df["bins"].apply(lambda x: x.left)
        model_df["bins"] = model_df["bins"].apply(lambda x: 0 if x == -0.001 else x)
        # save also the dataframe
        model_df.to_csv(f"data/responses/final_merged/results/country_specific/{model}_spider_web.csv", sep=",",
                        index=False)

        fig = px.line_polar(model_df, r='bins', theta='country_code', line_close=True,
                            color='template_id', range_r=[0, 0.8])
        # if we want the radar to be filled with colors comment this out
        # fig.update_traces(fill='toself')
        # save the plot as a jped
        fig.write_image(f"data/responses/final_merged/results/plots/spider_web_{model}.jpeg", width=1000, height=1000)


if __name__ == '__main__':
    pd.options.mode.chained_assignment = None  # Suppress SettingWithCopyWarning

    result_file = pd.read_csv("data/responses/final_merged/questionnaire_aggregated_en.csv", sep=",")
    result_file["model_name"] = result_file["model_name"].apply(
        lambda x: "flan-tf-xxl-chat" if x == "flan-t5-xxl" else x)
    resulf_file = add_model_type_chat(result_file)
    resulf_file = add_variante_type_info(resulf_file)

    models = resulf_file[resulf_file["chat_model"] == True]
    print(tabulate.tabulate(models.head(), headers="keys", tablefmt="simple"))
    flan  = models[models["model_name"] == "flan-tf-xxl-chat"]
    template = 0
    flan_0 = flan[flan["template_id"] == template]
    print("size of flan_0", len(flan_0))
    # create_aggregated_questionnaire_cross_template("data/responses/final_merged/results/dataframes_withpasses")
    # compute_all_results(models, "EN-CHAT")
    # agreement_cross_template(models)
    # add_random_baseline()
    # create_consistency_plot()
    # analysis_per_country()
    #error_bar_plot_consistency_country_specific_vs_countr_agnostic(models)
