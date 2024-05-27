import argparse
import math

import pandas as pd
import simpledorff
from sklearn.metrics import cohen_kappa_score

from reliability.utils import add_paraphrase_variants


def compute_krippendorff_alpha(df):
    """
    Given the dataframe for one specific model, for one specific template, compute the Krippendorff's alpha between
    the answers between original and all paraphrases.

    **Parameters**
    - df: dataframe for one specific model and template, each having a binary answer for the original and all
    paraphrases.

    **Returns**
    - Krippendorff's alpha (float): the agreement between the answers for the original and all paraphrases.
    """
    # we are only interested in the rows with original == True or paraphrase == True
    df = df[df["original"] | df["paraphrase"]]
    # the answers are specified in the column "binary", the annotators in "code" (each original and paraphrase have a
    # different code), and the unique_id is the unique identifier for each statement
    kripp_alpha = simpledorff.calculate_krippendorffs_alpha_for_df(df, class_col="binary",
                                                                   annotator_col="code",
                                                                   experiment_col="unique_id")
    return kripp_alpha


def compute_cohens_kappa(df, col1, col2):
    """
    Given the dataframe for one specific model, for one specific template, compute the cohens kappa between the answers
    of two variants (specified in col1 and col2). The cohens kappa can only be computed between exactly two variants.
    For example we can compute the cohens kappa between the original and the negation for which we would expcet a
    systematic disagreement (negative kappa) if a model is coherent and reliable.

    **Parameters**
    - df: dataframe for one specific model and template, each having a binary answer for the original, all
    paraphrases, negation and opposite.
    - col1: column name for the first variant (e.g. original)
    - col2: column name for the second variant (e.g. p1)

    **Returns**
    - cohens kappa (float): the agreement between the answers for the two variants.
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


def normalized_entropy(annotations, num_labels):
    """
    Given a list of annotations, compute the normalized entropy of the annotations.

    **Parameters**
    - annotations: list of annotations
    - num_labels: number of possible labels

    **Returns**
    - normalized entropy of the annotations (float)
    """
    annotations = [x for x in annotations if str(x) != 'nan']
    data_count = Counter(annotations)
    total_count = len(annotations)
    entropy = 0.0

    for count in data_count.values():
        probability = count / total_count
        entropy -= probability * math.log2(probability)

    normalized_entropy = entropy / math.log2(num_labels)

    return normalized_entropy


def get_kappa_scores_human_sample(df):
    """
    Compute the kappa scores for the human sample between the original and the negation, opposite, and paraphrase
    (same score as done for the humans)
    **Parameters**
    - df: dataframe for one specific model and template.

    **Returns**
    - a dictionary with the kappa scores for the human sample between the original and the negation, opposite, and paraphrase.
    """
    human_annotated = pd.read_csv("data/human_annotations/annotations_vaas_gold.csv", sep=",")
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


def get_aggregated_questionnaire_with_disagreement(df):
    """
    For a specific model and template, compute the normalized entropy for each statement, looking at the model responses
    for the original and paraphrase variants. The normalized entropy is high, if the answers are very diverse, and low,
    if the answers are similar. The entropy is 0 if the model responded the same to the original statement and all
    paraphrases.

    **Parameters**
    - df: dataframe for one specific model and template, each having a binary answer for the original and all
    paraphrases.

    **Returns**
    - aggregated: dataframe with the unique_id, the binary answers for the original and paraphrases, the entropy, and
    whether there is full agreement between the answers.
    """
    # for one model, one template, and one inverted setting, aggregate the answers for each statement and compute the
    # entropy and whether there is "full agreement" (i.e., all variants resulted on the same binary answer)
    df = df[df["original"] | df["paraphrase"]]
    aggregated = df.groupby("unique_id").agg({"binary": list}).reset_index()
    # Check for full agreement and calculate entropy
    aggregated["full_agreement"] = aggregated["binary"].apply(lambda x: len(set(x)) == 1)
    aggregated["entropy"] = aggregated["binary"].apply(normalized_entropy, num_labels=2)
    aggregated.loc[aggregated["full_agreement"], "entropy"] = 0
    return aggregated


def get_all_agreement_scores_one_template(df):
    """
    Given a dataframe, that only contains one model and one template, compute all agreement scores between:
    - original and inverted label order
    - original and paraphrase variants
    - original and negation
    - original and opposite
    - humans: original and negation, original and opposite, original and paraphrase

    **Parameters**
    - df: dataframe for one specific model and template, each having a binary answer for the original and all
    variants.

    **Returns**
    - all_agreement_scores: dataframe with all agreement scores, containing the following:
    * kappa_between_labels_original_and_inverted: agreement between original and inverted label order
    * krippendorff_alpha_semantic_equivalence: agreement between original and paraphrase variants
    * kappa_original_paraphrase_full_df: average kappa agreement between original and paraphrase variants
    * kappa_original_opposite_full_df: agreement between original and opposite
    * kappa_original_negation_full_df: agreement between original and negation
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
    krippendorff_alpha_semantic_equivalence = compute_krippendorff_alpha(df)
    # compute pair-wise cohens kappa for the part that was annotated by the humans
    pair_wise_kappas_hs = get_kappa_scores_human_sample(df)
    kappa_original_negation_full_df = compute_cohens_kappa(df, "original", "negation")
    kappa_original_opposite_full_df = compute_cohens_kappa(df, "original", "opposite")
    kappa_original_paraphrase_full_df = (compute_cohens_kappa(df, "original", "p_1") +
                                         compute_cohens_kappa(df, "original", "p_2") +
                                         compute_cohens_kappa(df, "original", "p_3")) / 3
    all_agreement_results = {"kappa_between_labels_original_and_inverted": kappa_between_labels_original_and_inverted,
                             "krippendorff_alpha_semantic_equivalence": krippendorff_alpha_semantic_equivalence,
                             "hs-original-paraphrase": pair_wise_kappas_hs["original-paraphrase"],
                             "hs-original-negation": pair_wise_kappas_hs["original-negation"],
                             "hs-original-opposite": pair_wise_kappas_hs["original-opposite"],
                             "kappa_original_negation_full_df": kappa_original_negation_full_df,
                             "kappa_original_opposite_full_df": kappa_original_opposite_full_df,
                             "kappa_original_paraphrase_full_df": kappa_original_paraphrase_full_df}
    df_agreement_scores = pd.DataFrame(all_agreement_results, index=[0])
    return df_agreement_scores


def agreement_across_templates(all_model_template_responses, result_dir):
    """
    This method computes the Krippendorff's alpha for the agreement between different templates for each model. It also
    computes the number of statements that have the same response across different templates. This is only done for the
    original variant of the statement, and only for the prompt with non-inverted labels. The results for each model will
    be saved to a dataframe in 'agreement_across_templates.csv' in the specified result directory.

    **Parameters**
    - all_model_template_responses: A DataFrame containing the responses to the questionnaire for different models,
    for different prompt templates.
    - result_dir: The directory to save the results.
    """
    results = {}
    for (model), df in all_model_template_responses.groupby(["model_name"]):
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
        number_same_response = df[df["binary"].apply(lambda x: len(set(x)) == 1)].shape[0]
        # add to results
        results[model] = {"krippendorff_alpha": kripp_alpha, "number_same_response": number_same_response}
    # create a dataframe from results
    df = pd.DataFrame(results.items(), columns=["model", "results"])
    df["krippendorff_alpha"] = df["results"].apply(lambda x: x["krippendorff_alpha"])
    df["number_same_response"] = df["results"].apply(lambda x: x["number_same_response"])
    df = df.drop(columns=["results"])
    df.to_csv(f"{result_dir}/agreement_across_templates.csv", sep=",", index=False)


def get_aggregated_results_agreement(df_responses_all_models_all_templates, result_dir):
    """
    The following method creates aggregated result files for agreement between responses for different statement variants,
    e.g. original and its paraphrases, original and negation, original and opposite.
    The results are computed for the sample annotated by humans (as comparison) and for the full dataset.

    **Parameters**
    - df_responses_all_models_all_templates: A DataFrame containing the responses to the questionnaire for different models,
    for different prompt templates.
    - result_dir: The directory to save the results.
    """
    agreement_results_human = []
    agreement_results_full = []
    for (model, template), df in df_responses_all_models_all_templates.groupby(["model_name", "template_id"]):
        agreement = get_all_agreement_scores_one_template(df)
        # split agreement in two dataframes, one only contains columns with "hs"
        agreement_human_sample = agreement.filter(regex="hs")
        agreement_full_dataset = agreement.drop(columns=agreement_human_sample.columns)
        # compute the mean of the absolute values of agreement
        agreement_human_sample["mean_absolute_agreement"] = agreement_human_sample.abs().mean(axis=1)
        agreement_full_dataset["mean_absolute_agreement"] = agreement_full_dataset.abs().mean(axis=1)

        # add model and template to the results
        agreement_human_sample["model"] = model
        agreement_human_sample["template"] = template
        agreement_full_dataset["model"] = model
        agreement_full_dataset["template"] = template

        agreement_results_human.append(agreement_human_sample)
        agreement_results_full.append(agreement_full_dataset)

    # create aggregated dataframes
    agreement_results_human = pd.concat(agreement_results_human)
    agreement_results_full = pd.concat(agreement_results_full)

    human_sample_mean_over_templates = agreement_results_human.groupby("model").mean()
    human_sample_max_mean_absolute_agreement = agreement_results_human.groupby("model").max()
    # add mean and max suffix to the column names
    human_sample_mean_over_templates.columns = [col + "_mean" for col in human_sample_mean_over_templates.columns]
    human_sample_max_mean_absolute_agreement.columns = [col + "_max" for col in
                                                        human_sample_max_mean_absolute_agreement.columns]

    # merge the two dataframes and drop duplicate columns
    merged = pd.merge(human_sample_mean_over_templates, human_sample_max_mean_absolute_agreement, on="model")
    merged = merged.loc[:, ~merged.columns.duplicated()]
    # drop template_mean and template_max cols
    merged = merged.drop(columns=["template_mean", "template_max"])
    # add the model name as a column
    merged["model"] = merged.index

    # then for the other sample
    full_mean_over_templates = agreement_results_full.groupby("model").mean()
    full_max_mean_absolute_agreement = agreement_results_full.groupby("model").max()
    # add mean and max suffix to the column names
    full_mean_over_templates.columns = [col + "_mean" for col in full_mean_over_templates.columns]
    full_max_mean_absolute_agreement.columns = [col + "_max" for col in
                                                full_max_mean_absolute_agreement.columns]

    # merge the two dataframes and drop duplicate columns
    merged_full = pd.merge(full_mean_over_templates, full_max_mean_absolute_agreement, on="model")
    merged_full = merged_full.loc[:, ~merged_full.columns.duplicated()]
    # drop template_mean and template_max cols
    merged_full = merged_full.drop(columns=["template_mean", "template_max"])
    # add the model name as a column
    merged_full["model"] = merged_full.index

    # save the results into result_dir
    merged.to_csv(f"{result_dir}/agreement_human_sample.csv", sep=",", index=False)
    merged_full.to_csv(f"{result_dir}/agreement_full_questionnaire.csv", sep=",", index=False)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--data_path", type=str, help="Path to the data file.")
    arg_parser.add_argument("--result_dir", type=str, help="Directory to save the results.")
    args = arg_parser.parse_args()

    df_responses_all_models_all_templates = pd.read_csv(args.data_path, sep=",")
    # compute the agreement across templates
    agreement_across_templates(df_responses_all_models_all_templates, args.result_dir)
    # compute the agreement for the full dataset and the human sample
    get_aggregated_results_agreement(df_responses_all_models_all_templates, args.result_dir)

    print(f"Agreement results saved successfully in {args.result_dir}")
