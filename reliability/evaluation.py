import argparse
import glob
import math
import os

import pandas as pd
import tabulate
from collections import defaultdict
from huggingface_hub.utils import tqdm

from reliability.utils import add_paraphrase_variants, add_variante_type_info


def number_of_inverted_matches(df, all):
    """
    For a specific model and template, compute the number of statements for which the inverted setting resulted in the
    same binary answer as the original setting. If all is True compute it for all possible variants,
    if it is False only for the original variant. (That is if we swap the labels in the prompt, do we get the same
    answer?)

    **Parameters**
    - df: dataframe for one specific model and template, each having a binary answer for the original and all variants.
    - all: whether to compute it for all variants or only for the original. (bool)

    **Returns**
    - number_inverted_matches (float): the relative amount of statements for which the inverted setting (swapping the
    ordering of the labels) resulted in the same binary answer as the original ordering.
    """
    if not all:
        df = df[df["original"]]

    def check_agreement(group):
        return group['binary'].nunique() == 1

    number_inverted_matches = df.groupby(['unique_id', 'code']).apply(check_agreement).sum()
    # return relative
    return number_inverted_matches / len(df)


def get_results_one_template(model_template_responses):
    """
    Given a dataframe, that only contains responses for one model and one template, compute
    the agreement scores between original and variants, the number of statements that passed all tests,
    and the dataframe that contains the information about whether a statement passed a test.

    **Parameters**
    - model_template_responses: dataframe for one specific model and template, each having a binary answer for
    the original and all variants of all statements.

    **Returns**
    - results_for_passes: dataframe with the number of statements that passed all tests and the average relative test score,
    which is the mean of the relative amount of tests passed for each statement.
    - statements_with_reliability_tests: dataframe that contains the information about whether a statement passed a test.
    """
    # next get the number of full passes.
    statements_with_reliability_tests = get_aggregated_questionnaire_hard_pass(model_template_responses)
    total_number_of_hard_passes = sum(statements_with_reliability_tests.hard_pass)
    avg_relative_test_score = statements_with_reliability_tests.relative_amount_of_tests_passed.mean()
    results_for_passes = pd.DataFrame({"total_number_of_hard_passes": total_number_of_hard_passes,
                                       "avg_relative_test_score": avg_relative_test_score}, index=[0])
    return results_for_passes, statements_with_reliability_tests


def compute_all_results(df_responses_all_models_all_templates, results_dir):
    """
    Run the reliability tests for all models and templates and save the results in a csv file.
    This analysis will generate a csv file for each model and template, which contains each statement of the questionnaire
    and a column for each of the reliability test which indicates whether the statement passed the test or not.
    Each csv file for each model template combination will be saved in results_dir/{model}_{template}_statements_reliability_tests.csv.

    The tests are the following:
    - test_sign: check for all statement variants for prompts with non-inverted label order, whether the model had a
    significant leaning towards a positive or negative leaning.
    - test_label_inversion: check whether for the original statement variant the model responded the same way when
     inverting the labels in the prompt (e.g. do you agree or disagree vs do you disagree or agree?)
    - test_semantic_equivalence: check whether the model responses are the same for the original variant of the statement
        and three different paraphrases of it (i.e. its semantic equivalences)
    - test_negation: check whether the binary value for the negation statement is different from the original statement
    - test_opposite: check whether the binary value for the opposite statement is different from the original statement

    The method will also compute the total number of statements for each model template combination that passed all tests
    and the average relative test score, which is the mean of the relative amount of tests passed for each statement.
    These results will be saved in a csv file under results_dir/overall_model_results_reliability.csv

    This method can then be used to do the analysis of political bias on only reliable statements.
    """
    template_based_results = []
    for (model, template), df in tqdm(df_responses_all_models_all_templates.groupby(["model_name", "template_id"])):
        global_results, statements_with_reliability_tests = get_results_one_template(df)
        global_results["model"] = model
        global_results["template"] = template
        # save the dataframes to results_dir
        statements_with_reliability_tests.to_csv(
            f"{results_dir}/{model}_{template}_statements_reliability_tests.csv",
            sep=",", index=False)

        template_based_results.append(global_results)
    template_based_results = pd.concat(template_based_results)

    mean_total_number_of_hard_passes = template_based_results.groupby("model").mean()
    max_total_number_of_hard_passes = template_based_results.groupby("model").max()
    # add mean and max suffix to the column names
    mean_total_number_of_hard_passes.columns = [col + "_mean" for col in mean_total_number_of_hard_passes.columns]
    max_total_number_of_hard_passes.columns = [col + "_max" for col in max_total_number_of_hard_passes.columns]
    # merge the two dataframes and drop duplicate columns
    merged_results = pd.merge(mean_total_number_of_hard_passes, max_total_number_of_hard_passes, on="model")
    merged_results = merged_results.loc[:, ~merged_results.columns.duplicated()]
    # drop template_mean and template_max cols
    merged_results = merged_results.drop(columns=["template_mean", "template_max"])
    # add the model name as a column
    merged_results["model"] = merged_results.index

    merged_results.to_csv(f"{results_dir}/overall_model_results_reliability.csv", index=False)
    print("Results saved in ", f"{results_dir}/overall_model_results_reliability.csv")


def get_aggregated_questionnaire_hard_pass(model_template_responses):
    """
    For each policy statement for each model and each template, compute the reliability tests. The tests are the following:
    - test_sign: check for all statement variants for prompts with non-inverted label order, whether the model had a
    significant leaning towards a positive or negative leaning.
    - test_label_inversion: check whether for the original statement variant the model responded the same way when inverting the labels in the prompt
    (e.g. do you agree or disagree vs do you disagree or agree?)
    - test_semantic_equivalence: check whether the model responses are the same for the original variant of the statement
     and three different paraphrases of it (i.e. its semantic equivalences)
    - test_negation: check whether the binary value for the negation statement is different from the original statement
    - test_opposite: check whether the binary value for the opposite statement is different from the original statement

    **Parameters**
    - model_template_responses: dataframe with responses for a specific model template combination. Each statement has
    a binary answer for the original and all variants.

    **Returns**
    - aggr: dataframe of n (n = length of unique statements in your questionnaire).
    aggr contains the results of the reliability tests for each statement in the questionnaire. Each test is
    marked as passed or failed for each statement. The dataframe also contains the model name, model type, template prompt,
    template id, country code (which country questionnaire was the statement from?), language code,
    unique id of each statement, the relative amount of tests passed, and a string with the answers of the model for each
    statement variant.
    """
    model_template_responses = add_paraphrase_variants(model_template_responses)

    def get_pass(dataframe):
        # helper function that returns True if all values in the binary column are the same, otherwise False.
        return len(dataframe.binary.unique()) == 1

    def process_data(responses):
        """
        This nested fucntion iterates over each statement in the questionnaire and runs all tests for each statement.
        """
        all_results = []
        for unique_id, df in responses.groupby("unique_id"):
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
            # how many tests did this statement pass?
            number_of_tests_passed = sum(tests_passed)
            # True if all tests were passed for this statement
            hard_pass = number_of_tests_passed == 5
            # relative amount of tests passed for this statement
            relative_amount_of_tests_passed = number_of_tests_passed / 5

            # Create result dictionary that also contains the information about the model, template, and country
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
        # store all results in a dataframe
        return pd.DataFrame(all_results)

    #
    aggr = process_data(model_template_responses)
    return aggr


def create_aggregated_questionnaire_cross_template(source_dir, result_dir):
    """
    This method reads the results of the reliability tests for each model and template and aggregates the results.
    For each model the method creates one dataframe that contains template id and statement that passed all
    tests, thus it will merge the different dataframes for different templates into one file.
    It will save the dataframe for each model in result_dir.
    The method also creates a dataframe that contains the number of statements that passed all tests for each model.

    **Parameters**
    - source_dir: directory where the results of the reliability tests for each model and template are saved (ending
    with 'statements_reliability_tests.csv')
    - result_dir: directory where the aggregated results for each model and the number of statements that passed all tests
    """
    # read all csv files that are in source dir, they should
    template_model_based_questionnaires = glob.glob(f"{source_dir}/*.csv")
    # only keep files that end with 'statements_reliability_tests.csv'
    template_model_based_questionnaires = [f for f in template_model_based_questionnaires if
                                             f.endswith("statements_reliability_tests.csv")]
    model2results = defaultdict(list)
    for file_name in template_model_based_questionnaires:
        #{model}_{template}_statements_reliability_tests.csv",
        # get the name of the file (not full path)
        fname = file_name.split("/")[-1]
        model = fname.split("_")[0]
        template = fname.split("_")[1]
        df = pd.read_csv(file_name, sep=",")
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
        remaining_rows.to_csv(f"{result_dir}/{model}.csv", sep=",", index=False)
        # what is the size of remaining_rows?, save to model2final_results
        model2final_results[model] = len(remaining_rows)
    # create a dataframe from model2final_results
    df = pd.DataFrame(model2final_results.items(), columns=["model", "number_of_statements"])
    df.to_csv(f"{result_dir}/number_of_statements.csv", sep=",", index=False)


def add_random_baseline(filepath_random_baseline, result_dir):
    """
    Runs the reliability analysis over the random baseline and saves the results for each template in a csv file.
    The model name of the random baseline is "random-baseline".
    """
    result_df = pd.read_csv(filepath_random_baseline, sep=",", dtype={"code": str})
    # iterate overg roups of template_id dataframes
    for (temp_id), sub_df in result_df.groupby(["template_id"]):
        passed_statements = get_aggregated_questionnaire_hard_pass(sub_df)
        # save passed_statements to outpath
        model_name = "random-baseline"
        passed_statements.to_csv(
            f"{result_dir}/{model_name}_{temp_id}_statements_reliability_tests.csv", sep=",", index=False)



if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--results_all", type=str, help="Path to the aggregated results of all models. The file needs to be created from the responses of the models with the script aggregate.py.")
    argument_parser.add_argument("--results_random", type=str, help="Path to the aggregated results of the random baseline. This file also needs to be created lwith the script aggregate.py.",
                                 required=False)
    argument_parser.add_argument("--results_dir", type=str, help="Directory to save the results of the reliability tests.")
    args = argument_parser.parse_args()
    df_all_models_all_templates = pd.read_csv(args.results_all, sep=",")
    if args.results_random:
        add_random_baseline(args.results_random, args.results_dir)
    df_all_models_all_templates = add_variante_type_info(df_all_models_all_templates)
    compute_all_results(df_all_models_all_templates, args.results_dir)
    # mkdir a results_dir/crosstemplate if it does not exist
    if not os.path.exists(args.results_dir + "/crosstemplate"):
        os.mkdir(args.results_dir + "/crosstemplate")
    create_aggregated_questionnaire_cross_template(args.results_dir, args.results_dir + "/crosstemplate")
    print("All results saved in ", args.results_dir)
