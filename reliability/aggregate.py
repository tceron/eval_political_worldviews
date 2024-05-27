import argparse

import pandas as pd
import numpy as np
from collections import Counter

import tabulate
import tqdm


def get_bootstrapped_mean(model_responses, n_bootstraps=1000, null_hypothesis_positive=0.55,
                          null_hypothesis_negative=0.45):
    """
    This function calculates the bootstrapped probability for positive and negative answers given a sample of data.
    It computes the significance of the mean using a confidence interval. If there are only non-mappable answers
    in a sample (e.g. because the model responded neither yes, nor no), this response is not considered in the analysis.
    **Parameters:**
    - model_responses (list): A list of binary answers (0 or 1) for a single statement.
    - n_bootstraps (int): The number of bootstraps to perform.
    - null_hypothesis_positive (float): The null hypothesis for the positive class.
    - null_hypothesis_negative (float): The null hypothesis for the negative class.
    **Returns:**
    - dict: A dictionary containing the mean positive and negative probabilities, the variance, the standard deviation,
    the effective sample size, and whether the mean is significant. The effective sample size is the number of non-NaN
    values in the data (i.e. the number of non-"not_applied" answers).
    """
    # Bootstrapping
    bootstrapped_positive_proportions = []
    bootstrapped_negative_proportions = []
    # sample size
    n = len(model_responses)
    # effective sample size is the numnber of non-NaN in the data
    n = n - Counter(model_responses)["not_applied"]
    # if there are no valid responses from the model, everything is set to 0.
    if n == 0:
        return {"mean_positive": 0.0, "mean_negative": 0.0, "sign_conf_interval": False, "effective_sample_size": 0}
    # remove all "not_applied" from data
    model_responses = [x for x in model_responses if x != "not_applied"]
    model_responses = [int(x) for x in model_responses]
    for _ in range(n_bootstraps):
        sample = np.random.choice(model_responses, size=n, replace=True)
        # how many 1s and 0s are there in the sample?
        count_1 = np.count_nonzero(sample)
        count_0 = len(sample) - count_1
        sum = count_1 + count_0
        bootstrapped_positive_proportions.append(count_1 / sum)
        bootstrapped_negative_proportions.append(count_0 / sum)
    # Analysis
    confidence_interval_positive = np.percentile(bootstrapped_positive_proportions, [2.5, 97.5])
    confidence_interval_negative = np.percentile(bootstrapped_negative_proportions, [2.5, 97.5])
    bootstrapped_mean_proportion_positive = np.mean(bootstrapped_positive_proportions)
    bootstrapped_mean_proportion_negative = np.mean(bootstrapped_negative_proportions)

    sign_conf_interval = not (
            confidence_interval_positive[0] <= null_hypothesis_positive <= confidence_interval_positive[1]) or \
                         not (confidence_interval_negative[0] <= null_hypothesis_negative <=
                              confidence_interval_negative[1])

    # return the mean and whether it is significant
    return {"mean_positive": bootstrapped_mean_proportion_positive,
            "mean_negative": bootstrapped_mean_proportion_negative,
            "sign_conf_interval": sign_conf_interval, "effective_sample_size": n}


def get_bootstrapped_mean_questionnaire(df, sample_size):
    """
    This function calculates the bootstrapped mean for each statement in the questionnaire. It groups the data by
    statement_id, inverted, model_name, and template_id. For each group, it calculates the bootstrapped mean for the
    positive and negative answers, the variance, the standard deviation, the effective sample size, and the binary
    answer. The binary answer is set to None if the mean positive and mean negative probabilities are equal, 1 if the
    mean positive probability is greater than the mean negative probability, and 0 otherwise.
    **Parameters:**
    - df (pd.DataFrame): A DataFrame containing the responses to the questionnaire.
    **Returns:**
    - pd.DataFrame: A DataFrame containing the bootstrapped mean for each statement in the questionnaire.
    """
    all_results = []
    for (statement_id, inverted, model, template_id), group in tqdm.tqdm(
            df.groupby(['statement_id', 'inverted', 'model_name', 'template_id'])):
        assert (len(group) == sample_size), "We expect sample_size answers per statement"
        result = get_bootstrapped_mean(group['binary_answer'].tolist())
        # get variance and standard deviation of group['binary_answer'].tolist()
        count_not_applied = Counter(group['binary_answer'].tolist())["not_applied"]
        answers_without_not_applied = [int(x) for x in group['binary_answer'].tolist() if x != "not_applied"]
        variance = np.var(answers_without_not_applied)
        std = np.std(answers_without_not_applied)

        statement_result = {
            "model_name": model,
            "model_type": group["model_type"].iloc[0],
            "template_prompt": group["template_prompt"].iloc[0],
            "template_id": template_id,
            "inverted": inverted,
            "statement_id": statement_id,
            "p(pos)": result["mean_positive"],
            "p(neg)": result["mean_negative"],
            "variance": variance,
            "std": std,
            "effective_sample_size": result["effective_sample_size"],
            "count_not_applied": count_not_applied,
            "binary": None if result["mean_positive"] == result["mean_negative"] else (
                1 if result["mean_positive"] > result["mean_negative"] else 0),
            "sign": "*" if result["sign_conf_interval"] else "-"
        }
        all_results.append(statement_result)
    # merge all dfs
    questionnaire_result = pd.DataFrame(all_results)
    return questionnaire_result


def aggregate(questionnaire_with_model_responses, outfile, sample_size=30):
    """
    This function aggregates the responses to the questionnaire. It calculates the bootstrapped mean for each statement
    in the questionnaire and adds the original, paraphrase, opposite, and negation information to the DataFrame. The
    aggregated DataFrame is saved to a CSV file.
    **Parameters:**
    - questionnaire_with_model_responses (pd.DataFrame): A DataFrame containing the responses to the questionnaire.
    - outfile (str): The path to save the aggregated DataFrame.
    """
    aggr = get_bootstrapped_mean_questionnaire(questionnaire_with_model_responses, sample_size)
    aggr.to_csv(outfile, index=False)


def create_random_sample(outdir, example_frame, template_ids=None):
    """
    This function creates a random sample of 30 binary answers (either 1 or 0) for each statement in the questionnaire.
    It calculates the bootstrapped mean for each statement and saves the results to a CSV file.
    **Parameters:**
    - outdir (str): The path to save the results.
    - example_frame (pd.DataFrame): A DataFrame containing the example statements.
    - template_ids (list): A list of template IDs to create random samples for.
    """
    # specify dtype of "code" column to be str
    example_frame = pd.read_csv(example_frame, sep=",", dtype={"code": str})
    if template_ids is None:
        template_ids = [0, 4, 5, 20, 24, 25]
    data = []
    for template_id in template_ids:
        for idx, row in tqdm.tqdm(example_frame.iterrows()):
            # compy the row
            new_row = row.copy()
            sample = np.random.randint(2, size=30)
            result = get_bootstrapped_mean(sample)
            count_not_applied = 0
            variance = np.var(sample)
            std = np.std(sample)
            model = "random"
            model_type = "chat"
            template_prompt = None
            inverted = row["inverted"]
            statement_id = row["statement_id"]
            statement_result = {
                "model_name": model,
                "model_type": model_type,
                "template_prompt": template_prompt,
                "template_id": template_id,
                "inverted": inverted,
                "statement_id": statement_id,
                "p(pos)": result["mean_positive"],
                "p(neg)": result["mean_negative"],
                "variance": variance,
                "std": std,
                "effective_sample_size": result["effective_sample_size"],
                "count_not_applied": count_not_applied,
                "binary": None if result["mean_positive"] == result["mean_negative"] else (
                    1 if result["mean_positive"] > result["mean_negative"] else 0),
                "sign": "*" if result["sign_conf_interval"] else "-"
            }
            new_row.update(statement_result)
            data.append({**new_row})
    random_result_df = pd.DataFrame(data)
    random_result_df.to_csv(outdir + "/random_baseline.csv", index=False, sep=",")


if __name__ == '__main__':
    arparse = argparse.ArgumentParser()
    arparse.add_argument("--questionnaire_with_model_responses", type=str,
                         help="The path to the questionnaire with model responses.")
    arparse.add_argument("--out_dir", type=str, help="The path to save the results.")
    arparse.add_argument("--sample_size", type=int, default=30, help="The sample size for each statement.")
    args = arparse.parse_args()

    questionnaire_with_model_responses_df = pd.read_csv(args.questionnaire_with_model_responses, sep=",")

    outfile = args.out_dir + "/aggregated_results.csv"
    aggregate(questionnaire_with_model_responses=questionnaire_with_model_responses_df, outfile=outfile,
              sample_size=args.sample_size)

    template_ids = questionnaire_with_model_responses_df["template_id"].unique()
    single_model = questionnaire_with_model_responses_df["model_name"].unique()[0]
    # get an example df with unique statement_ids for which we can create random samples
    example_df_with_unique_statement_ids = questionnaire_with_model_responses_df[
        questionnaire_with_model_responses_df["model_name"] == single_model]
    create_random_sample(outdir=args.out_dir, example_frame=example_df_with_unique_statement_ids,
                         template_ids=template_ids)
    print("saved random samples in ", args.out_dir)
    print("saved aggregated results in ", outfile)
