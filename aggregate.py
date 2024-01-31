from pathlib import Path

import pandas as pd
import numpy as np
import tabulate
from collections import Counter
import tqdm
from statement_retrieval import StatementRetriever


def get_bootstrapped_mean(data):
    """
    Given a sample of 30 for each statement, calculate the bootstrapped probability for positive and negative answers.
    Compute the significance of the mean using a confidence interval. If there are only non mappable answers in a sample,
    probability for each class is 0.0.
    """
    n_bootstraps = 1000

    null_hypothesis_positive = 0.55
    null_hypothesis_negative = 0.45
    # Bootstrapping
    bootstrapped_positive_proportions = []
    bootstrapped_negative_proportions = []
    # number of "not_applied" in data?
    n = len(data)
    # effective sample size is the numnber of non-NaN in the data
    n = n - Counter(data)["not_applied"]
    if n == 0:
        return {"mean_positive": 0.0, "mean_negative": 0.0, "sign_conf_interval": False, "effective_sample_size": 0}
    # remove all "not_applied" from data
    data = [x for x in data if x != "not_applied"]
    data = [int(x) for x in data]
    for _ in range(n_bootstraps):
        sample = np.random.choice(data, size=n, replace=True)
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


def get_bootstrapped_mean_questionnaire(df):
    all_results = []
    for (statement_id, inverted, model, template_id), group in tqdm.tqdm(
            df.groupby(['statement_id', 'inverted', 'model_name', 'template_id'])):
        assert (len(group) == 30), "We expect 30 answers per statement"
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


def add_variante_type_info(questionnaire_with_sign):
    """Add original, paraphrase, opposite, negation info to the dataframe as a single column"""
    statement_retriever = StatementRetriever(questionnaire_with_sign)
    statement_retriever.dataframe["original"] = statement_retriever.dataframe["types"].apply(lambda x: "original" in x)
    statement_retriever.dataframe["paraphrase"] = statement_retriever.dataframe["types"].apply(
        lambda x: "paraphrase" in x)
    statement_retriever.dataframe["opposite"] = statement_retriever.dataframe["types"].apply(lambda x: "opposite" in x)
    statement_retriever.dataframe["negation"] = statement_retriever.dataframe["types"].apply(lambda x: "negation" in x)
    return statement_retriever.dataframe


def report_aggregated_stats(questionnaire_with_sign):
    questionnaire_with_sign = add_variante_type_info(questionnaire_with_sign)
    # rename  flan-tf-xxl with flan-tf-xxl-chat
    questionnaire_with_sign["model_name"] = questionnaire_with_sign["model_name"].apply(
        lambda x: "flan-tf-xxl-chat" if x == "flan-t5-xxl" else x)

    model_name_results = []
    template_id_results = []
    non_average_results = []

    for variant_type in ["original", "paraphrase", "opposite", "negation"]:
        # get sub_df for each variant_type
        sub_df = questionnaire_with_sign[questionnaire_with_sign[variant_type]]
        # Group by 'model_name' and calculate statistics
        for model_name, group in sub_df.groupby('model_name'):
            model_name_results.append({
                "category": "model_name",
                "variant_type": variant_type,
                "name": model_name,
                "total": len(group),
                "sign": len(group[group["sign"] == "*"]),
                "sign %": len(group[group["sign"] == "*"]) / len(group)
            })

        # Group by 'template_id' and calculate statistics
        for (template_id, inverted), group in sub_df.groupby(['template_id', 'inverted']):
            template_id_results.append({
                "category": "template_id",
                "variant_type": variant_type,
                "name": template_id,
                "inverted": inverted,
                "total": len(group),
                "sign": len(group[group["sign"] == "*"]),
                "sign %": len(group[group["sign"] == "*"]) / len(group)
            })

        for (model_name, template_id, inverted), group in sub_df.groupby(
                ["model_name", "template_id", "inverted"]):
            non_average_results.append({
                "model_name": model_name,
                "template_id": template_id,
                "variant_type": variant_type,
                "inverted": inverted,
                "total": len(group),
                "sign": len(group[group["sign"] == "*"]),
                "sign %": len(group[group["sign"] == "*"]) / len(group)
            })

    # Convert the lists to DataFrames
    model_name_df = pd.DataFrame(model_name_results)
    template_id_df = pd.DataFrame(template_id_results)
    non_average_df = pd.DataFrame(non_average_results)

    print("------- MODEL NAME --------")
    # split model_name into "name" contains chat or not
    # combine the column variant_type and sign % into new columns, the name of the column is the variant_type and the
    # value is the sign %
    model_name_df = model_name_df.pivot(index='name', columns='variant_type', values='sign %').reset_index()
    # resort the columns, so that the original column is right after name column
    model_name_df = model_name_df[["name", "original", "paraphrase", "opposite", "negation"]]

    model_name_df["contains_chat"] = model_name_df["name"].apply(lambda x: "chat" in x)
    model_name_df["contains_chat"] = model_name_df["contains_chat"].apply(lambda x: "chat" if x else "generation")
    model_name_chat = model_name_df[model_name_df["contains_chat"] == "chat"]
    model_name_generation = model_name_df[model_name_df["contains_chat"] == "generation"]
    print(tabulate.tabulate(model_name_chat, headers='keys', tablefmt='psql', floatfmt=".3f"))
    print(tabulate.tabulate(model_name_generation, headers='keys', tablefmt='psql', floatfmt=".3f"))

    print("------- TEMPLATE ID --------")
    # combine the column variant_type, inverted and sign % into new columns, the name of the column is the variant_type + inverted and the
    # value is the sign %
    template_id_df = template_id_df.pivot(index='name', columns=['variant_type', 'inverted'],
                                          values='sign %').reset_index()
    # rename column name to template_id
    template_id_df = template_id_df.rename(columns={"name": "template_id"})
    # replace 1 with 'inverted' and 0 with 'non_inverted'
    template_id_df.columns = template_id_df.columns.map('{0[0]}_{0[1]}'.format)
    print(tabulate.tabulate(template_id_df, headers='keys', tablefmt='psql', floatfmt=".3f"))

    print("------- NON AVERAGE --------")
    # combine the column variant_type, inverted and sign % into new columns, the name of the column is the variant_type + inverted and the
    # value is the sign %
    non_average_df = non_average_df.pivot(index=['model_name', 'template_id', 'inverted'],
                                          columns='variant_type', values='sign %').reset_index()
    # combine inverted with negation, opposite, original, paraphrase into new columns, the name of the column is the inverted + variant_type and the
    # value is the sign %
    non_average_df = non_average_df.pivot(index=['model_name', 'template_id'],
                                          columns='inverted',
                                          values=['negation', 'opposite', 'original', 'paraphrase']).reset_index()
    non_average_df.columns = non_average_df.columns.map('{0[0]}_{0[1]}'.format)
    # remove _ from model_name and template_id
    # rename column model_name_ to model_name
    non_average_df = non_average_df.rename(columns={"model_name_": "model_name"})
    # rename column template_id_ to template_id
    non_average_df = non_average_df.rename(columns={"template_id_": "template_id"})
    non_average_df["contains_chat"] = non_average_df["model_name"].apply(lambda x: "chat" in x)
    non_average_df["contains_chat"] = non_average_df["contains_chat"].apply(lambda x: "chat" if x else "generation")
    non_average_chat = non_average_df[non_average_df["contains_chat"] == "chat"]
    non_average_generation = non_average_df[non_average_df["contains_chat"] == "generation"]
    # remove contains_chat column
    non_average_chat = non_average_chat.drop(columns=["contains_chat"])
    non_average_generation = non_average_generation.drop(columns=["contains_chat"])
    print(tabulate.tabulate(non_average_chat, headers='keys', tablefmt='psql', floatfmt=".3f"))
    print(tabulate.tabulate(non_average_generation, headers='keys', tablefmt='psql', floatfmt=".3f"))
    # save the results to data/responses/final_merged/aggregated/overview_statistics
    folder_path = f"data/responses/final_merged/aggregated/overview_statistics/"
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    model_name_chat.to_csv(folder_path + "model_name_chat.csv", index=False)
    model_name_generation.to_csv(folder_path + "model_name_generation.csv", index=False)
    template_id_df.to_csv(folder_path + "template_id.csv", index=False)
    non_average_chat.to_csv(folder_path + "non_average_chat.csv", index=False)
    non_average_generation.to_csv(folder_path + "non_average_generation.csv", index=False)


def aggregate():
    df = pd.read_csv("data/responses/en_answers_binary.csv", sep=",")
    print(tabulate.tabulate(df.head(), headers='keys', tablefmt='psql'))
    aggr = get_bootstrapped_mean_questionnaire(df)
    aggr.to_csv("data/responses/final_merged/questionnaire_aggregated_en.csv", index=False)


def create_random_sample():
    outdir = "data/responses/final_merged/results/predictions_random_baseline"
    template_ids = [0, 4, 5, 20, 24, 25]
    # specify dtype of "code" column to be str
    example_frame = pd.read_csv("data/responses/final_merged/results/template_example_df.csv", sep=",", dtype={"code": str})
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
    print(tabulate.tabulate(random_result_df.head(), headers='keys', tablefmt='psql'))


if __name__ == '__main__':
    # aggregate()
    # df = pd.read_csv("data/responses/final_merged/questionnaire_aggregated_en.csv", sep=",")
    # cut off all rows that have count_not_applied > 15
    # df = df[df["count_not_applied"] <= 15]
    # report_aggregated_stats(df)
    # create a random sample of 30 data points, either 1 or 0
    create_random_sample()
