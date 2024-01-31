import pandas as pd
import numpy as np
import tabulate
from collections import Counter
from statement_retrieval import StatementRetriever


def get_bootstrapped_mean(data):
    """
    Given a sample of 30 for each statement, calculate the bootstrapped probability for positive and negative answers.
    Compute the significance of the mean using a confidence interval.
    """
    n_bootstraps = 30
    # need to define this
    n = 30
    null_hypothesis = 0.5
    # Bootstrapping
    bootstrapped_positive_proportions = []
    bootstrapped_negative_proportions = []
    for _ in range(n_bootstraps):
        sample = np.random.choice(data, size=n, replace=True)
        # how many ones and zeros are in sample?
        freqs = Counter(sample)
        # Handling cases where "1" or "0" are not in the sample
        if "1" not in freqs:
            freqs["1"] = 0
        if "0" not in freqs:
            freqs["0"] = 0
        total_count = freqs["1"] + freqs["0"]
        if total_count == 0:
            bootstrapped_positive_proportions.append(0.0)
            bootstrapped_negative_proportions.append(0.0)
        else:

            bootstrapped_positive_proportions.append(freqs["1"] / total_count)
            bootstrapped_negative_proportions.append(freqs["0"] / total_count)
    # Analysis
    confidence_interval_positive = np.percentile(bootstrapped_positive_proportions, [2.5, 97.5])
    confidence_interval_negative = np.percentile(bootstrapped_negative_proportions, [2.5, 97.5])
    bootstrapped_mean_proportion_positive = np.mean(bootstrapped_positive_proportions)
    bootstrapped_mean_proportion_negative = np.mean(bootstrapped_negative_proportions)

    sign_conf_interval = not (confidence_interval_positive[0] <= null_hypothesis <= confidence_interval_positive[1]) or \
                         not (confidence_interval_negative[0] <= null_hypothesis <= confidence_interval_negative[1])

    # return the mean and whether it is significant
    return {"mean_positive": bootstrapped_mean_proportion_positive,
            "mean_negative": bootstrapped_mean_proportion_negative,
            "sign_conf_interval": sign_conf_interval}


def get_bootstrapped_mean_questionnaire(df):
    all_results = []
    for (policy_id, template_id, model), group in df.groupby(['unique_id', 'template_id', 'model_name']):
        result = get_bootstrapped_mean(group['binary_answer'].tolist())
        statement_result = {
            "model_name": model,
            "model_type": group["model_type"].iloc[0],
            "template_prompt": group["template_prompt"].iloc[0],
            "template_id": template_id,
            "country_code": group["country_code"].iloc[0],
            "unique_statement_id": policy_id,
            "p(pos)": result["mean_positive"],
            "p(neg)": result["mean_negative"],
            "sign": "*" if result["sign_conf_interval"] else "-"
        }
        all_results.append(statement_result)
    # merge all dfs
    questionnaire_result = pd.DataFrame(all_results)
    return questionnaire_result

def report_aggregated_stats(questionnaire_with_sign):
    # what is the relative amount of sign statements per model?
    # what is the relative amount of sign statements per model type?
    # what is the relative amount of sign statements per template?
    model_name_results = []
    model_type_results = []
    template_id_results = []

    # Group by 'model_name' and calculate statistics
    for model_name, group in questionnaire_with_sign.groupby('model_name'):
        model_name_results.append({
            "category": "model_name",
            "name": model_name,
            "total": len(group),
            "sign": len(group[group["sign"] == "*"]),
            "sign %": len(group[group["sign"] == "*"]) / len(group)
        })

    # Group by 'model_type' and calculate statistics
    for model_type, group in questionnaire_with_sign.groupby('model_type'):
        model_type_results.append({
            "category": "model_type",
            "name": model_type,
            "total": len(group),
            "sign": len(group[group["sign"] == "*"]),
            "sign %": len(group[group["sign"] == "*"]) / len(group)
        })

    # Group by 'template_id' and calculate statistics
    for template_id, group in questionnaire_with_sign.groupby('template_id'):
        template_id_results.append({
            "category": "template_id",
            "name": template_id,
            "total": len(group),
            "sign": len(group[group["sign"] == "*"]),
            "sign %": len(group[group["sign"] == "*"]) / len(group)
        })

    # Convert the lists to DataFrames
    model_name_df = pd.DataFrame(model_name_results)
    model_type_df = pd.DataFrame(model_type_results)
    template_id_df = pd.DataFrame(template_id_results)

    # Optionally, you can concatenate these DataFrames into a single DataFrame
    final_df = pd.concat([model_name_df, model_type_df, template_id_df], ignore_index=True)
    print(tabulate.tabulate(final_df, headers='keys', tablefmt='psql'))

if __name__ == '__main__':
    df = pd.read_csv("data/responses/final/development/samples_binary.csv", sep=",")
    print(tabulate.tabulate(df.head(), headers='keys', tablefmt='psql'))
    retriever = StatementRetriever(df)
    # get all of type "original"
    originals = retriever.get_filtered_statements(statement_type="original", translated=True)
    print(tabulate.tabulate(df.head(), headers='keys', tablefmt='psql'))
    print("size of original statements: ", len(originals))
    q = get_bootstrapped_mean_questionnaire(originals)
    report_aggregated_stats(q)
