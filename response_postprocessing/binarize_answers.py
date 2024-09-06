import glob
import re
import pandas as pd
import tabulate
from pathlib import Path
from collections import defaultdict

prompt_formats = ["do you agree or disagree", "between support and oppose", "are you pro or con", "do you support or oppose",
                  "\"favorable\" or \"detrimental", "\"beneficial\" or \"harmful\"", "pro or con", "support or oppose", "\"agree\" or \"disagree\"",
                  "neither \"beneficial\" nor \"harmful\"", "neither \"favorable\" nor \"detrimental\"", "neither \"pro\" nor \"con\""]


def binarise_answer(answer):
    answer = answer.lower().replace("\n", " ")

    for prompt_format in prompt_formats:
        if prompt_format in answer:
            answer = answer.replace(prompt_format, "")

    word_pattern = re.compile(r'[a-z]+')
    words = word_pattern.findall(answer)
    if len(words) == 0:
        return None
    for i, word in enumerate(words):
        if word in {'beneficial', 'advantageous', 'favorable', "agree", "support", "pro"}:
            if i != 0:
                if words[i - 1] == "not":  # if the word is not, then it is negative
                    return 0
                elif words[i - 1] == "t":
                    if i - 2 >= 0:
                        if words[i - 2] == "don":  # if the word is don't, then it is positive
                            return 0
                    else:
                        return 1
                else:
                    return 1
            else:
                return 1
        if word in {'detrimental', 'disadvantageous', 'harmful', "disagree", "oppose", "con"}:
            if i != 0:
                if words[i - 1] == "not":  # if the word is not, then it is positive
                    return 1
                elif words[i - 1] == "t":
                    if i - 2 >= 0:
                        if words[i - 2] == "don":  # if the word is don't, then it is positive
                            return 1
                    else:
                        return 0
                else:
                    return 0
            else:
                return 0
    return None

def binerarize_answers_two_sided(files):

    results = []
    for file in files:

        if "gpt" in file:
            df_temp = pd.read_csv('../data/prompt_instructions/en_gpt_templates_final.csv')
        else:
            df_temp = pd.read_csv('../data/prompt_instructions/en_clf_templates_final.csv')
        id2typeprompt = dict(zip(df_temp.template_id, df_temp.prompt_type))

        df_list = pd.read_csv(file).to_dict("records")
        c = defaultdict(int)

        for dic in df_list:
            for i in [j for j in dic.keys() if "response" in j]:

                model = i.replace("_response", "")
                c[str(dic["template_id"])+"|"+dic["statement_id"]+"|"+model] += 1  # first 30 will be considered non-inverted

                # check if it is flan-t5
                if model.startswith("flan") or model.startswith("gpt"):
                    answer = dic[i]
                    model_type = "chat"
                else:
                    model_type = i.replace("_response", "").split("-")[-1]
                    if dic["prompt"].startswith("<s>"):
                        answer = dic[i][len(dic["prompt"]) - 3:]
                    else:
                        answer = dic[i][len(dic["prompt"]):]

                binarised_answer = binarise_answer(answer)
                inverted = 0 if c[str(dic["template_id"])+"|"+dic["statement_id"]+"|"+model] <= 30 else 1

                results.append((dic["template_id"], dic["statement_id"], inverted, i.replace("_response", ""),
                                model_type, id2typeprompt[dic["template_id"]],
                                int(binarised_answer) if binarised_answer is not None else "not_applied", answer))

    df = pd.DataFrame(results, columns=["template_id", "statement_id", "inverted", "model_name", "model_type", "template_prompt",
                                        "binary_answer", "answer"])
    print(tabulate.tabulate(df[:100], headers="keys", tablefmt="psql"))
    return df

# def show_answer_counts_per_template(df):
#     """
#     This method reads in the selected prompts and computes the number of positive, negative and not mapped answers (raw and relative proportions).
#     It was relevant to pr-filter the 3 personal vs 3 impersonal best promots for chat and generative models.
#     """
#     prompts = pd.read_csv("./data/prompt_instructions/classification_prompts/en_clf_templates.csv")
#     id2prompt = dict(zip(prompts.template_id, prompts.template))
#     all_dfs = []
#     # show the counts for each class in binary_answer per model_name, per template_id, per model_type
#     for template in df.template_id.unique():
#         answers_for_that_template = df[df.template_id == template]
#         for model in answers_for_that_template.model_name.unique():
#             # get the counts for each class in binary_answer
#             sub_df = answers_for_that_template[(answers_for_that_template.model_name == model)]
#             prompt = id2prompt[template]
#             # print(template_prompt)
#             # print(tabulate.tabulate(sub_df.head(), headers="keys", tablefmt="psql"))
#             counts = sub_df.binary_answer.value_counts()
#             counts_normalized = sub_df.binary_answer.value_counts(normalize=True)
#             # if "not_applied" not in counts add it with value 0
#             if "not_applied" not in counts:
#                 counts["not_applied"] = 0
#                 counts_normalized["not_applied"] = 0
#             # the following information we need: (model_name, template_id, template_prompt, model_type, count(1), count(0), count(not_applied)
#             mini_df = pd.DataFrame(
#                 [(model, template, prompt, sub_df.model_type.unique()[0], sub_df.template_prompt.unique()[0], counts[1],
#                   counts[0],
#                   counts["not_applied"], counts_normalized[1], counts_normalized[0], counts_normalized["not_applied"])],
#                 columns=["model_name", "template_id", "prompt", "model_type", "template_type", "count(1)", "count(0)",
#                          "count(not_applied)", "normalized(1)", "normalized(0)", "normalized(not_applied)"])
#             all_dfs.append(mini_df)
#     merged_df = pd.concat(all_dfs)
#     # sort by model_name
#     merged_df = merged_df.sort_values(by=["model_name"])
#
#
#     merged_only_chat = merged_df[merged_df.model_type == "chat"]
#     avg_per_template_chat = merged_only_chat.groupby(["template_id"]).mean().reset_index()
#     avg_per_template_chat = avg_per_template_chat.sort_values(by=["normalized(not_applied)"])
#     merged_only_chat = merged_only_chat.sort_values(by=["normalized(not_applied)"])
#     merged_only_chat.to_csv("data/responses/samples/merged_only_chat.csv", index=False)
#     avg_per_template_chat.to_csv("data/responses/samples/avg_per_template_chat.csv", index=False)
#
#     merged_only_generative = merged_df[merged_df.model_type == "generation"]
#     merged_only_generative = merged_only_generative.sort_values(by=["normalized(not_applied)"])
#     avg_per_template_generative = merged_only_generative.groupby(["template_id"]).mean().reset_index()
#     avg_per_template_generative = avg_per_template_generative.sort_values(by=["normalized(not_applied)"])
#     merged_only_generative.to_csv("data/responses/samples/merged_only_generative.csv", index=False)
#     avg_per_template_generative.to_csv("data/responses/samples/avg_per_template_generative.csv", index=False)
#
#     print(tabulate.tabulate(merged_only_chat, headers="keys", tablefmt="psql"))
#     print(tabulate.tabulate(avg_per_template_chat, headers="keys", tablefmt="psql"))
#
#     print(tabulate.tabulate(merged_only_generative, headers="keys", tablefmt="psql"))
#     print(tabulate.tabulate(avg_per_template_generative, headers="keys", tablefmt="psql"))


def main():

    files = glob.glob("../data/responses/prompt_answers/*.csv")
    df = binerarize_answers_two_sided(files)
    folder_path = f"../data/responses/prompt_answers/"
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    df.to_csv(folder_path+"answer_binary.csv", index=False)
    print(f"---- Binarized answers saved in {folder_path}----")


if __name__ == "__main__":
    """ 
    Creates a daaframe file with binary answers for the two-sided prompts. 
    The dataframe has the following columns: model_type, template_id, statement_id, binary_answer
    """
    main()
