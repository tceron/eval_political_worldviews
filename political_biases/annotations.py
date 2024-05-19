import pandas as pd
from collections import defaultdict
import tabulate
from sklearn.metrics import cohen_kappa_score
import io
from utils import country2ches

def applytab(row):
    print('\t'.join(map(str,row.values)))

def get_paraphrase(x):
    x = list(x)
    x[-8] = "1"
    return "".join(x)

def get_opposite(x):
    x = list(x)
    x[-7], x[-10] = "1", "0"
    return "".join(x)

def get_negation(x):
    x = list(x)
    x[-6], x[-10] = "1", "0"
    return "".join(x)

def get_originals(df):
    tmp = [id for i, id in enumerate(df.ID.tolist()) if
           "1000011" in id]  # [id for i, id in enumerate(df.ID.tolist()) if int(id[-4])==1 and int(id[-8])>0]
    tmp = df[df.ID.isin(tmp)].sample(n=50)
    return tmp


def prepare_survey(df):
    tmp = get_originals(df).sample(n=50)
    l = []
    for i in tmp.ID.tolist():
        p = get_paraphrase(i)
        o = get_opposite(i)
        n = get_negation(i)
        for id in [p, o, n]:
            l.append((id, df[df.ID == id]["statement"].tolist()[0]))
    df_annot = pd.DataFrame(l, columns=["ID", "statement"])
    df_final = pd.concat([df_annot, tmp[["ID", "statement"]]], ignore_index=True).sample(len(df_annot)+len(tmp))
    df_final.insert(0, "id", list(range(len(df_final))))
    print(len(df_final))
    df_final.to_csv("./data/annotations/statement_consistency/annotation_policy.csv", index=False)
    df_final = df_final.drop(columns=["ID"])
    df_final["agree"]=[None]*len(df_final)
    df_final["disagree"]=[None]*len(df_final)
    print(len(df_final))
    print(len(df_final.drop_duplicates(subset=["statement"])))
    print(df_final)
    df_final.to_csv("./data/annotations/statement_consistency/survey_policy_issues.csv", index=False)

# df = pd.read_csv("./PolBiases/data/vaa/all_unique.csv")
# prepare_survey(df)

def get_result_annotations(file_name):

    df = pd.read_csv(f"./data/annotations/statement_consistency/annotated/{file_name}.csv")
    df_ori = pd.read_csv("./data/annotations/statement_consistency/annotation_policy.csv")
    df["ID"] = df_ori["ID"]
    df["annotation"] = [1 if i == "x" or i == "X" else 0 for i in df.agree.tolist()]

    dic = defaultdict(list)
    original = get_originals(df)
    for mod in ["paraphrase", "opposite", "negation"]:
        for i in original.ID.tolist():
            new_id = get_paraphrase(i) if mod == "paraphrase" else get_opposite(i) if mod == "opposite" else get_negation(i)
            dic[mod].append(df[df.ID == new_id]["annotation"].tolist()[0])
    dic["original"].extend(original.annotation.tolist())
    dic["ID"].extend(original.ID.tolist())
    df = pd.DataFrame(dic)
    return df

def calculate_agreement():
    results = []
    df_final = pd.DataFrame()
    for annotator in ["1", "2", "3", "4", "5", "6"]:
        df = get_result_annotations(f"survey_policy_issues_{annotator}")
        df["annotator"] = [annotator]*len(df)
        df_final = pd.concat([df_final, df], ignore_index=True)
        for mod in  ["paraphrase", "opposite", "negation"]:
            results.append((mod, annotator, round(cohen_kappa_score(df.original.tolist(), df[mod].tolist()), 3)))
    # df_final.to_csv(f"./data/annotations/statement_consistency/annotated/survey_policy_domains.csv", index=False)

    df_kappa = pd.DataFrame(results, columns=["original_vs", "annotator", "cohens_k"])
    df_kappa["mean"] = df_kappa.groupby(["original_vs"])["cohens_k"].transform("mean")
    df_kappa["std"] = df_kappa.groupby(["original_vs"])["cohens_k"].transform("std")
    df_kappa = df_kappa.drop_duplicates(subset=["original_vs"]).drop(columns=["cohens_k", "annotator"])
    df_kappa["mean"] = df_kappa["mean"].apply(lambda x: round(x, 2))
    df_kappa["std"] = df_kappa["std"].apply(lambda x: round(x, 2))

    # print(tabulate.tabulate(df_kappa, headers='keys', tablefmt='psql'))
    with io.StringIO() as buffer:
        df_kappa.to_csv(buffer, sep=' ')
        print(buffer.getvalue())

# calculate_agreement()


def read_annotations_smartvote():
    df = pd.read_csv("./data/annotations/spiderweb/annotations_vaas.csv").dropna(subset="policy_domain").sort_values(by="policy_domain")
    df_all = pd.read_csv("./PolBiases/data/vaa/all_unique.csv")
    dic = dict(zip(df_all.ID.tolist(), df_all.statement.tolist()))
    l = [dic[i] for i in df.ID.tolist()]
    df["statement"]=l
    df.to_csv("./data/annotations/spiderweb/smartvote.csv")

# read_annotations_smartvote()

############################################################################################
#Spider web annotations

policy_names = {'open_foreign_policy':" Open foreign policy",  'liberal_economic_policy':"Liberal economic policy",
                "restrictive_financial_policy":"Restrictive financial policy", "law_and_order":"Law and order",
                'restrictive_migration_policy':"Restrictive migration policy",  'expanded_environ_protection':"Expanded environment protection",
                "expanded_social_welfare_state": "Expanded social welfare state", "liberal_society":"Liberal society"}

def compare_annotations_spiderweb():
    df = {}
    df[1] = pd.read_csv("./data/annotations/spiderweb/annotations_vaas_gloria.csv").fillna(0)
    df[2] = pd.read_csv("./data/annotations/spiderweb/annotations_vaas_tanise.csv").fillna(0)
    df[3] = pd.read_csv("./data/annotations/spiderweb/annotations_vaas_max.csv").fillna(0)
    results = []
    for d1, d2 in zip([1, 2, 3], [2, 3, 1]):
        for col in df[d1].columns[4:]:
            kappa = round(cohen_kappa_score(df[d1][col].tolist(), df[d2][col].tolist()), 3)
            results.append((col, kappa))

        # print(tabulate.tabulate(df.set_index("ID"), headers='keys', tablefmt='psql'))
    df = pd.DataFrame(results, columns=["category", "cohens_k"])
    df["mean_cohen"] = df.groupby(["category"])["cohens_k"].transform("mean")
    df["std_cohen"] = df.groupby(["category"])["cohens_k"].transform("std")
    df = df.drop_duplicates(subset=["category"]).drop(columns=["cohens_k"])
    df["mean_cohen"] = df["mean_cohen"].apply(lambda x: round(x, 3))
    df["std_cohen"] = df["std_cohen"].apply(lambda x: round(x, 3))
    df["category"] = df["category"].apply(lambda x: policy_names[x])
    # print(tabulate.tabulate(df_kappa, headers='keys', tablefmt='psql'))
    with io.StringIO() as buffer:
        df.to_csv(buffer, sep=' ')
        print(buffer.getvalue())

# compare_annotations_spiderweb()

def translate_smart_vote():

    df = pd.read_csv("./data/annotations/spiderweb/annotations_vaas.csv")
    df_all = pd.read_csv("./PolBiases/data/vaa/all_unique.csv")
    dic_all = dict(zip(df_all.ID.tolist(), df_all.statement.tolist()))
    dic = dict(zip(df.ID.tolist(), df.statement.tolist()))
    en = []
    for i in df.ID.tolist():
        if i in dic_all.keys():
            en.append(dic_all[i])
        else:
            en.append(dic[i])
    df["statement"] = en
    print(df)
    df.to_csv("./data/annotations/spiderweb/annotations_vaas.csv", index=False)

# translate_smart_vote()

# df =  pd.read_csv("./data/annotations/spiderweb/annotations_vaas_tanise.csv")
# # dropnas if all columns are nan from column 4 onwards
# df = df.dropna(subset=df.columns[4:], how='all')
# print(df)

# df = pd.read_csv("./PolBiases/data/vaa/pl.csv")
# print(df.party.unique())

def check_polish_alliances():
    df_ches = pd.read_csv("data/1999-2019_CHES_dataset_means(v3).csv")
    df_ches = df_ches[(df_ches.year == 2019) & (df_ches["country"]==country2ches["pl"])]
    party2ches = dict(zip([i.lower() for i in df_ches.party.tolist()], df_ches.lrgen.tolist()))
    print(party2ches)

def counts_policy_annotated():
    df =  pd.read_csv("./data/annotations/spiderweb/annotations_vaas_gold.csv")
    df["unique_id"] = df.ID.apply(lambda x: "".join(x.split("_")[:2]))
    dic = {}
    for i in df.columns[4:-1]:
        # counts of 1s and -1s in each column
        sum = df[i].value_counts().to_dict()
        dic[i] = sum[1]+sum[-1]
    return dic

def convert(x):
    if x == -1:
        return 1
    else:
        return x

# df = pd.read_csv("./data/responses/policy_stances.csv")
# for (model), group in df.groupby("model"):
#     print(model)
#     group["position"]=group["position"].apply(lambda x: convert(x))
#     group["sum"]=group.groupby(["policy_domain"])["answer_model"].transform("sum")
#     group = group.drop_duplicates(subset=["policy_domain"])
#     print(group)