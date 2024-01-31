import pandas as pd
import glob
import shutil
from collections import defaultdict
import matplotlib.pyplot as plt


code2answer = {100:"Yes", 75:"Rather yes", 25: "Rather no", 0: "No"}

def swiss_questions_lang(lang):
    ids={}
    questions = pd.read_excel(f"./data/ch/23_ch_nr-questions.{lang}.xlsx").to_dict("records")
    for question in questions:
        if question["type"]=="4-options":
            ids[question["ID_question"]]=question["question"]
    return ids

def read_swiss_data():
    cand_answers=[]
    ids_de = swiss_questions_lang("de")
    ids_en = swiss_questions_lang("en")
    candidates = pd.read_excel("./data/ch/23_ch_nr-candidates.de.xlsx").to_dict("records")
    for dic in candidates:
        for i in ids_de:
            if dic["answer_"+str(i)] in code2answer.keys():
                position = code2answer[dic["answer_"+str(i)] ]
            else:
                position = None
            cand_answers.append((dic["ID_candidate"], dic["party_short"], position, i, ids_de[i], ids_en[i]))
    df = pd.DataFrame(cand_answers, columns=["candidateID", "party", "position", "statementID", "statement", "translation"])
    return df

def store_orignal_ids():
    df = read_swiss_data()
    map2code = {enu:i for i, enu in enumerate(set(df.statementID.tolist()))}
    print(map2code)
    df["statementID"] = df["statementID"].apply(lambda x: map2code[x])
    f = open("./data/ch/statements2ID.txt", "w")
    for k,v in map2code.items():
        f.write(str(k)+"\t"+str(v)+"\n")
    f.close()
    df.to_csv("./PolBiases/data/vaa/ch.csv", index=False)
# store_orignal_ids()

def swiss_multilingual_unique(lang):
    results = []
    f = open("./data/ch/statements2ID.txt", "r")
    map2code = {}
    for line in f:
        k, v = line.rstrip().split("\t")
        map2code[int(k)] = int(v)
    ids = swiss_questions_lang(lang)
    ids_en = swiss_questions_lang("en")
    for id in ids.keys():
        results.append((map2code[id], ids[id], ids_en[id]))
    df = pd.DataFrame(results, columns=["statementID", "statement", "translation"])
    df.to_csv(f"./PolBiases/data/vaa/ch_{lang}_unique.csv", index=False)

# swiss_multilingual_unique("it")

def voting(x):
    if x == 4:
        return 1  # agnostic
    else:
        return 0  # specific

def country_specific_voting():
    df = pd.read_csv("./data/country_specific_vs_agnostic.csv")
    df["votes"]=df["votes"].apply(lambda x: voting(x))
    dic = dict(zip([i.lstrip().rstrip() for i in df.statement.tolist()], df.votes.tolist()))
    return dic


def hu_voting():
    """Add column for country-specific vs agnostic voting. 0 is specific and 1 is agnostic."""
    dic_voting = country_specific_voting()
    f="./data/hu_unique.csv"
    df = pd.read_csv(f)
    labels = []
    for i in df.translation.tolist():
        i = i.lstrip().rstrip()
        labels.append(dic_voting[i])
    df["country_agnostic"] = labels
    if "country-specific" in df.columns:
        df.drop(columns=["country-specific"], inplace=True)
    if 'country specific' in df.columns:
        df.drop(columns=['country specific'], inplace=True)
    print(df.columns)
    print(df)
    df.to_csv(f, index=False)

# hu_voting()
# exit()


def fix_voting():
    """Add column for country-specific vs agnostic voting. 0 is specific and 1 is agnostic."""
    dic_voting = country_specific_voting()
    files = glob.glob("./PolBiases/data/vaa/*_unique.csv")

    for f in files:
        if "ch" not in f:
            print(f)
            df = pd.read_csv(f)
            labels = []
            for i in df.translation.tolist():
                i = i.lstrip().rstrip()
                labels.append(dic_voting[i])
            df["country_agnostic"] = labels
            if "country-specific" in df.columns:
                df.drop(columns=["country-specific"], inplace=True)
            if 'country specific' in df.columns:
                df.drop(columns=['country specific'], inplace=True)
            print(df.columns)
            df.to_csv(f, index=False)


def add(x):
    x = "Question: " + x + " <sep> Answer:"
    return x

def check_data():
    df = pd.read_csv("./PolBiases/data/vaa/all_unique.csv")
    tmp = df.iloc[[i for i, code in enumerate(df.ID.tolist()) if "1000000"in code or "1000010"in code]]
    print(len(tmp))

    duplicate = df[df.duplicated('statement', keep = False)].sort_values(by=['statement'])
    for id, stat in zip(duplicate.ID.tolist(), duplicate.statement.tolist()):
        print(id, stat)
# check_data()
# exit()
# files = glob.glob("./PolBiases/data/vaa/*_unique.csv")
# df = pd.DataFrame()
# for f in files:
#     if "all" not in f:
#         tmp = pd.read_csv(f)
#         df = pd.concat([df, tmp], axis=0)
# print(len(df))


def translation_prompts():
    countries = ["it", "hu", "nl", "es", "pl", "fr"]
    files = glob.glob("./PolBiases/data/prompt_instructions/en_*txt")
    for lang in countries:
        for f in files:
            new_name = f.replace("en_", lang+"_")
            shutil.copy(f, new_name)
# translation_prompts()

sentence = {"en": "Create three paraphrases for each of the following sentences:",
            "de": "Erstellen Sie drei Paraphrasen für jeden der folgenden Sätze:",
           "fr": "Créez trois paraphrases pour chacune des phrases suivantes:",
           "it": "Crei tre parafrasi per ciascuna delle seguenti frasi:",
           "es": "Cree tres paráfrasis para cada una de las frases siguientes:",
           "hu": "Készítsen három parafrázist az alábbi mondatok mindegyikéhez:",
           "nl": "Maak drie parafrases voor elk van de volgende zinnen:",
           "pl": "Proszę utworzyć trzy parafrazy dla każdego z poniższych zdań:"}

def create_sets_for_paraphrases(country, type_stat):
    df = pd.read_csv(f"./PolBiases/data/vaa/{country}_unique.csv")
    f=open(f"./data/new_paraphrases/{country}_{type_stat}_para.txt", "w")
    for i,st in enumerate(df[type_stat].tolist()):
        if i==0 or i%5==0:
            f.write("\n\n")
            f.write(sentence[country]+"\n")
        f.write(st+"\n")
    f.close()

def create_source_language_file():
    countries = ["nl", "es", "de", "es", "pl", "it", "hu"]
    countries = ["de"]
    types = ["statement", "opposite"]
    for country in countries:
        for type in types:
            create_sets_for_paraphrases(country, type)
# create_source_language_file()


def fix_swiss_dataset():
    lang="fr"
    df_lang = pd.read_csv(f"./PolBiases/data/vaa/ch_{lang}_unique.csv")
    cols = {59:"statement", 119:"negation", 179:"opposite"}
    f=open(f"./data/ch_eng_{lang}.txt", "r")
    l=[]
    for i, line in enumerate(f.readlines()):
        line = line.rstrip("\n")
        if i in cols.keys():
            l.append(line)
            df_lang[cols[i]]=l
            l=[]
        else:
            l.append(line)
    print(len(l))
    print(df_lang)
    df_lang.to_csv(f"./PolBiases/data/vaa/ch_{lang}_unique.csv", index=False)


def create_all_templates_inverted():
    files = glob.glob("./PolBiases/data/prompt_instructions/inverted/*.txt")
    results=[]
    c=0
    for f in files:
        f_name = f.split("/")
        for l in open(f, "r"):
            l = l.rstrip("\n")
            if "generation" in f_name[-1]:
                model_type = "generation"
            else:
                model_type = "chat"
            results.append((str(c)+"_inverted", l, f_name[-1].split("_")[0], model_type, f_name[-1].replace("en_instructions_", "").replace(".txt", "")))
            c+=1
    df = pd.DataFrame(results, columns=["template_id", "template", "lang", "model_type", "prompt_type"])
    print(df)
    df.to_csv("./PolBiases/data/prompt_instructions/all_templates_inverted.csv", index=False)


def sample_optimization():
    df = pd.read_csv(f"./PolBiases/data/vaa/all_unique.csv")
    df["country"] = [i[:2] for i in df.ID.tolist()]
    df["original"] = [1 if i[-10] == "1" else 0 for i in df.ID.tolist()]
    df["country_agnostic"] = [1 if i[-5] == "1" else 0 for i in df.ID.tolist()]
    df["translated"] = [1 if i[-4] == "1" else 0 for i in df.ID.tolist()]

    final = pd.DataFrame()
    for c in set(df.country.tolist()):
        tmp = df[df.country==c]
        for ag in set(df.country_agnostic.tolist()):
            tmp2 = tmp[(tmp.country_agnostic==ag)&(tmp.original==1)&(tmp.translated==1)]
            sample = tmp2.sample(n=5, random_state=1)
            final = pd.concat([final, sample], axis=0)
            print(sample)
    final.to_csv("./PolBiases/data/vaa/all_unique_sampled.csv", index=False)


def remove_paraphrases_non_original():
    df = pd.read_csv(f"./PolBiases/data/vaa/all_unique.csv")
    print(len(df))
    ids = [i for i, id in enumerate(df.ID.tolist()) if (id[-6]=="1" or id[-7]=="1") and int(id[-8])>0]
    df = df.drop(ids)
    df.to_csv(f"./PolBiases/data/vaa/all_unique.csv", index=False)

# remove_paraphrases_non_original()

def total_originals():
    df = pd.read_csv(f"./PolBiases/data/vaa/all_unique.csv")
    tmp = [i for i, id in enumerate(df.ID.tolist()) if "1000000" in id or "1000010" in id]
    print(len(tmp))
# total_originals()

def total_english():
    df = pd.read_csv(f"./PolBiases/data/vaa/all_unique.csv")
    tmp = [id for i, id in enumerate(df.ID.tolist()) if int(id[-4])==0]
    dic = defaultdict(int)
    for i in tmp:
        dic[i.split("_")[0]]+=1
    print(dic)
    percountry=[]
    for k in dic.keys():
        percountry.append(dic[k]/6)
        print(k, int(dic[k]/6))
    print(sum(percountry))
# total_english()


def inspecting():
    df = pd.read_csv(f"./PolBiases/data/vaa/all_unique.csv")
    tmp = [id for i, id in enumerate(df.ID.tolist()) if int(id[-4])==1 and "hu_" in id and int(id[-8])>0]
    print(len(tmp))
# inspecting()

# lang="fr"
# df = pd.read_csv(f"./PolBiases/data/prompt_instructions/classification_prompts/{lang}_clf_templates_final.csv")
# df_en = pd.read_csv(f"./PolBiases/data/prompt_instructions/classification_prompts/en_clf_templates_final.csv")
# df["final_template_id"] = df_en["final_template_id"]
# df["lang"] = [lang]*len(df)
# print(df)
# df.to_csv(f"./PolBiases/data/prompt_instructions/classification_prompts/{lang}_clf_templates_final.csv", index=False)

def sample_10():
    df = pd.read_csv(f"./PolBiases/data/vaa/all_unique.csv")
    tmp = [i for i, id in enumerate(df.ID.tolist()) if int(id[-4])==0 and "de_" in id and int(id[-8])==0 and int(id[-6])==0 and int(id[-7])==0]
    tmp = df.iloc[tmp][:10]
    tmp.to_csv(f"./PolBiases/data/vaa/de_unique_sample.csv", index=False)

# df = pd.read_csv(f"./PolBiases/data/vaa/all_unique.csv")
# tmp = [i for i, id in enumerate(df.ID.tolist()) if int(id[-4])==0]
# print(len(tmp))

def convert_answers(x):
    x = x.lower()
    if x == "yes" or x == "rather yes":
        return "agree"
    else:
        return "disagree"

ch_ches = {'die mitte':5.9, 'fdp':5.9, 'sp':2.5, 'svp':7.5, 'grüne':2.5, 'evp':5}

def transform_swiss_votes():
    results=[]
    df = pd.read_csv("./PolBiases/data/vaa/ch_individual.csv")
    df["party"]=[i.lower() for i in df.party.tolist()]
    df = df[df["party"].isin(ch_ches.keys())]
    df = df[df['position'].notna()]
    df["position"]=df.position.apply(lambda x:convert_answers(x))
    for st in df.statementID.unique():
        for p in df.party.unique():
            tmp = df[(df.statementID==st)&(df.party==p)]
            dic = tmp.position.value_counts().to_dict()
            for i in ["agree", "disagree"]:
                if i not in dic.keys():
                    dic[i]=0
            perc_agree = round(100*(dic["agree"]/(dic["agree"]+dic["disagree"])), 2)
            perc_disagree = round(100 * (dic["disagree"] / (dic["agree"] + dic["disagree"])), 2)
            if dic["agree"]>dic["disagree"]:
                results.append((p, st, "agree", perc_agree, dic["agree"], dic["disagree"]))
            else:
                results.append((p, st, "disagree", perc_disagree, dic["agree"], dic["disagree"]))

    df = pd.DataFrame(results, columns=["party", "statementID", "position", "perc_agree", "agree", "disagree"])
    # df.to_csv("./PolBiases/data/vaa/ch_parties.csv", index=False)

    # plot perc_agree column in a histogram
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 4))
    for p, ax in zip(df.position.unique(), axes):
        tmp = df[df.position==p]
        ax.hist(tmp.perc_agree.tolist(), bins=10)
        ax.set_xlabel(f"Percentage of {p} per statement")
        ax.set_ylabel("Number of statements")
    plt.tight_layout()
    plt.savefig(f'data/responses/swiss_candidates.jpeg', dpi=600)
    plt.show()

transform_swiss_votes()

# df = pd.read_csv("./PolBiases/data/vaa/ch.csv")
# print(df.party.unique())