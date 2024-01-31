import mantel
import pandas as pd
from sklearn.decomposition import PCA

parties_ches = {"de": {"CDU / CSU":"CDU", "SPD": "SPD", "FDP": "FDP", "GRÜNE": "GRUNEN", "DIE LINKE":"LINKE", "AfD": "AfD"},
                "nl": {"SP": "SP","DENK": "DENK", "ChristenUnie": "CU", "PVV": "PVV", "VVD": "VVD", "Partij voor de Dieren": "PvdD", "SGP": "SGP",
                       "CDA": "CDA", "D66": "D66", "Partij van de Arbeid": "PvdA",  "GroenLinks": "GL", "50PLUS": "50PLUS",  "Forum voor Democratie": "FvD"},
                "it": {"PD":"PD", "Lega":"LN", "M5S":"M5S", "FdI":"FdI", "FI":"FI"},
                "hu": {"FIDESZ-KDNP":"Fidesz-KDNP"},
                "pl":{"Konfederacja Wolnosc i Niepodleglosc":"Konfederacia", "Prawo i Sprawiedliwosc":"PiS",
                      "Nowa Lewica":"Lew+Wio+SLD",  #"Lewica Razem+Wiosnia+SLD"
                      "Trzecia Droga":"PSL", "Koalicja Obywatelska":"PO+Nowo"},
                "es":{"PSOE":"PSOE", "PP":"PPP", "Vox":"Vox", "Junts":"PdeCat", "ERC":"ERC"},
                "ch":{ 'Die Mitte':'die-mitte', 'FDP':'fdp_ch', 'SP':'sp', 'SVP':'svp', 'Grüne':'grüne',  'EVP':'evp'}}

ch_ches = {'die-mitte':5.9, 'fdp_ch':5.9, 'sp':2.5, 'svp':7.5, 'grüne':2.5, 'evp':5}

country_color = {"de": "red", "nl": "blue", "it": "green", "hu": "orange", "pl": "purple", "es": "brown", "ch": "pink"}

country2ches = {"de":3, "hu":23, "pl":26, "ch":36, "it":8, "nl":10, "es":5} #[ 3  5  8 10 23 26]

models_renamed = {"mistral-7b-chat": "MISTRAL-7b", 'flan-tf-xxl-chat': "FLAN-T5-XXL-11b", 'llama-2-7b-chat': "LLAMA2-7b",
                  "random": "RANDOM CHOICE", "llama-2-13b-chat": "LLAMA2-13b", "llama-2-70b-chat": "LLAMA2-70b", "chat-gpt-3.5-turbo-0613": "GPT3.5-turbo-20b"}

sorter_models = ['MISTRAL-7b', 'LLAMA2-7b', 'FLAN-T5-XXL-11b', 'LLAMA2-13b', "GPT3.5-turbo-20b", 'LLAMA2-70b']

positions = {'economy_labor':0.1, 'democracy_media_digit':0.2, 'society_ethics':0.3,
 'security_armed_forces':0.4, 'welfare_state_family':-0.1, 'foreign_relations':-0.2,
 'environ_protection':-0.3, 'education':-0.4, 'migration_integration':0.5, 'finance_taxes':-0.5,
 'energy_transportation':0.0}

policy_colors = {'economy_labor':"grey", 'democracy_media_digit':"blue", 'society_ethics':"red",
 'security_armed_forces':"orange", 'welfare_state_family':"purple", 'foreign_relations':"pink",
 'environ_protection':"green", 'education':"black", 'migration_integration':"brown", 'finance_taxes':"yellow",
 'energy_transportation':"cyan"}

policy_names = {'open_foreign_policy':" Open foreign policy",  'liberal_economic_policy':"Liberal economy",
                "restrictive_financial_policy":"Restrictive finance", "law_and_order":"Law and order",
                'restrictive_migration_policy':"Restrictive migration",  'expanded_environ_protection':"Envir. protection",
                "expanded_social_welfare_state": "Social welfare state", "liberal_society":"Liberal society"}


party2ches = {'piraten': 2.142857074737549, 'cdu': 5.857142925262451, 'csu': 7.190476417541504, 'linke': 1.428571462631226, 'afd': 9.2380952835083,
              'grunen': 3.238095283508301, 'spd': 3.61904764175415, 'dietier': 2.333333253860474, 'fdp': 6.428571224212646, 'pnv': 6.0, 'ppp': 8.066666603088379,
              'pais': 2.733333349227905, 'ehb': 1.285714268684387, 'iu': 1.866666674613953, 'cs': 7.199999809265137, 'bng': 3.142857074737549, 'psoe': 3.599999904632568,
              'podemos': 1.933333277702332, 'cc': 6.714285850524902, 'pdecat': 6.666666507720947, 'vox': 9.714285850524902, 'erc': 3.200000047683716, 'ri': 4.3125,
              'm5s': 4.777777671813965, 'si': 1.444444417953491, 'fdi': 9.052631378173828, 'pd': 3.21052622795105, 'ln': 8.789473533630371, 'svp': 7.5, 'fi': 6.947368621826172,
              'pvda': 3.615384578704834, 'fvd': 9.538461685180664, 'pvv': 8.692307472229004, 'gl': 2.307692289352417, 'cu': 5.07692289352417, 'd66': 5.153846263885498,
              'cda': 6.846153736114502, 'pvdd': 2.384615421295166, 'sp': 2.5, 'vvd': 7.615384578704834, 'sgp': 8.538461685180664, '50plus': 5.083333492279053,
              'denk': 4.363636493682861, 'mm': 4.400000095367432, 'lmp': 3.933333396911621, 'dk': 3.333333253860474, 'fidesz-kdnp': 8.333333015441895, 'mszp': 3.400000095367432,
              'jobbik': 7.733333110809326, 'e14': 3.571428537368774, 'wiosnia': 1.894736886024475, 'psl': 5.285714149475098, 'po': 5.38095235824585, 'pis': 7.550000190734863,
              'konfederacia': 9.526315689086914, 'kukiz': 7.111111164093018, 'nowo': 5.157894611358643, 'sld': 3.38095235824585, 'lewica razem': 1.277777791023254, 'die-mitte': 5.9,
              'fdp_ch': 5.9, 'grüne': 2.5, 'evp': 5, "lew+wio+sld":2.1844890117645264, "po+nowo":5.269423484802246, "psl": 5.285714149475098}

policy2count={'liberal_economic_policy': 55, 'restrictive_financial_policy': 29, 'law_and_order': 19, 'restrictive_migration_policy': 16, 'expanded_environ_protection': 32, 'expanded_social_welfare_state': 38, 'liberal_society': 44}


def compute_mantel(cat_arr, text_arr):
    """ Compute Mantel test between two distance matrices.
    :returns correlation, pvalues"""

    r, pval, z = mantel.test(cat_arr, text_arr, perms=10000, method='spearman', tail='two-tail')
    return r, pval

def diagonal_to_zero(df):
    for i in range(len(df)):
        df.iloc[i, i] = 0
    return df


def run_pca(mat, n_comp, parties):
    pca = PCA(n_components=n_comp)
    pca_transformed = pca.fit_transform(mat)
    df = pd.DataFrame(pca_transformed)
    df["party"]=parties
    return dict(zip(df.party, df[0]))

def convert_answers(x):
    if x == "agree" or x == "completely agree" or x == "rather agree":
        return 1
    elif x == "disagree" or x == "completely disagree" or x == "rather disagree":
        return 0
    else:
        return -1

def convert_unique_id(x):
    return int(x[2:])


def retrieve_template_type():
    template = pd.read_csv("./PolBiases/data/prompt_instructions/classification_prompts/en_clf_templates_final.csv")
    template_gpt = pd.read_csv("./PolBiases/data/prompt_instructions/classification_prompts/en_gpt_templates_final.csv")
    template2type = dict(zip(template.template_id.tolist(), template.prompt_type.tolist()))
    template2type_gpt = dict(zip(template_gpt.inversion_id.tolist(), template_gpt.prompt_type.tolist()))
    return template2type, template2type_gpt
