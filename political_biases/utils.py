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

models_renamed = {"mistral-7b-chat": "MISTRAL-7b", 'flan-tf-xxl-chat': "FLAN-T5-XXL-11b", 'flan-t5-xxl-chat': "FLAN-T5-XXL-11b", 'llama-2-7b-chat': "LLAMA2-7b",
                  "random": "RANDOM", "random-chat": "RANDOM", "llama-2-13b-chat": "LLAMA2-13b", "llama-2-70b-chat": "LLAMA2-70b", "chat-gpt-3.5-turbo-0613": "GPT3.5-turbo-20b",
                  "chat-gpt-3.5-turbo-0613-chat": "GPT3.5-turbo-20b", 'alwaysagree':'alwaysAgree', 'alwaysdisagree':'alwaysDISagree'}

sorter_models ={'disagree':['alwaysDISagree', 'RANDOM','MISTRAL-7b', 'LLAMA2-7b', 'FLAN-T5-XXL-11b', 'LLAMA2-13b', "GPT3.5-turbo-20b", 'LLAMA2-70b'],
                'agree':['alwaysAgree', 'RANDOM', 'MISTRAL-7b', 'LLAMA2-7b', 'FLAN-T5-XXL-11b', 'LLAMA2-13b', "GPT3.5-turbo-20b", 'LLAMA2-70b'],
                'only_models':['MISTRAL-7b', 'LLAMA2-7b', 'FLAN-T5-XXL-11b', 'LLAMA2-13b', "GPT3.5-turbo-20b", 'LLAMA2-70b'],
                'all':['alwaysDISagree', 'alwaysAgree', 'RANDOM','MISTRAL-7b', 'LLAMA2-7b', 'FLAN-T5-XXL-11b', 'LLAMA2-13b', "GPT3.5-turbo-20b", 'LLAMA2-70b'],
                'simulations':['alwaysDISagree', 'alwaysAgree']}

color_models = {'alwaysDISagree':'violet', 'alwaysAgree':'olive', 'RANDOM':'navy', 'MISTRAL-7b':"red", 'LLAMA2-7b':"green", 'FLAN-T5-XXL-11b':"purple",
                'LLAMA2-13b':"orange", "GPT3.5-turbo-20b":"brown", 'LLAMA2-70b':"blue"}

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

policy2count={'open_foreign_policy': 25, 'liberal_economic_policy': 55, 'restrictive_financial_policy': 29, 'law_and_order': 19,
              'restrictive_migration_policy': 16, 'expanded_environ_protection': 32, 'expanded_social_welfare_state': 38,
              'liberal_society': 44}

id2idx = {'ch16': 0, 'hu2': 1, 'ch14': 2, 'pl17': 3, 'ch50': 4, 'pl16': 5, 'es19': 6, 'pl13': 7, 'ch10': 8, 'it26': 9, 'it27': 10, 'hu3': 11, 'it2': 12, 'nl4': 13, 'de37': 14, 'nl20': 15, 'nl5': 16, 'es16': 17, 'nl17': 18, 'pl11': 19, 'hu10': 20, 'hu11': 21, 'nl18': 22, 'es4': 23, 'hu18': 24, 'hu20': 25, 'ch49': 26, 'de1': 27, 'hu37': 28, 'ch6': 29, 'de3': 30, 'hu33': 31, 'ch24': 32, 'de6': 33, 'nl1': 34, 'ch18': 35, 'de8': 36, 'pl8': 37, 'ch56': 38, 'ch58': 39, 'hu40': 40, 'hu31': 41, 'hu32': 42, 'it30': 43, 'ch46': 44, 'ch55': 45, 'hu17': 46, 'ch44': 47, 'de36': 48, 'it17': 49, 'es14': 50, 'hu16': 51, 'hu1': 52, 'hu6': 53, 'ch13': 54, 'it1': 55, 'it3': 56, 'it5': 57, 'it21': 58, 'it22': 59, 'it24': 60, 'nl13': 61, 'ch15': 62, 'ch1': 63, 'hu5': 64, 'it12': 65, 'de38': 66, 'es5': 67, 'es9': 68, 'it14': 69, 'nl25': 70, 'ch36': 71, 'de24': 72, 'hu28': 73, 'it11': 74, 'nl22': 75, 'de17': 76, 'de35': 77, 'de9': 78, 'es21': 79, 'it18': 80, 'ch28': 81, 'ch43': 82, 'ch54': 83, 'de27': 84, 'hu29': 85, 'pl3': 86, 'ch11': 87, 'de18': 88, 'de29': 89, 'de32': 90, 'nl10': 91, 'nl30': 92, 'ch38': 93, 'de15': 94, 'de26': 95, 'es3': 96, 'hu14': 97, 'hu4': 98, 'ch52': 99, 'ch53': 100, 'de4': 101, 'de7': 102, 'it20': 103, 'it23': 104, 'ch42': 105, 'de10': 106, 'de28': 107, 'es8': 108, 'it13': 109, 'it16': 110, 'nl12': 111, 'ch19': 112, 'ch26': 113, 'es23': 114, 'es7': 115, 'it15': 116, 'nl9': 117, 'pl10': 118, 'ch30': 119, 'de20': 120, 'hu25': 121, 'hu9': 122, 'nl23': 123, 'nl28': 124, 'pl5': 125, 'ch37': 126, 'ch39': 127, 'hu13': 128, 'it10': 129, 'it9': 130, 'nl24': 131, 'pl9': 132, 'ch25': 133, 'ch40': 134, 'es6': 135, 'hu19': 136, 'hu21': 137, 'nl7': 138, 'pl2': 139, 'pl7': 140, 'ch27': 141, 'ch4': 142, 'es1': 143, 'hu34': 144, 'hu7': 145, 'hu8': 146, 'it19': 147, 'pl19': 148, 'pl20': 149, 'ch22': 150, 'ch41': 151, 'de11': 152, 'de30': 153, 'hu15': 154, 'hu23': 155, 'hu24': 156, 'nl15': 157, 'nl3': 158, 'ch2': 159, 'ch8': 160, 'ch9': 161, 'de2': 162, 'es15': 163, 'es18': 164, 'es2': 165, 'it25': 166, 'nl2': 167, 'pl14': 168, 'ch12': 169, 'ch32': 170, 'ch33': 171, 'ch34': 172, 'ch35': 173, 'de14': 174, 'es20': 175, 'es22': 176, 'it7': 177, 'it8': 178, 'nl21': 179, 'nl29': 180, 'pl12': 181, 'ch23': 182, 'ch47': 183, 'ch51': 184, 'ch57': 185, 'de21': 186, 'de31': 187, 'es12': 188, 'es24': 189, 'hu22': 190, 'hu38': 191, 'hu39': 192, 'nl11': 193, 'nl14': 194, 'nl26': 195, 'nl27': 196, 'pl6': 197, 'ch0': 198, 'ch17': 199, 'ch20': 200, 'ch21': 201, 'ch29': 202, 'ch31': 203, 'ch3': 204, 'ch45': 205, 'ch48': 206, 'ch59': 207, 'ch5': 208, 'ch7': 209, 'de12': 210, 'de13': 211, 'de16': 212, 'de19': 213, 'de22': 214, 'de23': 215, 'de25': 216, 'de33': 217, 'de34': 218, 'de5': 219, 'es10': 220, 'es11': 221, 'es13': 222, 'es17': 223, 'hu12': 224, 'hu30': 225, 'hu36': 226, 'it28': 227, 'it29': 228, 'it4': 229, 'it6': 230, 'nl16': 231, 'nl19': 232, 'nl6': 233, 'nl8': 234, 'pl15': 235, 'pl18': 236, 'pl1': 237, 'pl4': 238}

policy2idx = {'expanded_environ_protection': 5, 'expanded_social_welfare_state': 6, 'law_and_order': 3, 'liberal_economic_policy': 1, 'liberal_society': 7, 'open_foreign_policy': 0,
 'restrictive_financial_policy': 2, 'restrictive_migration_policy': 4}

rile_color = {"left": "lightcoral", "center": "mediumseagreen", "right": "slateblue"}

colors_agree_disagree = {"agree": "skyblue", "disagree": "lightcoral"}

test_names = {"hard_pass": "Statements that successfully passed ALL tests", "test_sign": "Statements that passed the SIGNIFICANCE test",
            "test_label_inversion":"Statements have have passed the LABEL INVERSION test", "test_opposite":"Statements that passed the SEMANTIC OPPOSITE test",
              "test_negation":"Statements that passed the NEGATION test","test_semantic_equivalence":"Statements that passed the PARAPHRASE test",
              "test_sign-test_negation":"Statements that passed the SIGNIFICANCE & NEGATION testS",
              "test_sign-test_semantic_equivalence":"Statements that passed the SIGNIFICANCE & PARAPHRASE tests",
              "test_sign-test_opposite":"Statements that passed the SIGNIFICANCE & SEMANTIC OPPOSITE tests",
              "test_sign-test_label_inversion":"Statements that passed the SIGNIFICANCE & LABEL INVERSION tests",
              "test_label_inversion-test_semantic_equivalence":"Statements that passed the LABEL INVERSION & PARAPHRASE tests",
              "test_label_inversion-test_negation":"Statements that passed the LABEL INVERSION & NEGATION tests",
              "test_label_inversion-test_opposite":"Statements that passed the LABEL INVERSION & SEMANTIC OPPOSITE tests",
              "test_sign-test_label_inversion-test_semantic_equivalence":"Statements that passed the SIGNIFICANCE & LABEL INVERSION & PARAPHRASE tests",
              False:"All statements - No reliability test considered"}

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
