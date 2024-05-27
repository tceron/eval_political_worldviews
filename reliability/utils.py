import pandas as pd

from .statement_retrieval import StatementRetriever


def add_paraphrase_variants(df):
    """
    Add three columns, one for each paraphrase (p1, p2, p3), which indicate whether the statement was one of the
    three paraphrases and which one.

    **Parameters**
    - df: dataframe for one specific model and template, each having a binary answer for the original and all
    paraphrases.

    **Returns**
    - df: dataframe with the three columns for the paraphrase variants.
    """
    # Extract paraphrase variant from 'code' and convert to integer
    df.loc[:, "paraphrase_variant"] = df["code"].apply(lambda x: int(x[2]) if x[2] != "0" else 0)

    # Convert paraphrase variant into three columns (p1, p2, p3) and drop column "p_0"
    df = pd.concat([df, pd.get_dummies(df["paraphrase_variant"], prefix="p")], axis=1)
    df = df.drop(columns=["p_0"])

    # Convert the dummies into boolean values using .loc
    df.loc[:, "p_1"] = df["p_1"].astype(bool)
    df.loc[:, "p_2"] = df["p_2"].astype(bool)
    df.loc[:, "p_3"] = df["p_3"].astype(bool)

    return df


def add_variante_type_info(questionnaire_with_sign):
    """Add original, paraphrase, opposite, negation info to the dataframe as a single column"""
    statement_retriever = StatementRetriever(questionnaire_with_sign)
    statement_retriever.dataframe["original"] = statement_retriever.dataframe["types"].apply(lambda x: "original" in x)
    statement_retriever.dataframe["paraphrase"] = statement_retriever.dataframe["types"].apply(
        lambda x: "paraphrase" in x)
    statement_retriever.dataframe["opposite"] = statement_retriever.dataframe["types"].apply(lambda x: "opposite" in x)
    statement_retriever.dataframe["negation"] = statement_retriever.dataframe["types"].apply(lambda x: "negation" in x)
    return statement_retriever.dataframe
