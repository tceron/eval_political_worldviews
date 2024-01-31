import pandas as pd


class StatementRetriever:

    def __init__(self, dataframe):
        """
        Initialize StatementRetriever with a pandas DataFrame.
        Example usage:
        retriever = StatementRetriever(dataframe)
        retriever.get_filtered_statements(statement_type="original", translated=True)
        retriever.get_filtered_statements(statement_type="paraphrase", translated=True)
        retriever.get_filtered_statements(statement_type="adversarial", translated=True)
        if you want to after-wards filter for country-agnostic statements, you can do it like this:
        df = retriever.get_filtered_statements(statement_type="original", translated=True)
        retriever = StatementRetriever(df)
        retriever.get_filtered_statements(statement_type="country-agnostic", translated=True)
        ...

        Args:
            dataframe (pd.DataFrame): DataFrame containing statements and IDs.
        """
        self.dataframe = dataframe
        self._prepare_dataframe()

    def _prepare_dataframe(self):
        """ Prepare the dataframe by extracting relevant information from ID and code columns. """
        split_ids = self.dataframe["statement_id"].str.split("_")
        self.dataframe["country_code"] = split_ids.str[0]
        self.dataframe["statement_id"] = split_ids.str[1]
        self.dataframe["code"] = split_ids.str[2]
        self.dataframe["language_code"] = split_ids.str[3]
        self.dataframe["unique_id"] = self.dataframe["country_code"] + self.dataframe["statement_id"]
        self.dataframe["types"] = self.dataframe["code"].apply(self.id_to_type)
        self.dataframe["translated"] = self.dataframe["code"].apply(self.id2translation_status)

    def get_filtered_statements(self, statement_type, translated=False):
        """
        Returns filtered statements based on type and translation status.

        Args:
            statement_type (str): The type of statement to filter (e.g., 'original', 'adversarial').
            translated (bool): If True, returns statements in English, else in the original language.

        Returns:
            pd.DataFrame: Filtered DataFrame.
        """

        # Filter the DataFrame based on the presence of type t
        filtered_df = self.dataframe[self.dataframe['types'].apply(lambda types: statement_type in types)]
        if translated:
            filtered_df = filtered_df[filtered_df['translated'] == 'translated']
        else:
            filtered_df = filtered_df[filtered_df['translated'] == 'source']
        return filtered_df

    def id2translation_status(self, single_code):
        """ Map a single code to a translation status. """
        if int(single_code[6]) == 1:
            return "translated"
        else:
            return "source"

    def get_english_statements(self):
        """ Return only statements in English. """
        return self.dataframe[self.dataframe['translated'] == 'translated']

    def get_source_lang_statements(self):
        """ Return only statements in the source language. """
        return self.dataframe[self.dataframe['translated'] == 'source']

    @staticmethod
    def id_to_type(single_code):
        """ Map a single code to statement types. """
        types = set()

        # Check for original statement ( = not a paraphrase but the original statement in whatever language)
        if int(single_code[0]) == 1 and int(single_code[2]) == 0:
            types.add("original")

        # Check for non-adversarial (neither negated nor opposite, but can be paraphrase)
        if int(single_code[3]) == 0 and int(single_code[4]) == 0:
            types.add("non-adversarial")

        # Check for adversarial (negated or opposite), can be paraphrase of an opposite statement
        if int(single_code[3]) == 1 or int(single_code[4]) == 1:
            types.add("adversarial")

        # Check for negation, can be negation of original or any of the paraphrases of the original
        if int(single_code[4]) == 1:
            types.add("negation")

        # Check for opposite. can be opposite of original, or an automatically created
        # paraphrase of manually created opposite
        if int(single_code[3]) == 1:
            types.add("opposite")

        # Check for paraphrase (non-adversarial and adversarial)
        if int(single_code[2]) > 0:
            types.add("paraphrase")

        # Check for paraphrase (non-adversarial) (that is, paraphrase of original)
        if int(single_code[0]) == 1 and int(single_code[2]) > 0:
            types.add("paraphrase-non-adversarial")

        # Check for paraphrase (adversarial)
        if int(single_code[0]) == 0 and int(single_code[2]) > 0:
            types.add("paraphrase-adversarial")

        # Check for country-specific
        if int(single_code[5]) == 1:
            types.add("country-agnostic")

        return types
