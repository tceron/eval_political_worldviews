# Robust evaluation of political worldview in LLMs

## Coding information for the IDs 

String with the following format:

    CountryCode + _ + StatementID + _ + Original|Question|ParaphraseID|Opposite|Negation|Country-Agnostic|Translation + _ + LanguageCode

e.g. _ch_15_0000001_fr_

* Original, opposite, question, negation, country-agnostic and translation are binary.
* **StatementID** is the ID of the statement in the original dataset.
* **ParaphraseID** = 0 means it's not a paraphrase. From 1 on it's the ID.
* **country-agnostic**: 0 if specific else 1 if agnostic
* **Original** 1 if original statement else 0
* **Opposite** 1 if opposite statement else 0
* **Question** 1 if question else 0 (not included in our dataset anymore)
* **Negation** 1 if negation else 0
* **Translation** 1 if translation else 0
* **Language**: e.g. en or es
* **CountryCode**: e.g. ch, de, fr, it


## Chapel Hill Survey

The data from the Chapel Hill Expert Survey can be found [here](https://www.chesdata.eu/ches-europe).
