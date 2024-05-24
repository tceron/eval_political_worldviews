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
* **Language**: e.g. en, es, de...
* **CountryCode**: e.g. ch, de, fr, it


## Chapel Hill Survey

The data from the Chapel Hill Expert Survey can be found [here](https://www.chesdata.eu/ches-europe).


# Code 

## Run models to generate answers given prompts

The code to run the models can be found in the `run_model` folder. The code is based on the Hugging Face transformers library.

[...]

## Evaluation

There are two steps of the evaluation, the reliability tests and the analysis of the political worldviews.

### Reliability tests

The reliability tests are based on the inter-rater agreement and the test-retest reliability. The code can be found in the `reliability` folder.

[...]

initial commit
### Analysis of the political worldviews

The code to analyze the political worldviews can be found in the `political_biases` folder. 

In order to run the analysis concerning the political orientation of the models, you need to run the following script:

    python3 leaning_analysis.py --passed_test hard_pass -- condition 0

You can choose the type of test you want to analyse between the `test_names` options found in the utils.py file. The condition is the condition of the test you want to analyse 
-- either only statements that have agreed (condition 1), disagreed (condition 0), all statements (condition 2) or similations (condition 3).

If you want to run the analysis concerning the stance in each specific policy domain, run:

    python3 domain_specific.py --passed_test hard_pass

The arguments are the same as above. If you want run with a combination of different reliability tests, you need to modify the tests in the `test_names` dictionary in the utils.py file. 

Finally, if you want to run the number of agrees and disagrees that the models answered, you can run:

    python3 plot_agree_disagree.py --passed_test hard_pass




