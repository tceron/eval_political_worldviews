import glob

import pandas as pd
from sympy.physics.control.control_plots import plt


def create_consistency_plot(source_dir, out_dir):
    """
    This function creates a plot that shows the consistency of the models over all tests.
    """
    template_model_based_questionnaires = glob.glob(f"{source_dir}/*.csv")
    # only keep files that end with 'statements_reliability_tests.csv'
    template_model_based_questionnaires = [f for f in template_model_based_questionnaires if
                                           f.endswith("statements_reliability_tests.csv")]
    model2results = []
    for filen_name in template_model_based_questionnaires:
        model = filen_name.split("chat")[0].split("/")[-1].replace("-", "")
        df = pd.read_csv(filen_name, sep=",")
        relative_passes_sing_test = len(df[df["test_sign"]]) / len(df)
        relative_passes_label_inversion = len(df[df["test_label_inversion"]]) / len(df)
        relative_passes_semantic_equivalence = len(df[df["test_semantic_equivalence"]]) / len(df)
        relative_passes_negation = len(df[df["test_negation"]]) / len(df)
        relative_passes_opposite = len(df[df["test_opposite"]]) / len(df)
        relative_passes = len(df[df["hard_pass"]]) / len(df)
        results_dir = {"sign": relative_passes_sing_test,
                       "label order": relative_passes_label_inversion,
                       "semantic equivalence": relative_passes_semantic_equivalence,
                       "negation": relative_passes_negation,
                       "opposite": relative_passes_opposite,
                       "all tests": relative_passes,
                       "model": model,
                       "template_id": df["template_id"].iloc[0]}
        model2results.append(results_dir)
    # convert to dataframe
    df = pd.DataFrame(model2results)
    # now reduce over template_id, merge all values from the columns that contain "relative_passes" and create a column "mean" and "std"
    df_mean = df.groupby(["model"]).agg({"sign": "mean",
                                         "label order": "mean",
                                         "semantic equivalence": "mean",
                                         "negation": "mean",
                                         "opposite": "mean",
                                         "all tests": "mean"}).reset_index()
    df_std = df.groupby(["model"]).agg({"sign": "std",
                                        "label order": "std",
                                        "semantic equivalence": "std",
                                        "negation": "std",
                                        "opposite": "std",
                                        "all tests": "std"}).reset_index()

    melted_df = df_mean.melt(id_vars=['model'], var_name='test', value_name='mean')
    melted_df["std"] = df_std.melt(id_vars=['model'], var_name='test', value_name='std')["std"]
    df = melted_df
    # Get a list of unique tests and models
    tests = df['test'].unique()
    models = df['model'].unique()
    # print(tabulate.tabulate(df, headers="keys", tablefmt="simple"))

    # Create a figure with subplots
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 9))  # Adjusted figsize for clarity

    # Flatten axes array for easy iteration (if len(tests) is 6)
    axes = axes.flatten()

    # If only one test is present, axes is not a list, so we convert it into a list for consistency.
    if len(tests) == 1:
        axes = [axes]

    # Loop through each test and create a plot
    models_renamed = {"mistral7b": "MISTRAL-7B", "flantfxxl": "FLAN-T5-XXL", "llama27b": "LLAMA2-7B",
                      "random": "RANDOM CHOICE", "llama213b": "LLAMA2-13B", "llama270b": "LLAMA2-70B"}
    # rename models
    df["model"] = df["model"].apply(lambda x: models_renamed[x])

    for ax, test in zip(axes, tests):
        # Filter the dataframe for the current test
        test_df = df[df['test'] == test]
        # sort the models such they are in the following order: human, random, bert, gpt2, dialoGPT
        models_sorted = ["MISTRAL-7B", "FLAN-T5-XXL", "LLAMA2-7B", "LLAMA2-13B", "LLAMA2-70B", "RANDOM CHOICE"]

        test_df["model"] = pd.Categorical(test_df["model"], categories=models_sorted, ordered=True)
        test_df = test_df.sort_values(by="model")
        # add a horizontal line at 0.5
        ax.axvline(x=0.5, color='grey', linestyle='--')
        # add horizontal bars at error bar ends

        ax.errorbar(test_df['mean'], test_df['model'], xerr=test_df['std'], fmt='o', color='green', ecolor='lightgray',
                    elinewidth=3, capsize=5)
        ax.set_title(f'{test} Consistency')
        ax.set_xlim(0, 1)
        ax.set_xlabel('Mean')
        ax.set_ylabel('Model')
        ax.invert_yaxis()  # Invert y-axis to match the provided plot
        ax.grid(True)
    # Adjust layout
    plt.tight_layout()
    # Save figure
    plt.savefig(f'{out_dir}/consistency_overall.jpeg', dpi=300)

