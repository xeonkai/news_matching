import numpy as np
import random
import pandas as pd
import streamlit as st

# Given a list of 5 topics, randomly assign a probability to each topic and return sorted list of topics.
def random_label_generator(label, seed=None):
    if seed:
        # Set np seed for reproducibility
        np.random.seed(seed)
    # Generate random probabilities for each topic
    probabilities = np.random.dirichlet(np.ones(len(label)), size=1)
    probabilities = probabilities.tolist()[0]
    # Sort topics by probability
    labels_with_probabilities = sorted(
        zip(label, probabilities), key=lambda x: x[1], reverse=True
    )
    return labels_with_probabilities


# Extract dictionary of top-k topics and their associated probabilities from a list of topics with probabilities
def extract_top_k_labels(labels_with_probabilities, k):
    top_k_labels = dict(labels_with_probabilities[:k])
    return top_k_labels


# Given a dataframe, assign a list of themes sorted by random probabilities to each row and return the dataframe with the theme column appended.
def assign_labels_to_dataframe(df, themes, indexes, subindexes, k, seed=None):
    if seed:
        # Set np seed for reproducibility
        np.random.seed(seed)

    # For each row, generate a distribution of themes
    df["Predicted_Themes"] = df.apply(
        lambda x: extract_top_k_labels(random_label_generator(themes, seed), k), axis=1
    )
    df["Predicted_Indexes"] = df.apply(
        lambda x: extract_top_k_labels(random_label_generator(indexes, seed), k), axis=1
    )
    df["Predicted_SubIndexes"] = df.apply(
        lambda x: extract_top_k_labels(random_label_generator(subindexes, seed), k),
        axis=1,
    )

    return df


# Given a dataframe, assign a list of predicted theme chains sorted by random probabilities to each row and return the dataframe with the theme column appended.
def assign_theme_chain_to_dataframe(df, theme_chains, k, seed=None):
    # st.dataframe(theme_chains)
    if seed:
        # Set np seed for reproducibility
        np.random.seed(seed)

    # For each row, generate a distribution of themes
    df["Predicted_Theme_Chains"] = df.apply(
        lambda x: extract_top_k_labels(random_label_generator(theme_chains, seed), k),
        axis=1,
    )

    return df


# Melt the dataframe to get a row for each topic and their associated probability
def melt_dataframe(df):
    # Melt the dataframe to get a row for each topic and their associated probability
    new_df = (
        pd.DataFrame(df["Predicted_Themes"].tolist(), index=df["Headline"])
        .reset_index()
        .melt("Headline", var_name="Topic", value_name="Probability")
        .dropna()
        .reset_index(drop=True)
        .sort_values(by=["Headline", "Probability"], ascending=[True, False])
    )

    return new_df


# Example:


if __name__ == "__main__":
    topics = ["Topic 1", "Topic 2", "Topic 3", "Topic 4", "Topic 5"]
    print(random_label_generator(topics))
