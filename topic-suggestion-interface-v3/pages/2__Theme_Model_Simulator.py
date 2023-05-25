import streamlit as st
import pandas as pd
from functions.random_label_generator import (
    assign_labels_to_dataframe,
    assign_theme_chain_to_dataframe,
)
from functions.taxonomy_reader import (
    read_taxonomy,
    generate_label_chains,
    reformat_taxonomy,
    convert_chain_to_list,
)
from streamlit_extras.dataframe_explorer import dataframe_explorer
import utils.design_format as format
import utils.utils as utils

st.set_page_config(page_title="Theme Model Simulator",
                   page_icon="📰", layout="wide")

st.title("🔮 Theme Model Simulator")
format.horizontal_line()
format.align_text(
    """
    In this page, you can run the theme model simulator. For the sake of this demo, we will randomly assign each article to themes with their associated probabilties.
    """,
    "justify",
)

format.horizontal_line()


def run_theme_model_simulator(taxonomy_chains, k=5):
    uploaded_data = utils.get_cached_object("csv_file_filtered")

    if utils.check_session_state_key("K"):
        k = utils.get_cached_object("K")
    utils.cache_object(k, "K")

    uploaded_data_with_themes = assign_theme_chain_to_dataframe(
        uploaded_data.copy(), taxonomy_chains, k
    )

    # sort by facebook interactions
    uploaded_data_with_themes = uploaded_data_with_themes.sort_values(
        by=["facebook_interactions"], ascending=False
    ).reset_index(drop=True)

    utils.cache_object(uploaded_data_with_themes,
                       "csv_file_with_predicted_labels")
    utils.customDisppearingMsg(
        "Running Theme Model Simulator", wait=3, type_="info")


def run():

    if utils.check_session_state_key("csv_file_filtered"):
        taxonomy = read_taxonomy()
        taxonomy_chains = generate_label_chains(taxonomy)

        utils.cache_object(taxonomy, "taxonomy")
        st.subheader("Previewing the Taxonomy")

        taxonomy_chains_df = pd.DataFrame(pd.DataFrame(taxonomy_chains, columns=["Chain"]).apply(
            lambda x: convert_chain_to_list(x[0]), axis=1).to_list(), columns=["Theme", "Index", "Subindex"])
        taxonomy_chains_df_explorer = dataframe_explorer(taxonomy_chains_df)
        st.dataframe(taxonomy_chains_df_explorer, use_container_width=True)

        with st.form("Theme Model Simulator Form"):
            k = st.number_input(
                "Top-K predictions for each article",
                min_value=1,
                max_value=999,
                value=5,
            )
            if st.form_submit_button("Run Theme Model Simulator"):
                run_theme_model_simulator(taxonomy_chains, k)

        if utils.check_session_state_key("csv_file_with_predicted_labels"):
            format.horizontal_line()
            st.subheader("Dataframe with Themes")
            uploaded_data_with_themes = utils.get_cached_object(
                "csv_file_with_predicted_labels"
            )
            st.dataframe(uploaded_data_with_themes, use_container_width=True)
    else:
        utils.no_file_uploaded()


if __name__ == "__main__":
    run()
