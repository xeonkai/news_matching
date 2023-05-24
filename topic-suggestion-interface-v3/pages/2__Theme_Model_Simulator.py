import streamlit as st
from functions.random_label_generator import (
    assign_labels_to_dataframe,
    assign_theme_chain_to_dataframe,
)
from functions.taxonomy_reader import (
    read_taxonomy,
    generate_label_chains,
    reformat_taxonomy,
)
import utils.design_format as format
import utils.utils as utils

st.set_page_config(page_title="Theme Model Simulator", page_icon="📰", layout="wide")

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
    uploaded_data = utils.get_cached_object("csv_file")

    if utils.check_session_state_key("K"):
        k = utils.get_cached_object("K")
    utils.cache_object(k, "K") 

    uploaded_data_with_themes = assign_theme_chain_to_dataframe(
        uploaded_data.copy(), taxonomy_chains, k
    )

    # sort by facebook interactions
    uploaded_data_with_themes = uploaded_data_with_themes.sort_values(
        by=["Facebook Interactions"], ascending=False
    ).reset_index(drop=True)

    utils.cache_object(uploaded_data_with_themes, "csv_file_with_predicted_labels")
    utils.customDisppearingMsg("Running Theme Model Simulator", wait=3, type_="info")


def run():
    if utils.check_session_state_key("csv_file"):
        taxonomy = reformat_taxonomy(read_taxonomy())
        # st.write(taxonomy)
        taxonomy_chains = generate_label_chains(taxonomy)
        utils.cache_object(taxonomy, "taxonomy")
        st.subheader("Previewing the Taxonomy")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.write(taxonomy)
        with c2:
            st.dataframe(taxonomy_chains)

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
