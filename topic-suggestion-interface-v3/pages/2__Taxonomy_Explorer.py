import streamlit as st
import pandas as pd
from functions.taxonomy_reader import (
    read_taxonomy,
    generate_label_chains,
    convert_chain_to_list,
)
from streamlit_extras.dataframe_explorer import dataframe_explorer
import utils.design_format as format
import utils.utils as utils

st.set_page_config(page_title="Taxonomy Explorer",
                   page_icon="📰", layout="wide")

st.title("🔮 Taxonomy Explorer")
format.horizontal_line()
format.align_text(
    """
    In this page, you can explore the taxonomy containing the Theme and Index from the range of data selected. You may filter the table by Theme and Index.
    """,
    "justify",
)

format.horizontal_line()


def run():

    if utils.check_session_state_key("csv_file_with_predicted_labels"):
        taxonomy = read_taxonomy()
        taxonomy_chains = generate_label_chains(taxonomy)

        utils.cache_object(taxonomy, "taxonomy")
        st.subheader("Previewing the Taxonomy")

        taxonomy_chains_df = pd.DataFrame(pd.DataFrame(taxonomy_chains, columns=["Chain"]).apply(
            lambda x: convert_chain_to_list(x[0]), axis=1).to_list(), columns=["Theme", "Index"])
        taxonomy_chains_df_explorer = dataframe_explorer(taxonomy_chains_df)
        st.dataframe(taxonomy_chains_df_explorer, use_container_width=True)

    else:
        utils.no_file_uploaded()


if __name__ == "__main__":
    run()
