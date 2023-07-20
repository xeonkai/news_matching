import streamlit as st
from functions.tabular_indexing import (
    display_aggrid_by_theme,
    display_stats,
    process_table,
    slice_table,
)
from utils import core

st.set_page_config(
    page_title="Article Indexing",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("📝 Article Indexing")
st.markdown("""---""")
st.markdown(
    """
    In this page, you can view the themes and indexes predicted for each news article. 
    If the predicted theme and index are not correct, select "-Enter New Label" in the drop-down menu, then select the correct theme and index from the respective dropdown menus.
    The subindex column is for the user to enter the subindex of the article. If the article does not have a subindex, leave the column blank.
    You may click the headline to open the article in a new tab.
    Select the rows that you have verified and/or edited using the checkbox on the left, then click on the "Confirm" button to save the changes.
    If you want to select multiple checkboxes at once, hold down the "Shift" key while selecting the checkboxes.
    """,
)
st.markdown("""---""")


def run():
    if "subset_df_with_preds" not in st.session_state:
        st.warning(
            "No data selected yet! Please select the required data from the Data Explorer page!",
            icon="⚠️",
        )
        return

    if "taxonomy" not in st.session_state:
        taxonomy_chains_df = core.fetch_taxonomy()
        taxonomy = taxonomy_chains_df.groupby("Theme")["Index"].apply(list).to_dict()
        st.session_state["taxonomy"] = taxonomy

    uploaded_data_with_indexes = st.session_state["subset_df_with_preds"]

    uploaded_data_with_indexes = process_table(
        uploaded_data_with_indexes, st.session_state["taxonomy"]
    )
    table_collection = slice_table(uploaded_data_with_indexes)

    display_stats(uploaded_data_with_indexes)

    if "current_theme_index" not in st.session_state:
        st.session_state["current_theme_index"] = 0
    current_index_index = st.session_state["current_theme_index"]

    st.markdown("""---""")

    display_aggrid_by_theme(table_collection, current_index_index)

    st.markdown("""---""")


if __name__ == "__main__":
    run()
