import streamlit as st
import utils.design_format as format
import utils.utils as utils
from functions.tabular_indexing import (
    process_table,
    slice_table,
    display_aggrid_by_theme,
    display_stats,
)

st.set_page_config(
    page_title="Article Indexing",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("📝 Article Indexing")
format.horizontal_line()
format.align_text(
    """
    In this page, you can view the themes and indexes predicted for each news article. 
    If the predicted theme and index are not correct, select "-Enter New Label" in the drop-down menu, then select the correct theme and index from the respective dropdown menus.
    The subindex column is for the user to enter the subindex of the article. If the article does not have a subindex, leave the column blank.
    Select the rows that you have verified and/or edited using the checkbox on the left, then click on the "Confirm" button to save the changes.
    """,
    "justify",
)

format.horizontal_line()


def run():
    if utils.check_session_state_key("csv_file_with_predicted_labels"):
        uploaded_data_with_indexes = utils.get_cached_object(
            "csv_file_with_predicted_labels"
        )

        uploaded_data_with_indexes = process_table(uploaded_data_with_indexes)
        table_collection = slice_table(uploaded_data_with_indexes)

        display_stats(uploaded_data_with_indexes)

        if utils.check_session_state_key("current_theme_index"):
            current_index_index = utils.get_cached_object(
                "current_theme_index")
        else:
            current_index_index = 0
            utils.cache_object(current_index_index, "current_theme_index")

        format.horizontal_line()

        display_aggrid_by_theme(table_collection, current_index_index)

        format.horizontal_line()
    else:
        utils.no_file_uploaded()


if __name__ == "__main__":
    run()
