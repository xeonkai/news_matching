import streamlit as st
import utils.design_format as format
import utils.utils as utils
from functions.tabular_theme_indexing import (
    process_table,
    slice_table,
    display_aggrid_by_theme,
    display_stats,
)

st.set_page_config(
    page_title="Tabular Theme Indexing",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("📝 Tabular Theme Indexing")
format.horizontal_line()
format.align_text(
    """
    In this page, you can verify if the predicted themes and indexes are correct. 
    If they are not correct, select "-Enter New.." in the respective drop-down menu, then key in the new Theme/Index/Subindex in the corresponding cells on the right.
    Select the rows that you have verified and/or edited using the checkbox on the left, then click on the "Confirm" button to save the changes.
    """,
    "justify",
)

format.horizontal_line()


def run():
    if utils.check_session_state_key("csv_file_filtered"):
        if utils.check_session_state_key("csv_file_with_predicted_labels"):
            uploaded_data_with_themes = utils.get_cached_object(
                "csv_file_with_predicted_labels"
            )

            # st.dataframe(uploaded_data_with_themes)

            table_collection = slice_table(
                process_table(uploaded_data_with_themes))

            display_stats(uploaded_data_with_themes)

            if utils.check_session_state_key("current_theme_index"):
                current_theme_index = utils.get_cached_object(
                    "current_theme_index")
            else:
                current_theme_index = 0
                utils.cache_object(current_theme_index, "current_theme_index")

            format.horizontal_line()

            display_aggrid_by_theme(table_collection, current_theme_index)

            format.horizontal_line()
        else:
            utils.customDisppearingMsg(
                "Please run the Theme Model Simulator first!",
                wait=-1,
                type_="warning",
                icon="⚠️",
            )

    else:
        utils.no_file_uploaded()


if __name__ == "__main__":
    run()
