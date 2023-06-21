import streamlit as st
import utils.design_format as format
import utils.utils as utils
from functions.grid_response_consolidator import consolidate_grid_responses
import io
import pandas as pd

st.set_page_config(page_title="Download Indexed Data",
                   page_icon="📰", layout="wide")

st.title("📊 Download Indexed Data")
format.horizontal_line()
format.align_text(
    """
    In this page, you can download the indexed data that has been labelled with the guided indexing tool.
    """,
    "justify",
)

format.horizontal_line()


def run():
    if utils.check_session_state_key("csv_file_with_predicted_labels"):
        if utils.check_session_state_key("grid_responses"):
            st.subheader("Labelled Articles")
            grid_responses = utils.get_cached_object("grid_responses")
            csv_file = utils.get_cached_object(
                "csv_file_with_predicted_labels")

            selected_rows = [
                grid_response["selected_rows"]
                for grid_response in grid_responses.values()
            ]
            selected_rows = selected_rows[0] if len(
                selected_rows) else selected_rows

            if len(selected_rows):
                consolidated_df = consolidate_grid_responses(
                    grid_responses
                )
                st.metric(
                    label="Total Labelled Articles", value=consolidated_df.shape[0]
                )

                st.dataframe(consolidated_df, use_container_width=True)

                # pivot table of count of each theme and index
                st.subheader("Count of each theme and index")
                df_pivot = consolidated_df.groupby(
                    ["theme", "index"]).size().reset_index(name="count")
                st.dataframe(df_pivot, use_container_width=True)

                buffer = io.BytesIO()

                # build excel workbook with 2 sheets - consolidated_df and df_pivot
                with pd.ExcelWriter(buffer) as writer:
                    consolidated_df.to_excel(
                        writer, sheet_name="Labelled Articles", index=False
                    )
                    df_pivot.to_excel(
                        writer, sheet_name="Theme and Index Count", index=False
                    )
                buffer.seek(0)

                # download button for excel file
                st.download_button(
                    label="Download Labelled Articles as Excel File",
                    data=buffer,
                    file_name="Labelled_Articles.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

            else:
                utils.customDisppearingMsg(
                    "No articles have been labelled yet. Please do so in the Tabular Index Indexing page!",
                    wait=-1,
                    type_="warning",
                    icon="⚠️",
                )

        else:
            utils.customDisppearingMsg(
                "Please label the dataset first using the Guided Index Indexing page first!",
                wait=-1,
                type_="warning",
                icon="⚠️",
            )
    else:
        utils.no_file_uploaded()


if __name__ == "__main__":
    run()
