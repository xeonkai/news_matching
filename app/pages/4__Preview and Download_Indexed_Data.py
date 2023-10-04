import io

import numpy as np
import pandas as pd
import streamlit as st
from functions.grid_response_consolidator import (
    consolidate_grid_responses,
    extract_unlabelled_articles,
)
from utils import core

st.set_page_config(page_title="Download Indexed Data", page_icon="📰", layout="wide")

st.title("📊 Download Indexed Data")
st.markdown("""---""")
st.markdown(
    """
    In this page, you can download the indexed data that has been labelled with the guided indexing tool.
    """
)

st.markdown("""---""")


def update_labels(df):
    file_handler = core.FileHandler(core.DATA_DIR)
    labelled_df = (
        df[["link", "theme", "index"]]
        .copy()
        .assign(
            label=lambda df: df[["theme", "index"]]
            .agg(" > ".join, axis=1)
            .str.strip(" > ")
            .replace("", np.nan)
        )
        .dropna(how="all")
    )
    file_handler.update_labels(labelled_df)


def run():
    if "subset_df_with_preds" not in st.session_state:
        st.warning(
            "No data selected yet! Please select the required data from the Data Explorer page!",
            icon="⚠️",
        )
        return

    if "grid_responses" not in st.session_state:
        st.warning(
            "Please label the dataset first using the Guided Index Indexing page first!",
            icon="⚠️",
        )
        return

    st.subheader("Labelled Articles")
    csv_file = st.session_state["subset_df_with_preds"]
    grid_responses = st.session_state["grid_responses"]
    date = grid_responses['general']['data']['published'][0].date().strftime('%Y/%m/%d')

    selected_rows = [
        grid_response["selected_rows"] for grid_response in grid_responses.values()
    ]
    selected_rows = selected_rows[0] if len(selected_rows) else selected_rows

    if not len(selected_rows):
        st.warning(
            "No articles have been labelled yet. Please do so in the Tabular Index Indexing page!",
            icon="⚠️",
        )
        return
        # st.write(grid_responses['general']['data'])
    consolidated_df = consolidate_grid_responses(grid_responses)
    st.metric(label="Total Labelled Articles", value=consolidated_df.shape[0])

    st.dataframe(consolidated_df, use_container_width=True)

    unlabelled_articles = extract_unlabelled_articles(consolidated_df, csv_file)
    # st.write(unlabelled_articles)

    # pivot table of count of each theme and index
    # st.subheader("Count of each theme and index")
    theme_index_pivot = (
        consolidated_df.groupby(["theme", "index"])['facebook_interactions'].agg(['sum', 'count']).reset_index(names=["theme", "index"]).rename(columns={"sum": "sum_of_interactions"})
    )
    # st.dataframe(theme_index_pivot, use_container_width=True)

    # pivot table of count of each theme, index and subindex
    st.subheader("Count of each theme, index and subindex")
    theme_index_subindex_pivot = (
        consolidated_df.groupby(["theme", "index", "subindex"])['facebook_interactions'].agg(['sum', 'count']).reset_index(
            names=["theme", "index", "subindex"]).rename(columns={"sum": "sum_of_interactions"})
    )
    st.dataframe(theme_index_subindex_pivot, use_container_width=True)

    consolidated_df = pd.concat(
        [consolidated_df, unlabelled_articles], ignore_index=True
    ).reset_index(drop=True)

    buffer = io.BytesIO()

    # build excel workbook with 2 sheets - consolidated_df and df_pivot
    with pd.ExcelWriter(buffer) as writer:
        consolidated_df.to_excel(writer, sheet_name="Articles", index=False)
        theme_index_pivot.to_excel(writer, sheet_name="Theme Index Pivot", index=False)
        theme_index_subindex_pivot.to_excel(writer, sheet_name="Theme Index Subindex Pivot", index=False)
        # unlabelled_articles.to_excel(
        #     writer, sheet_name="Unlabelled Articles", index=False
        # )
    buffer.seek(0)

    # download button for excel file
    st.download_button(
        label="Save Results & Download Articles as Excel File",
        data=buffer,
        file_name=f"Labelled_Articles_{date}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        on_click=update_labels,
        args=(consolidated_df,),
    )


if __name__ == "__main__":
    run()
