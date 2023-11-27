import datetime

import streamlit as st
from streamlit_extras.no_default_selectbox import selectbox
from functions.tabular_indexing import (
    display_aggrid_by_theme,
    display_stats,
    process_table,
    slice_table,
)
from functions.grid_response_consolidator import consolidate_grid_responses
from utils import core
from st_pages import add_page_title

add_page_title(layout="wide")

# st.set_page_config(
#     page_title="Data Selection & Article Indexing", page_icon="📰", layout="wide"
# )


def data_selection():
    # st.title("🔎 Data Selection")
    st.markdown("""---""")
    st.markdown(
        """
        Please select the data to work on from the DataBase. You may perform the relevant filtering of the data using the filters side bar on the left.
        """
    )
    st.markdown("""---""")

    embedding_model = st.cache_resource(core.load_embedding_model)()
    classification_model = st.cache_resource(core.load_classification_model)()
    file_handler = core.FileHandler(core.DATA_DIR)

    if file_handler.list_csv_files_df().empty:
        st.error(
            "There is no data. Please return to the Home page and upload a csv file."
        )
        return
    # Metrics placeholder
    col1, col2 = st.columns([1, 1])
    with col1:
        nrows_metric = st.empty()

    filter_bounds = st.cache_data(file_handler.get_filter_bounds)()
    filter_bounds["labels"].insert(0, "All, excluding unlabelled")
    # max_fb_interactions, domain_list, min_date, max_date = get_filter_bounds()

    # Filter inputs by user
    with st.sidebar:
        with st.form(key="filter_params"):
            st.markdown("# Filters")

            # numerical entry filter for minimum facebook engagement
            min_engagement = int(
                st.number_input(
                    label="Minimum no. of Facebook Interactions:",
                    min_value=0,
                    max_value=filter_bounds["max_fb_interactions"],
                    value=100,
                )
            )
            # filter for date range of articles
            date_range = st.date_input(
                "Date range of articles",
                value=(
                    filter_bounds["max_date"] - datetime.timedelta(days=1),
                    filter_bounds["max_date"],
                ),
                min_value=filter_bounds["min_date"],
                max_value=filter_bounds["max_date"],
            )

            start_time = st.time_input(
                "Start time on first selected day", value=datetime.time(0, 0)
            )

            end_time = st.time_input(
                "End time on last selected day", value=datetime.time(23, 59)
            )

            # combine date and time
            start_date = datetime.datetime.combine(date_range[0], start_time)
            end_date = datetime.datetime.combine(date_range[1], end_time)
            date_range = (start_date, end_date)

            # selection-based filter for article domains to be removed
            domain_filter = st.multiselect(
                label="Article domains to exclude",
                options=filter_bounds["domain_list"],
                default=[],
            )

            label_filter = selectbox(
                label="Labelled articles to include",
                options=filter_bounds["labels"],
                no_selection_label="All, including unlabelled",
            )

            max_results = int(
                st.number_input(
                    label="Max results to show",
                    min_value=0,
                    max_value=10_000,
                    value=500,
                )
            )

            submit_button = st.form_submit_button(label="Submit")

    # filters dataset according to filters set in sidebar
    if submit_button:
        try:
            columns = (
                "published",
                "headline",
                "summary",
                "link",
                "facebook_page_name",
                "domain",
                "facebook_interactions",
                "label",
                "source",
            )
            results_filtered_df = file_handler.filtered_query(
                columns, domain_filter, min_engagement, date_range, label_filter
            )
            # Classify & embed for subsequent steps
            processed_table = results_filtered_df.pipe(
                core.label_df,
                model=classification_model,
                column="headline",
            ).pipe(
                core.embed_df,
                model=embedding_model,
                column="headline",
            )

            results_filtered_df.to_csv("predicted.csv", index=False)

            st.session_state["subset_df_with_preds"] = processed_table

            # Update metrics on filtered data
            nrows_metric.metric(
                label="Total number of Rows", value=results_filtered_df.shape[0]
            )

            st.write(results_filtered_df.iloc[:max_results])
            st.markdown("---")

            # Reset theme jumper
            if "current_theme_index" in st.session_state:
                del st.session_state["current_theme_index"]
        except ValueError as ve:
            st.error(
                "The filters selected produces an empty dataframe. Please re-adjust filters to view data."
            )


def article_indexing():
    st.title("📝 Article Indexing")
    st.markdown("""---""")
    st.markdown("## Instructions")
    st.markdown(
        "- In this page, you can view the themes and indexes predicted for each news article. "
    )
    st.markdown(
        """- If the predicted theme and index are not correct, select "-Enter New Label" in the drop-down menu, then select the correct theme and index from the respective dropdown menus."""
    )
    st.markdown(
        "- The subindex column is for the user to enter the subindex of the article. If the article does not have a subindex, leave the column blank."
    )
    st.markdown("- You may click the headline to open the article in a new tab.")
    st.markdown(
        """- Select the rows that you have verified and/or edited using the checkbox on the left, then click on the "Confirm" button to save the changes."""
    )
    st.markdown(
        """- If you want to select multiple checkboxes at once, hold down the "Shift" key while selecting the checkboxes."""
    )
    st.markdown("""---""")

    if "subset_df_with_preds" not in st.session_state:
        st.warning(
            "No data selected yet! Please select the required data by submitting relevant filters on the sidebar!",
            icon="⚠️",
        )
        return

    if "taxonomy" not in st.session_state:
        taxonomy_chains_df = core.fetch_latest_taxonomy()
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

    if "grid_responses" not in st.session_state:
        st.session_state["grid_responses"] = {}

    else:
        grid_responses = st.session_state["grid_responses"]
        consolidated_df = consolidate_grid_responses(grid_responses)
        # temp_df = consolidated_df.merge(
        #     uploaded_data_with_indexes[["link", "theme_ref", "index_ref"]],
        #     on="link",
        # )

        # percentage_correct = (
        #     (temp_df["theme"] == temp_df["theme_ref"])
        #     & (temp_df["index"] == temp_df["index_ref"])
        # ).mean()

        # st.write(f"{percentage_correct:.0%}")

        st.success(f"{consolidated_df.shape[0]} articles labelled.")

        st.markdown("""---""")


if __name__ == "__main__":
    data_selection()
    article_indexing()
