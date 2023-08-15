import datetime

import streamlit as st
from streamlit_extras.no_default_selectbox import selectbox
from utils import core

st.set_page_config(page_title="Data Selection", page_icon="📰", layout="wide")

st.title("🔎 Data Selection")
st.markdown("""---""")
st.markdown(
    """
    In this page, you are able to select the data that you would like to work on from the DataBase. You may perform the relevant filtering of the data using the filtering side bar on the left.
    """
)
st.markdown("""---""")


def run():
    embedding_model = st.cache_resource(core.load_embedding_model)()
    classification_model = st.cache_resource(core.load_classification_model)()
    file_handler = core.FileHandler(core.DATA_DIR)

    if file_handler.list_csv_files_df().empty:
        st.error(
            "There is no data. Please return to the Home page and upload a csv file."
        )
    else:
        # Metrics placeholder
        col1, col2 = st.columns([1, 1])
        with col1:
            nrows_metric = st.empty()

        filter_bounds = file_handler.get_filter_bounds()
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

                st.write(
                    results_filtered_df.iloc[:max_results]
                    # .limit(
                    #     max_results
                    # )  # Limit rows in case too much data sent to browser
                    # .drop(
                    #     "vector", axis="columns"
                    # )  # Need "vector" embedding column for subsequent steps, but don't want to show
                )
            except ValueError as ve:
                st.error(
                    "The filters selected produces an empty dataframe. Please re-adjust filters to view data."
                )


if __name__ == "__main__":
    run()
