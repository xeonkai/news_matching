import streamlit as st
import utils.design_format as format
import utils.utils as utils
import datetime
import duckdb

st.set_page_config(page_title="Data Selection", page_icon="📰", layout="wide")

st.title("🔎 Data Selection")
format.horizontal_line()
format.align_text(
    """
    In this page, you are able to select the data that you would like to work on from the DataBase. You may perform the relevant filtering of the data using the filtering side bar on the left.
    """,
    "justify",
)

format.horizontal_line()


def run():
    # Metrics placeholder

    col1, col2 = st.columns([1, 1])
    with col1:
        nrows_metric = st.empty()

    # Get filter bounds

    # Maximum interactions in db
    max_fb_interactions = duckdb.sql(
        f"SELECT MAX(facebook_interactions) FROM "
        f"read_parquet('{utils.PROCESSED_DATA_DIR.absolute()}/*.parquet')"
    ).fetchone()[0]

    # Domain list ordered by frequency
    domain_list = [
        domain
        for domain, count in duckdb.sql(
            "SELECT domain, COUNT(*) AS count FROM "
            f"read_parquet('{utils.PROCESSED_DATA_DIR.absolute()}/*.parquet') "
            "GROUP BY domain "
            "ORDER BY count DESC"
        ).fetchall()
    ]
    min_date, max_date = duckdb.sql(
        f"SELECT MIN(published), MAX(published) "
        f"FROM read_parquet('{utils.PROCESSED_DATA_DIR.absolute()}/*.parquet')"
    ).fetchone()

    # Filter inputs by user
    with st.sidebar:
        with st.form(key="filter_params"):
            st.markdown("# Filters")

            # numerical entry filter for minimum facebook engagement
            min_engagement = int(
                st.number_input(
                    label="Minimum no. of Facebook Interactions:",
                    min_value=0,
                    max_value=max_fb_interactions,
                    value=100,
                )
            )
            # filter for date range of articles
            date_range = st.date_input(
                "Date range of articles",
                value=(
                    max_date - datetime.timedelta(days=1),
                    max_date,
                ),
                min_value=min_date,
                max_value=max_date,
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
                options=domain_list,
                default=[],
            )

            max_results = int(
                st.number_input(
                    label="Number of results to show",
                    min_value=0,
                    max_value=1000,
                    value=50,
                )
            )

            submit_button = st.form_submit_button(label="Submit")

    # filters dataset according to filters set in sidebar
    if submit_button:
        columns = (
            "published",
            "headline",
            "summary",
            "link",
            "domain",
            "facebook_interactions",
            "vector",
            "suggested_labels",
            "suggested_labels_score",
        )

        query = (
            f"SELECT {','.join(columns)} "
            f"FROM read_parquet('{utils.PROCESSED_DATA_DIR}/*.parquet', filename=true) "
            f"WHERE domain NOT IN {tuple(domain_filter) if domain_filter else ('NULL',)} "
            f"AND facebook_interactions >= {min_engagement} "
            f"AND published BETWEEN '{date_range[0]}' AND '{date_range[1]}' "
            f"ORDER BY facebook_interactions DESC   "
            # f"{f'LIMIT {limit}' if limit else ''} "
        )
        results_filtered = duckdb.sql(query)

        # TODO: Move page to indexer page, materialize only when needed
        results_filtered_df = results_filtered.to_df()
        st.session_state["csv_file_with_predicted_labels"] = results_filtered_df

        # Update metrics on filtered data
        nrows_metric.metric(label="Number of Rows",
                            value=results_filtered_df.shape[0])

        st.write(
            results_filtered.limit(
                max_results
            )  # Limit rows in case too much data sent to browser
            .to_df()
            .drop(
                "vector", axis="columns"
            )  # Need "vector" embedding column for subsequent steps, but don't want to show
        )


if __name__ == "__main__":
    run()
