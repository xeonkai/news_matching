import datetime
import os

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from st_pages import add_page_title
from sklearn.metrics.pairwise import cosine_similarity
from utils import core

pd.options.mode.chained_assignment = None

load_dotenv()
GSHEET_TAXONOMY_ID = os.environ.get("GSHEET_TAXONOMY_ID")
gsheet_taxonomy_url = "https://docs.google.com/spreadsheets/d/" + GSHEET_TAXONOMY_ID

add_page_title(layout="wide")

embedding_model = st.cache_resource(core.load_embedding_model)()
classification_model = st.cache_resource(core.load_classification_model)()
file_handler = st.cache_resource(core.FileHandler)(core.DATA_DIR)
taxonomy_themes_series = st.cache_data(core.fetch_latest_themes, ttl=60 * 5)()
taxonomy_indexes_series = st.cache_data(core.fetch_latest_index, ttl=60 * 5)()


@st.cache_data
def filter_process_data(domain_filter, min_engagement, date_range):
    results_filtered_df = file_handler.filtered_query(
        domain_filter, min_engagement, date_range
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

    # Streamlit editable df requires list type to be not na
    processed_table["themes"] = processed_table["themes"].fillna("").apply(list)
    processed_table["indexes"] = processed_table["indexes"].fillna("").apply(list)

    # Inefficient recursive similarity, seeded by top fb interactions
    # Doesn't matter cause only computed on first load / save
    processed_table["similarity_rank"] = pd.Series(dtype="int")
    ref_vector = np.array(
        processed_table.nlargest(1, "facebook_interactions")
        .reset_index(drop=True)
        .at[0, "vector"]
    ).reshape(1, -1)
    for i in range(len(processed_table)):
        temp_df = processed_table[processed_table["similarity_rank"].isna()]
        temp_df["similarity"] = cosine_similarity(
            np.array(temp_df["vector"].to_list()), ref_vector
        )
        ref_vector = np.array(
            temp_df.nlargest(2, "similarity")
            .nsmallest(1, "similarity")
            .reset_index(drop=True)
            .at[0, "vector"]
        ).reshape(1, -1)

        processed_table.at[temp_df["similarity"].idxmax(), "similarity_rank"] = i

    processed_table = (
        processed_table.sort_values(by=["similarity_rank"], ascending=True)
        .reset_index(drop=True)
        .drop(columns=["similarity_rank"])
    )

    return processed_table


def data_selection():
    # st.title("🔎 Data Selection")
    st.markdown("""---""")
    st.markdown(
        """
        Please select the data to work on from the database. You may perform the relevant filtering of the data using the filters side bar on the left.
        """
    )
    st.markdown("""---""")

    if file_handler.list_csv_files_df().empty:
        st.error(
            "There is no data. Please return to the Home page and upload a csv file."
        )
        return

    filter_bounds = st.cache_data(file_handler.get_filter_bounds, ttl=15)()

    # Filter inputs by user
    with st.sidebar:
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
        col_start_time, col_end_time = st.columns([1, 1])
        with col_start_time:
            start_time = st.time_input(
                "Start time on first selected day", value=datetime.time(0, 0)
            )
        with col_end_time:
            end_time = st.time_input(
                "End time on last selected day", value=datetime.time(23, 59)
            )

        # combine date and time
        start_date = datetime.datetime.combine(date_range[0], start_time)
        end_date = datetime.datetime.combine(date_range[1], end_time)
        datetime_bounds = (start_date, end_date)

        # selection-based filter for article domains to be removed
        domain_filter = st.multiselect(
            label="Article domains to exclude",
            options=filter_bounds["domain_list"],
            default=[],
        )

    # filters dataset according to filters set in sidebar
    try:
        # Classify & embed for subsequent steps
        processed_table = filter_process_data(
            domain_filter, min_engagement, datetime_bounds
        )

        # Metrics placeholder
        col_nrows, col_fb_interactions, col_taxonomy, _ = st.columns([1, 1, 1, 2])
        # Update metrics on filtered data
        with col_nrows:
            st.metric(label="Total Articles", value=processed_table.shape[0])
        with col_fb_interactions:
            st.metric(
                label="Total Facebook Interactions",
                value=f"{processed_table['facebook_interactions'].sum():,}",
            )
        with col_taxonomy:
            st.markdown(f"[Link to Taxonomy :scroll:]({core.gsheet_taxonomy_url})")
        
        st.caption(
            '<div style="text-align: right;">Download data by hovering here </div>',
            unsafe_allow_html=True,
        )

        column_config = {
            "link": st.column_config.LinkColumn("link", width="small"),
            "facebook_link": st.column_config.LinkColumn(
                "facebook_link", width="medium"
            ),
            "themes": st.column_config.ListColumn("themes", width="small"),
            "indexes": st.column_config.ListColumn("indexes", width="small"),
        }

        view_df = st.data_editor(
            data=processed_table,
            column_config=column_config,
            column_order=[
                "published",
                "headline",
                "subindex",
                "facebook_interactions",
                "link",
                "facebook_link",
                "domain",
                "themes",
                "indexes",
            ],
            disabled=[
                "published",
                "headline",
                "summary",
                "link",
                "facebook_link",
                "domain",
                "facebook_page_name",
                "facebook_interactions",
            ],
            height=600,
        )
        _, col_save = st.columns([5, 1])
        with col_save:
            save_indexing_btn = st.button("Save Indexing & Refresh")

        subindex_groups = view_df.groupby("subindex")

        out_groups = {}
        for subindex, g_df in subindex_groups:
            if len(subindex) == 0:
                continue
            df_col, theme_col, index_col = st.columns([2, 1, 1])
            with theme_col:
                existing_themes = (
                    g_df["themes"]
                    .explode()
                    .drop_duplicates()
                    .dropna()[lambda s: s.isin(taxonomy_themes_series)]
                )
                theme = st.multiselect(
                    f'Themes for "{subindex}"',
                    taxonomy_themes_series,
                    default=existing_themes,
                )
            with index_col:
                existing_indexes = (
                    g_df["indexes"]
                    .explode()
                    .drop_duplicates()
                    .dropna()[lambda s: s.isin(taxonomy_indexes_series)]
                )
                index = st.multiselect(
                    f'Indexes for "{subindex}"',
                    taxonomy_indexes_series,
                    default=existing_indexes,
                )
            # Reverse order to reflect changes in df
            with df_col:
                g_df["themes"] = g_df.apply(lambda _: theme, axis=1)
                g_df["indexes"] = g_df.apply(lambda _: index, axis=1)
                g_df.insert(0, "Select", True)
                st.write(
                    g_df[
                        [
                            # "Select",
                            "headline",
                            "subindex",
                            "themes",
                            "indexes",
                        ]
                    ]
                )
                out_groups[subindex] = g_df

        _, col_save_extra = st.columns([5, 1])
        with col_save_extra:
            extra_save_indexing_btn = st.button("Save Indexing & Refresh", key="extra_save")

        if save_indexing_btn or extra_save_indexing_btn:
            file_handler.update_subindex(view_df)
            for subindex, group_df in out_groups.items():
                file_handler.update_labels(group_df)
            # filter_process_data.clear()
            st.cache_data.clear()
            st.rerun()

    except ValueError as ve:
        st.error(
            f"The filters selected produces an empty dataframe. Please re-adjust filters to view data. \n\n {ve}"
        )


if __name__ == "__main__":
    data_selection()
