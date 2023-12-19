import datetime
import os

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from st_pages import add_page_title
from streamlit_extras.no_default_selectbox import selectbox
from utils import core
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
GSHEET_TAXONOMY_ID = os.environ.get("GSHEET_TAXONOMY_ID")
gsheet_taxonomy_url = "https://docs.google.com/spreadsheets/d/" + GSHEET_TAXONOMY_ID

add_page_title(layout="wide")

embedding_model = st.cache_resource(core.load_embedding_model)()
classification_model = st.cache_resource(core.load_classification_model)()
file_handler = st.cache_resource(core.FileHandler)(core.DATA_DIR)


def filter_featurize(columns, domain_filter, min_engagement, date_range):
    if file_handler.list_csv_files_df().empty:
        # TODO: Raise error, decouple streamlit
        st.error(
            "There is no data. Please return to the Home page and upload a csv file."
        )
        return

    results_filtered_df = file_handler.filtered_query(
        columns, domain_filter, min_engagement, date_range
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
    # processed_table.insert(0, "Select", False)

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
    taxonomy_df = st.cache_data(core.fetch_latest_taxonomy)()
    filter_bounds = st.cache_data(file_handler.get_filter_bounds)()

    # st.title("🔎 Data Selection")
    st.markdown("""---""")
    st.markdown(
        """
        Please select the data to work on from the database. You may perform the relevant filtering of the data using the filters side bar on the left.
        """
    )
    st.markdown("""---""")

    # Metrics placeholder
    col_nrows, col_fb_interactions, _ = st.columns([1, 1, 3])
    with col_nrows:
        nrows_metric = st.empty()
    with col_fb_interactions:
        fb_interactions_metric = st.empty()

    # filter_bounds["theme"].insert(0, "All, excluding unthemed")

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

        # theme_filter = selectbox(
        #     label="Labelled articles to include",
        #     options=filter_bounds["theme"],
        #     no_selection_label="All, including unlabelled",
        # )

    # filters dataset according to filters set in sidebar
    try:
        columns = (
            "published",
            "headline",
            "summary",
            "link",
            "facebook_page_name",
            "domain",
            "facebook_interactions",
            "themes",
            "indexes",
            "subindex",
            "source",
        )

        # Classify & embed for subsequent steps
        processed_table = filter_featurize(
            columns, domain_filter, min_engagement, datetime_bounds
        )

        # Update metrics on filtered data
        nrows_metric.metric(label="Total Articles", value=processed_table.shape[0])

        fb_interactions_metric.metric(
            label="Total Facebook Interactions",
            value=f"{processed_table['facebook_interactions'].sum():,}",
        )

        column_config = {
            "link": st.column_config.LinkColumn("link", width="small"),
            "facebook_interactions": st.column_config.NumberColumn(
                "Facebook Interactions", width="small"
            ),
            "suggested_labels": st.column_config.ListColumn(
                "Suggested Theme", width="medium"
            ),
            "suggested_labels_score_trunc": st.column_config.LineChartColumn(
                "Suggested Theme Score",
                width="small",
                y_min=0,
                y_max=1,
            ),
            "themes": st.column_config.ListColumn("themes", width="small"),
            "indexes": st.column_config.ListColumn("indexes", width="small"),
        }

        view_df = st.data_editor(
            data=(processed_table),
            column_config=column_config,
            # hide_index=True,
            column_order=[
                "published",
                "headline",
                "subindex",
                "suggested_labels",
                "link",
                "domain",
                "facebook_interactions",
                "themes",
                "indexes",
            ],
            disabled=[
                "published",
                "headline",
                "summary",
                "link",
                "domain",
                "facebook_page_name",
                "facebook_interactions",
            ],
            height=600,
        )

        _, col_save = st.columns([5, 1])
        with col_save:
            save_indexing_btn = st.button("Save Indexing")

        subindex_groups = view_df.groupby("subindex")
        # st.write(subindex_groups)
        out_groups = {}
        for subindex, g_df in subindex_groups:
            if len(subindex) == 0:
                continue
            df_col, theme_col, index_col = st.columns([4, 1, 1])
            with theme_col:
                existing_themes = list(set(g_df["themes"].explode().dropna()))
                theme = st.multiselect(
                    f'Themes for "{subindex}"',
                    taxonomy_df["Theme"].unique(),
                    default=existing_themes,
                )
                # g["themes"] = theme
            with index_col:
                existing_indexes = list(set(g_df["indexes"].explode().dropna()))
                index = st.multiselect(
                    f'Indexes for "{subindex}"',
                    taxonomy_df["Index"].unique(),
                    default=existing_indexes,
                )
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

        # st.write(out_groups["cck car ramming"])

        if save_indexing_btn:
            file_handler.update_subindex(view_df)
            for subindex, group_df in out_groups.items():
                file_handler.update_themes(group_df)
    except ValueError as ve:
        st.error(
            f"The filters selected produces an empty dataframe. Please re-adjust filters to view data. \n\n {ve}"
        )


def slice_table(df):
    """Function to slice dataframe

    Args:
        df (pandas.core.frame.DataFrame): Dataframe to be sliced

    Returns:
        dict: Sliced dataframe
    """

    df_theme_grouped = df.copy().groupby("theme_ref")

    top_themes = (
        df_theme_grouped["facebook_interactions"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )

    # top_themes = get_top_themes(df)
    df_collection = {}
    for theme in top_themes:
        # # Sort by similarity to first, starting with most FB interactions
        # df_theme = df_theme_grouped.get_group(theme)

        # top_vector = np.array(
        #     df_theme.nlargest(1, "facebook_interactions")
        #     .reset_index(drop=True)
        #     .at[0, "vector"]
        # ).reshape(1, -1)

        # df_theme["similarity"] = cosine_similarity(
        #     np.array(df_theme["vector"].to_list()),
        #     top_vector
        # )

        # df_collection[theme] = df_theme.sort_values(
        #     by=["similarity"], ascending=False
        # )

        # Sort by recursive similiarity to first, starting with most FB interactions
        df_theme = df_theme_grouped.get_group(theme)
        df_theme["similarity_rank"] = pd.Series(dtype="int")
        ref_vector = np.array(
            df_theme.nlargest(1, "facebook_interactions")
            .reset_index(drop=True)
            .at[0, "vector"]
        ).reshape(1, -1)
        for i in range(len(df_theme)):
            temp_df = df_theme[df_theme["similarity_rank"].isna()]
            temp_df["similarity"] = cosine_similarity(
                np.array(temp_df["vector"].to_list()), ref_vector
            )
            ref_vector = np.array(
                temp_df.nlargest(2, "similarity")
                .nsmallest(1, "similarity")
                .reset_index(drop=True)
                .at[0, "vector"]
            ).reshape(1, -1)

            df_theme.at[temp_df["similarity"].idxmax(), "similarity_rank"] = i

        df_collection[theme] = df_theme.sort_values(
            by=["similarity_rank"], ascending=True
        )

        # # Sort by fB interactions
        # df_collection[theme] = df_theme_grouped.get_group(theme).sort_values(
        #     by=["facebook_interactions"], ascending=False
        # )
    return df_collection


if __name__ == "__main__":
    data_selection()
