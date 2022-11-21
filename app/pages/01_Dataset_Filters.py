import streamlit as st
import pandas as pd
import sys
from similar_words import similar_words as sw
from article_similarity.search_methods import (
    load_similarity_search_methods,
    SIM_SEARCH_METHODS,
)
from preprocess_utils import preprocess_utils as preprocess

# TODO: Check if this is the best practice for appending a different folder
sys.path.append("./scripts")

##### main page #####

st.set_page_config(page_title="Dataset Filters", layout="wide")

st.title("Dataset Filters")
st.subheader("Dataset")
daily_news = st.file_uploader("Upload news dataset here:", type=["csv", "xlsx"])

tokenized_df = None

# read uploaded text/csv file and preprocess "Headline" and "Summary" text
if daily_news is not None:
    # save file name for future output name
    file_name = daily_news.name.replace(".xlsx", "")
    file_name = file_name.replace(".csv", "")
    st.session_state["file_name"] = file_name

    if daily_news.type == "text/csv":
        file_details = {
            "filename": daily_news.name,
            "filetype": daily_news.type,
            "filesize": daily_news.size,
        }
        df = pd.read_csv(daily_news, encoding="latin1")
        st.write("Uploaded dataset: ", file_details["filename"])
        st.write("Number of articles: ", str(df.shape[0]))
    elif (
        daily_news.type
        == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    ):
        file_details = {
            "filename": daily_news.name,
            "filetype": daily_news.type,
            "filesize": daily_news.size,
        }
        df = pd.read_excel(daily_news)
        st.write("Uploaded dataset: ", file_details["filename"])
        st.write("Number of articles: ", str(df.shape[0]))

    tokenized_df = preprocess.tokenised_preprocessing(df, "Summary")
    tokenized_df = preprocess.tokenised_preprocessing(df, "Headline")
    tokenized_df["clean_Summary"] = tokenized_df.apply(
        lambda x: x["clean_Summary"]
        + x["Summary"].split(" ")
        + preprocess.remove_punctuation_text(x["Summary"]).split(" "),
        axis=1,
    )
    tokenized_df["clean_Headline"] = tokenized_df.apply(
        lambda x: x["clean_Headline"]
        + x["Headline"].split(" ")
        + preprocess.remove_punctuation_text(x["Headline"]).split(" "),
        axis=1,
    )
    tokenized_df = tokenized_df[
        [
            "Published",
            "Headline",
            "Summary",
            "Link",
            "Domain",
            "Facebook Interactions",
            "clean_Summary",
            "clean_Headline",
        ]
    ]
    tokenized_df = tokenized_df.astype(
        {
            "Published": "datetime64",
            "Headline": "string",
            "Summary": "string",
            "Link": "string",
            "Domain": "string",
            "Facebook Interactions": "int",
        }
    )
    tokenized_df["id"] = tokenized_df.index
    st.dataframe(
        tokenized_df[
            [
                "Published",
                "Headline",
                "Summary",
                "Link",
                "Domain",
                "Facebook Interactions",
            ]
        ]
    )
    st.session_state["initial_dataframe"] = tokenized_df
    st.subheader("Filtered Dataset")

    ##### sidebar #####
if tokenized_df is not None:
    with st.sidebar:

        with st.form(key="filter_params"):
            st.markdown("# Filters")

            # numerical entry filter for minimum facebook engagement
            min_engagement = int(
                st.number_input(
                    label="Minimum no. of Facebook Interactions:",
                    min_value=0,
                    max_value=max(tokenized_df["Facebook Interactions"]),
                    value=0,
                )
            )
            # filter for date range of articles
            date_range = st.date_input(
                "Date range of articles",
                value=(
                    tokenized_df["Published"].min(),
                    tokenized_df["Published"].max(),
                ),
                min_value=min(tokenized_df["Published"]),
                max_value=max(tokenized_df["Published"]),
            )

            # selection-based filter for article domains to be removed
            domain_filter = st.multiselect(
                label="Article domains to exclude",
                options=tokenized_df["Domain"].unique(),
                default=None,
            )

            # keyword-based filter based on tokens separated by comma and space. If ALL input tokens present in headline/summary of article, then include article
            all_kw_filter = str(
                st.text_input(
                    label="Keep articles containing ALL keywords (separate by comma and space ', '): ",
                    value="",
                )
            )

            # keyword-based filter based on tokens separated by comma and space. If ANY of the input tokens present in headline/summary of article, then INCLUDE article
            any_kw_filter = str(
                st.text_input(
                    label="Keep articles containing ANY keywords (separate by comma and space ', '): ",
                    value="",
                )
            )

            # keyword-based filter based on tokens separated by comma and space. If ANY of the input tokens OR similar words to input tokens present in headline/summary of article, then INCLUDE article
            # similar articles are identified based on open-source API in https://relatedwords.org/
            similar_option = st.selectbox(
                'Consider similar keywords to input keywords for "ANY" option?',
                ("Yes", "No"),
            )

            # keyword-based filter based on tokens separated by comma and space. If ANY of the input tokens present in headline/summary of article, then EXCLUDE article
            kw_filter_remove = str(
                st.text_input(
                    label="Remove articles containing ANY keywords (separate by comma and space ', '): ",
                    value="",
                )
            )

            sim_search_method = st.selectbox(
                "Similarity Metric",
                SIM_SEARCH_METHODS,
            )
            sim_search_k = int(
                st.number_input(
                    label="Top n articles",
                    min_value=1,
                    value=len(tokenized_df),
                    step=1,
                )
            )
            sim_search_text = st.text_area(
                "News text",
            )

            submit_button = st.form_submit_button(label="Submit")
            # filters dataset according to filters set in sidebar
            if submit_button:
                df_filtered = tokenized_df[
                    lambda df: df["Facebook Interactions"] >= min_engagement
                ]
                df_filtered = df_filtered[
                    lambda df: df["Published"].dt.date.between(*date_range)
                ]
                df_filtered = df_filtered[lambda df: ~df["Domain"].isin(domain_filter)]
                # text-based filters are applied to both summary and headline
                if len(kw_filter_remove) > 0:
                    kw_filter_remove = kw_filter_remove.split(", ")
                    df_filtered = df_filtered[
                        lambda df: (
                            (
                                df["clean_Summary"].apply(
                                    lambda x: bool(set(x) & set(kw_filter_remove))
                                )
                                == False
                            )
                            & (
                                df["clean_Headline"].apply(
                                    lambda x: bool(set(x) & set(kw_filter_remove))
                                )
                                == False
                            )
                        )
                    ]
                if len(all_kw_filter) > 0:
                    all_kw_filter = all_kw_filter.split(", ")
                    df_filtered = df_filtered[
                        lambda df: (
                            df["clean_Summary"].apply(
                                lambda x: bool(set(all_kw_filter).issubset(set(x)))
                            )
                            | df["clean_Headline"].apply(
                                lambda x: bool(set(all_kw_filter).issubset(set(x)))
                            )
                        )
                    ]
                if len(any_kw_filter) > 0:
                    if similar_option == "Yes":
                        any_kw_filter = sw.scraping_similar_phrases(any_kw_filter, 5)
                    else:
                        any_kw_filter = any_kw_filter.split(", ")
                    df_filtered = df_filtered[
                        lambda df: (
                            df["clean_Summary"].apply(
                                lambda x: bool(set(x) & set(any_kw_filter))
                            )
                            | df["clean_Headline"].apply(
                                lambda x: bool(set(x) & set(any_kw_filter))
                            )
                        )
                    ]
                if len(df_filtered) > 0:
                    searcher = load_similarity_search_methods(
                        sim_search_method,
                        "multi-qa-MiniLM-L6-cos-v1",
                        df_filtered.copy(),
                    )
                    distances, indexes = searcher(
                        sim_search_text,
                        k=sim_search_k,
                    )
                    df_filtered = (
                        df_filtered.iloc[indexes]  # .reset_index(drop=True)
                        # .assign(similarity_score=distances)
                    )

                df_filtered["filtered_id"] = range(len(df_filtered))
                st.session_state["df_filtered"] = df_filtered

    if "df_filtered" in st.session_state:
        # Preview filtered dataframe and allow for progression to next page for Topic Discovery
        st.write(f"Similar keywords considered: {any_kw_filter}")
        st.write("Number of articles: ", str(st.session_state["df_filtered"].shape[0]))
        st.dataframe(
            st.session_state["df_filtered"][
                [
                    "Published",
                    "Headline",
                    "Summary",
                    "Link",
                    "Domain",
                    "Facebook Interactions",
                ]
            ]
        )
        final_button = st.button(
            "Discover topics in this set of articles?", key=None, help=None
        )

        if final_button:
            st.write("Proceed with Topic Discovery on this subset of articles!")
