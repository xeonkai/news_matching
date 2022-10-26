import streamlit as st
import pandas as pd
import sys

sys.path.append("./scripts")

from similar_words import similar_words as sw
from preprocess_utils import preprocess_utils as preprocess


st.set_page_config(page_title = "Dataset Filters",
                  layout="wide")

if "model" in st.session_state:
    del st.session_state['model']
if "df_after_form_completion" in st.session_state:
    del st.session_state['df_after_form_completion']

tokenized_df = None

##### main page #####

st.title("Dataset Filters")
daily_news = st.file_uploader("Upload news dataset here:", type=["csv", "xlsx"])

st.subheader("Dataset")

if 'df_remaining' in st.session_state:
    df = st.session_state["df_remaining"]
    tokenized_df = df[["Published", "Headline", "Summary", "Link", "Domain", "Facebook Interactions", "clean_Summary", "clean_Headline"]]
    tokenized_df["id"] = tokenized_df.index
    st.session_state['initial_dataframe'] = tokenized_df
    st.dataframe(tokenized_df)

elif daily_news is not None:
    if daily_news.type == "text/csv":
        file_details = {"filename": daily_news.name, "filetype": daily_news.type,
                                "filesize": daily_news.size}
        df = pd.read_csv(daily_news, encoding = "latin1")
        st.write("Uploaded dataset: ", file_details["filename"])
        st.write("Number of articles: ", str(df.shape[0]))
    elif daily_news.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        file_details = {"filename": daily_news.name, "filetype": daily_news.type,
                                "filesize": daily_news.size}
        df = pd.read_excel(daily_news)
        st.write("Uploaded dataset: ", file_details["filename"])
        st.write("Number of articles: ", str(df.shape[0]))
    tokenized_df = preprocess.tokenised_preprocessing(df, "Summary")
    tokenized_df = preprocess.tokenised_preprocessing(df, "Headline")
    tokenized_df["clean_Summary"] = tokenized_df.apply(lambda x: x["clean_Summary"] + x["Summary"].split(" ") + preprocess.remove_punctuation_text(x["Summary"]).split(" "), axis = 1)
    tokenized_df["clean_Headline"] = tokenized_df.apply(lambda x: x["clean_Headline"] + x["Headline"].split(" ") + preprocess.remove_punctuation_text(x["Headline"]).split(" "), axis = 1)
    tokenized_df = tokenized_df[["Published", "Headline", "Summary", "Link", "Domain", "Facebook Interactions", "clean_Summary", "clean_Headline"]]
    tokenized_df = tokenized_df.astype({'Published': 'datetime64', 'Headline': 'string', 
                                        'Summary': 'string', 'Link': 'string',
                                        'Domain': 'string', 'Facebook Interactions': 'int'})
    tokenized_df["id"] = tokenized_df.index
    st.dataframe(tokenized_df[["Published", "Headline", "Summary", "Link", "Domain", "Facebook Interactions"]])
    st.session_state['initial_dataframe'] = tokenized_df
    st.subheader("Filtered Dataset")

    ##### sidebar #####
if tokenized_df is not None:
    with st.sidebar:

        with st.form(key = "filter_params"):
            st.markdown("# Filters")

            min_engagement = int(
                st.number_input(
                    label = "Minimum no. of Facebook Interactions:", 
                    min_value = 0,
                    max_value = max(tokenized_df["Facebook Interactions"]),
                    value = 0
                )
            )

            date_range = st.date_input("Date range of articles", 
                            value = (tokenized_df["Published"].min(), tokenized_df["Published"].max()),
                            min_value=min(tokenized_df["Published"]),
                            max_value=max(tokenized_df["Published"]))


            domain_filter = st.multiselect(label = "Article domains to exclude", options = tokenized_df["Domain"].unique(), default = None)

            all_kw_filter = str(
                st.text_input(
                    label = "Keep articles containing ALL keywords (separate by comma and space ', '): ", 
                    value = ""
                )
            )

            any_kw_filter = str(
                st.text_input(
                    label = "Keep articles containing ANY keywords (separate by comma and space ', '): ", 
                    value = ""
                )
            )

            similar_option = st.selectbox('Consider similar keywords to input keywords for "ANY" option?',
        ('Yes', 'No'))

            kw_filter_remove = str(
                st.text_input(
                    label = "Remove articles containing keywords (separate by comma and space ', '): ", 
                    value = ""
                )
            )

            submit_button = st.form_submit_button(label = 'Submit')

            if submit_button:
                df_filtered = tokenized_df[lambda df: df["Facebook Interactions"] >= min_engagement]
                df_filtered = df_filtered[lambda df: df["Published"].dt.date.between(*date_range)]
                df_filtered = df_filtered[lambda df: ~df["Domain"].isin(domain_filter)]
                if len(kw_filter_remove) > 0:
                    kw_filter_remove = kw_filter_remove.split(", ")
                    df_filtered = df_filtered[lambda df: ((df["clean_Summary"].apply(lambda x: bool(set(x) & set(kw_filter_remove))) == False) & 
                                               (df["clean_Headline"].apply(lambda x: bool(set(x) & set(kw_filter_remove))) == False))]
                if len(all_kw_filter) > 0:    
                    all_kw_filter = all_kw_filter.split(", ")     
                    df_filtered = df_filtered[lambda df: ((df["clean_Summary"].apply(lambda x: bool(set(all_kw_filter).issubset(set(x)))) == True)| 
                                               (df["clean_Headline"].apply(lambda x: bool(set(all_kw_filter).issubset(set(x)))) == True)) ]
                if len(any_kw_filter) > 0:
                    if similar_option == "Yes":
                        any_kw_filter = sw.scraping_similar_phrases(any_kw_filter, 5)
                    else:
                        any_kw_filter = any_kw_filter.split(", ")  
                    df_filtered = df_filtered[lambda df: ((df_filtered["clean_Summary"].apply(lambda x: bool(set(x) & set(any_kw_filter))) == True) |
                                                (df_filtered["clean_Headline"].apply(lambda x: bool(set(x) & set(any_kw_filter))) == True))]
                    #df_filtered = df_filtered.drop(["clean_content", "clean_title"], axis = 1)
                #df_filtered = df_filtered.reset_index(True)
                df_filtered["filtered_id"] = range(len(df_filtered))
                st.session_state['df_filtered'] = df_filtered


    if "df_filtered" in st.session_state:
        st.write(f"Similar keywords considered: {any_kw_filter}")
        st.write("Number of articles: ", str(st.session_state["df_filtered"].shape[0]))
        st.dataframe(st.session_state["df_filtered"][["Published", "Headline", "Summary", "Link", "Domain", "Facebook Interactions"]])
        final_button = st.button("Discover topics in this set of articles?", key=None, help=None)

        if final_button:
            st.write("Proceed with Topic Discovery on this subset of articles!")
