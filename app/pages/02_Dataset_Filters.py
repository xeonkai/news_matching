import streamlit as st
from pathlib import Path
import pandas as pd
import sys

sys.path.append("./scripts")

from similar_words import scraping_similar_phrases
from preprocess_utils import tokenised_preprocessing

st.set_page_config(page_title = "Dataset Filters",
                  layout="wide")

#@st.cache(allow_output_mutation=True)
#def load_news_data(data_path):
#    df = pd.read_parquet(data_path)[lambda df: df["source"] == "Online News"]
#    return df.copy(), df["title"].to_list(), df["content"].to_list(), df["date"].to_list(), df["actual impressions"].to_list(), tokenised_preprocessing(df, "content")

#project_folder = Path().absolute().parent
#data_path = Path(project_folder, "data", "processed", "sg_sanctions_on_russia.parquet")
#data_path = Path("data", "processed", "sg_sanctions_on_russia.parquet")
#df, titles, content, dates, actual_imp, tokenized_df = load_news_data(data_path)

if "model" in st.session_state:
    del st.session_state['model']
if "after_form_completion" in st.session_state:
    del st.session_state['after_form_completion']
if "title_or_content" in st.session_state:
    del st.session_state['title_or_content']
if "selected_model" in st.session_state:
    del st.session_state['selected_model']

tokenized_df = None

##### main page #####

st.title("Dataset Filters")
daily_news = st.file_uploader("Upload news dataset here:", type=["csv", "xlsx"])

st.subheader("Dataset")

#data_path = Path("data", "intermediate_data", "df_remaining.parquet.parquet")
#if data_path.is_file():
#    df = pd.read_parquet(data_path)
#    tokenized_df = tokenised_preprocessing(df, "content") 
#st.session_state["df_filtered"] = None

if 'df_remaining' in st.session_state:
    df = st.session_state["df_remaining"]
    tokenized_df = tokenised_preprocessing(df, "content")
    st.session_state['initial_dataframe'] = st.dataframe(df)

elif daily_news is not None:
    if daily_news.type == "text/csv":
        file_details = {"filename": daily_news.name, "filetype": daily_news.type,
                                "filesize": daily_news.size}
        df = pd.read_csv(daily_news, encoding = "latin1")
        st.write("Uploaded dataset: ", file_details["filename"])
        st.write("Number of articles: ", str(df.shape[0]))
        st.dataframe(df)
    elif daily_news.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        file_details = {"filename": daily_news.name, "filetype": daily_news.type,
                                "filesize": daily_news.size}
        df = pd.read_excel(daily_news)
        st.write("Uploaded dataset: ", file_details["filename"])
        st.write("Number of articles: ", str(df.shape[0]))
        st.dataframe(df)
    tokenized_df = tokenised_preprocessing(df, "content") 
    st.session_state['initial_dataframe'] = df

    st.subheader("Filtered Dataset")

    #st.session_state["df_filtered"] = None


    ##### sidebar #####
if tokenized_df is not None:
    with st.sidebar:

        with st.form(key = "filter_params"):
            st.markdown("# Filters")

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

            min_engagement = int(
                st.slider(
                    label = "Minimum no. of Facebook engagements:", 
                    min_value = 1,
                    max_value = max(tokenized_df["actual impressions"]),
                    step = 50
                )
            )

            #date_range = st.date_input("Date range of articles", 
            #            value = (tokenized_df["date"].min(), tokenized_df["date"].max()),
            #            min_value=min(tokenized_df["date"]),
            #            max_value=max(tokenized_df["date"]))


            domain_filter = st.multiselect(label = "Article domains to exclude", options = tokenized_df["domain"].unique(), default = None)
            submit_button = st.form_submit_button(label = 'Submit')

            if submit_button:
                df_filtered = tokenized_df[lambda df: df["actual impressions"] >= min_engagement]
                #df_filtered = df_filtered[lambda df: df["date"].dt.date.between(*date_range)]
                df_filtered = df_filtered[lambda df: ~df["domain"].isin(domain_filter)]
                if len(kw_filter_remove) > 0:
                    kw_filter_remove = kw_filter_remove.split(", ")
                    df_filtered = df_filtered[lambda df: df["clean_content"].apply(lambda x: bool(set(x) & set(kw_filter_remove))) == False]
                if len(all_kw_filter) > 0:    
                    all_kw_filter = all_kw_filter.split(", ")     
                    df_filtered = df_filtered[lambda df: df["clean_content"].apply(lambda x: bool(set(all_kw_filter).issubset(set(x)))) == True]
                if len(any_kw_filter) > 0:
                    if similar_option == "Yes":
                        any_kw_filter = scraping_similar_phrases(any_kw_filter, 5)
                    else:
                        any_kw_filter = any_kw_filter.split(", ")  
                    #text_column = df["content"].str.lower()
                    df_filtered = df_filtered[df_filtered["clean_content"].apply(lambda x: bool(set(x) & set(any_kw_filter))) == True]
                    df_filtered = df_filtered.drop(["clean_content"], axis = 1)
                #df_filtered = df_filtered.reset_index(True)
                st.session_state['df_filtered'] = df_filtered
                #YOU FORGOT TO SAVE UNUSED ARTICLES!!!!


    #def save_filtered_dataset(df):
    #    data_processed_path = Path("data", "intermediate_data", "sg_sanctions_on_russia_filtered.parquet")
    #    df.to_parquet(data_processed_path)

    if "df_filtered" in st.session_state:
        st.write(f"Similar keywords considered: {any_kw_filter}")
        st.write("Number of articles: ", str(st.session_state["df_filtered"].shape[0]))
        st.dataframe(st.session_state["df_filtered"])
        st.button("Proceed with this subset of articles?", key=None, help=None)
        #st.dataframe(st.session_state['initial_dataframe'])

