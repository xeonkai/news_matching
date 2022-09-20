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


##### main page #####

st.title("Dataset Filters")
daily_news = st.file_uploader("Upload news dataset here:", type=["csv", "xlsx"])

st.subheader("Dataset")

if daily_news is not None:
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
     
#TODO: write a function to compress preprocessing steps together  
  

    st.subheader("Filtered Dataset")

    st.session_state["df_filtered"] = None


    ##### sidebar #####

    with st.sidebar:

        with st.form(key = "filter_params"):
            st.markdown("# Filters")
            kw_filter = str(
                st.text_input(
                    label = "Keep articles containing keywords (separate by comma and space ', '): ", 
                    value = ""
                )
            )

            similar_option = st.selectbox('Consider similar keywords to input keywords?',
        ('Yes', 'No'))

            min_engagement = int(
                st.slider(
                    label = "Minimum no. of Facebook engagements:", 
                    min_value = 1,
                    max_value = max(tokenized_df["actual impressions"]),
                    step = 50
                )
            )

            date_range = st.date_input("Date range of articles", 
                        value = (tokenized_df["date"].min(), tokenized_df["date"].max()),
                        min_value=min(tokenized_df["date"]),
                        max_value=max(tokenized_df["date"]))

            submit_button = st.form_submit_button(label = 'Submit')

            if submit_button:
                df_filtered = tokenized_df[lambda df: df["actual impressions"] >= min_engagement]
                df_filtered = df_filtered[lambda df: df["date"].dt.date.between(*date_range)]
                if len(kw_filter) > 0:
                    if similar_option == "Yes":
                        kw_filter = scraping_similar_phrases(kw_filter, 5)
                    else:
                        kw_filter = [kw_filter]
                    #text_column = df["content"].str.lower()
                    df_filtered = df_filtered[df_filtered["clean_content"].apply(lambda x: bool(set(x) & set(kw_filter))) == True]
                    df_filtered = df_filtered.drop(["clean_content"], axis = 1)
                
                st.session_state['df_filtered'] = df_filtered

    def save_filtered_dataset(df):
        data_processed_path = Path("data", "intermediate_data", "sg_sanctions_on_russia_filtered.parquet")
        df.to_parquet(data_processed_path)

    if st.session_state["df_filtered"] is not None:
        st.write(f"Similar keywords considered: {kw_filter}")
        st.write("Number of articles: ", str(st.session_state["df_filtered"].shape[0]))
        st.dataframe(st.session_state["df_filtered"])
        st.button("Proceed with this subset of articles?", key=None, help=None, on_click=save_filtered_dataset, args=[df_filtered])





