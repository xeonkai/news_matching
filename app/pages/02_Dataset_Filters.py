import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np

st.set_page_config(page_title = "Dataset Filters",
                  layout="wide")

@st.cache(allow_output_mutation=True)
def load_news_data(data_path):
    df = pd.read_parquet(data_path)[lambda df: df["source"] == "Online News"]
    return df.copy(), df["title"].to_list(), df["content"].to_list(), df["date"].to_list(), df["actual impressions"].to_list()

project_folder = Path().absolute().parent
data_path = Path(project_folder, "data", "processed", "sg_sanctions_on_russia.parquet")
df, titles, content, dates, actual_imp = load_news_data(data_path)


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
        
st.subheader("Filtered Dataset")

st.session_state["df_filtered"] = None


##### sidebar #####

with st.sidebar:
    with st.form(key = "filter_params"):
        st.markdown("# Filters")
        kw_filter = str(
            st.text_input(
                label = "Keep articles containing specific keywords:", 
                value = ""
            )
        )

        min_engagement = int(
            st.slider(
                label = "Minimum no. of Facebook engagements:", 
                min_value = 1,
                max_value = max(actual_imp),
                step = 50
            )
        )
        
        submit_button = st.form_submit_button(label = 'Submit')
        
        if submit_button:
            if len(kw_filter) > 0:
                kw_list = kw_filter.split(", ")
                text_column = df["content"].str.lower()
                
            df_filtered = df[lambda df: df["actual impressions"] >= min_engagement]
            
            st.session_state['df_filtered'] = df_filtered

if st.session_state["df_filtered"] is not None:
    st.write("Number of articles: ", str(st.session_state["df_filtered"].shape[0]))
    st.dataframe(st.session_state["df_filtered"])


selected_model = st.sidebar.selectbox("Model Type:", ["Top2Vec"])