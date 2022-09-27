import streamlit as st
import time
import nltk
from nltk.corpus import stopwords
#nltk.download('stopwords')
from pathlib import Path
import sys
import pandas as pd
from top2vec import Top2Vec
import topic_discovery.topic_discovery_script as td
from st_aggrid import AgGrid, GridUpdateMode
from st_aggrid.grid_options_builder import GridOptionsBuilder
#sys.path.append(str(Path(__file__).resolve().parent.parent))


st.set_page_config(page_title = "Topic Discovery")

#@st.cache(allow_output_mutation=True)
#def load_news_data(data_path):
#    df = pd.read_parquet(data_path)[lambda df: df["source"] == "Online News"]
#    return df.copy()[['id', 'title', 'content', 'url', 'date', 'domain']], df["title"].to_list(), df["content"].to_list(), df["date"].to_list()

#project_folder = Path().absolute().parent
#data_path = Path(project_folder, "data", "processed", "sg_sanctions_on_russia.parquet")
#data_path = Path("data", "intermediate_data", "sg_sanctions_on_russia_filtered.parquet")
#df, titles, content, dates = load_news_data(data_path)

df = st.session_state["df_filtered"][['id', 'source', 'title', 'content', 'url', 'date', 'domain', 'actual impressions']]

#with open(Path("data", "intermediate_data", "selected_model_variable.pickle"), 'rb') as file:
#    selected_model = pickle.load(file)

title_or_content = st.session_state["title_or_content"]
selected_model = st.session_state["selected_model"]

#with open(Path("data", "intermediate_data", "title_or_content_variable.pickle"), 'rb') as file:
#    title_or_content = pickle.load(file)

st.title("Topic Discovery")
st.sidebar.markdown("# Settings")

start_time = time.perf_counter()

@st.cache(allow_output_mutation=True)
def load_model(checkpoint, title_or_content):
    if "model" in st.session_state:
        model = st.session_state["model"]
    elif checkpoint == "Top2Vec":
        model = Top2Vec(documents=df[title_or_content.lower()].to_list(), embedding_model="all-MiniLM-L6-v2", workers=8, min_count = 20)
        st.session_state["model"] = model
    return model

    #@st.cache(allow_output_mutation=True)
    #def load_model(checkpoint, title_or_content):
    #    if checkpoint == "Top2Vec":
    #        model_path = Path("scripts", f"top2vec_{title_or_content.lower()}")
    #        if model_path.is_file():
    #            model = Top2Vec.load(model_path)
    #    return model


model = load_model(selected_model, title_or_content)

st.header("Top News Topics") 

dict_gridtable = {}
dict_dfs = {}
dict_checkboxes = {}
dict_topics = {}
dict_subtopics = {}


def top_n_topics(model, df, num_topics, keywords, topic_granularity, ngram_value):
    if topic_granularity != model.get_num_topics():
        model.hierarchical_topic_reduction(num_topics=topic_granularity)
        red_status = True
    else:
        red_status = False
    df_labelled = td.generate_df_topic_labels(model, df, red_status)
    if keywords != "": #TODO: create an exception if any words in kw have not been learned by model
        tokenized_kw = nltk.word_tokenize(keywords)
        subset_topics = model.search_topics(keywords=tokenized_kw, num_topics=num_topics, reduced=red_status)[3]
    else:
        subset_topics = model.get_topic_sizes(reduced=red_status)[1][0:num_topics]
    for topic_num in subset_topics:
        fig = td.ngram_bar_visualiser(model, topic_num, red_status, ngram_value)
        st.plotly_chart(fig)
        st.text("Number of articles in topic: " + str(model.get_topic_sizes(reduced=red_status)[0][topic_num]))
        st.markdown("#### " + "Articles from Topic " + str(topic_num + 1))

        gd = GridOptionsBuilder.from_dataframe(df_labelled[df_labelled["topic_number"]==topic_num])
        gd.configure_selection(selection_mode='multiple', use_checkbox=True)
        gridoptions = gd.build()

        dict_gridtable[topic_num] = AgGrid(df, height=400, gridOptions=gridoptions,
                        update_mode=GridUpdateMode.SELECTION_CHANGED, key = f"aggrid_{topic_num}")

        dict_checkboxes[topic_num] = st.checkbox("Tick if this topic is well-classified", key = f"checkbox_{topic_num}")
        dict_dfs[topic_num] = df_labelled[df_labelled["topic_number"]==topic_num]
        dict_topics[topic_num] = st.text_input(label = "Topic label", value = "", key = f"topic_label_{topic_num}")
        dict_subtopics[topic_num] = st.text_input(label = "Sub-topic label (if any)", value = "", key = f"subtopic_label_{topic_num}")
        #st.text_input(label = "IDs of poorly classified articles (if any) separated by ', '", value = "")

        st.text(" ")
        st.text(" ")
        st.text(" ")

ngram_size = int(
    st.sidebar.slider(
        label="How many words should be in identified phrases?", min_value=1, value=2, max_value=4, step=1
    )
)

num_of_topics = int(
    st.sidebar.slider(
        label="How many topics to be displayed?",
        min_value=1,
        value=model.get_num_topics(),
        max_value=model.get_num_topics(),
        step=1,
    )
)

topic_granularity = int(
    st.sidebar.slider(
        label="How granular should topics be?", 
        min_value=1, 
        value=model.get_num_topics(), 
        max_value=model.get_num_topics(),
        step=1
    )
)

downstream_keywords = str(
    st.sidebar.text_input("Search for keywords in topics", ""
    )
)

with st.form("to_concatenate dataframes"):
    top_n_topics(model, df, num_of_topics, downstream_keywords, topic_granularity, ngram_size)
    submitted = st.form_submit_button("Submit")

    if submitted:

        approved_dfs = []
        for topic_num in dict_dfs:
            ids_to_remove = []
            if dict_checkboxes[topic_num]:
                for selected_row in dict_gridtable[topic_num]["selected_rows"]:
                    ids_to_remove.append(selected_row["id"])

                temp_df = dict_dfs[topic_num]
                temp_df = temp_df[~temp_df.id.isin(ids_to_remove)]
                temp_df["topic"] = dict_topics[topic_num]
                temp_df["subtopic"] = dict_subtopics[topic_num]
                approved_dfs.append(temp_df)

        #approved_dfs = [dict_dfs[topic_num] for topic_num in dict_checkboxes if dict_checkboxes[topic_num]]
        combined_df = pd.concat(approved_dfs, ignore_index = True)

        #nested_lst = [[dict_topics[topic_num]]*len(dict_dfs[topic_num]) for topic_num in dict_checkboxes if dict_checkboxes[topic_num]]
        #combined_df["topic"] = [item for sublist in nested_lst for item in sublist]

        #nested_lst = [[dict_subtopics[topic_num]]*len(dict_dfs[topic_num]) for topic_num in dict_checkboxes if dict_checkboxes[topic_num]]
        #combined_df["subtopic"] = [item for sublist in nested_lst for item in sublist]

        st.session_state["after_form_completion"] = combined_df
       

if "after_form_completion" in st.session_state:
    st.dataframe(st.session_state["after_form_completion"])
    output = td.df_to_excel(st.session_state["after_form_completion"])
    
    st.download_button("Press to Download",data = output.getvalue(), file_name = 'df_test.xlsx', mime="application/vnd.ms-excel")
    final_button = st.button("Save remaining data to continue topic modelling on remaining data!")#, on_click = update_remaining_df, args = [df])

    if final_button:
        pre_filtering_df = st.session_state['initial_dataframe']
        st.session_state["df_remaining"] = pre_filtering_df[~pre_filtering_df.id.isin(st.session_state["after_form_completion"]["id"])]
        st.dataframe(st.session_state["df_remaining"])
        st.text("Return to Dataset Filters page and continue!")
        #data_processed_path = Path("data", "intermediate_data", "df_remaining.parquet")
        #df_remaining = df[~df.id.isin(df["id"])]
        #df_remaining.to_parquet(data_processed_path)
       # Remove used ones from df. Save as pickle.
