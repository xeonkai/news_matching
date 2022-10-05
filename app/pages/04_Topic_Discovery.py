import streamlit as st
import time
import nltk
from nltk.corpus import stopwords
#nltk.download('stopwords')
#import sys
import pandas as pd
from top2vec import Top2Vec
import matplotlib.pyplot as plt
#from app.topic_discovery.topic_discovery_script import wordcloud_generator
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

#df = st.session_state["df_filtered"][['id', 'source', 'title', 'content', 'url', 'date', 'domain', 'actual impressions']]
df = st.session_state["df_filtered"][['id', 'title', 'content', 'url', 'date', 'domain', 'actual impressions']]
title_or_content = st.session_state["title_or_content"]
selected_model = st.session_state["selected_model"]

st.title("Topic Discovery")
st.sidebar.markdown("# Settings")

start_time = time.perf_counter()

@st.cache(allow_output_mutation=True)
def load_model(checkpoint, title_or_content):
    if "model" in st.session_state:
        model = st.session_state["model"]
    elif checkpoint == "Top2Vec":
        hdbscan_args = {'min_cluster_size': 2,'metric': 'euclidean', 'cluster_selection_method': 'leaf', 'min_samples': 1}
        umap_args = {'n_neighbors': min(10, len(df[title_or_content.lower()].to_list())), 'n_components': 3, 'metric': 'cosine'}
        model = Top2Vec(documents=df[title_or_content.lower()].to_list(), 
                        embedding_model="all-MiniLM-L6-v2", workers=8, min_count = 2,
                        hdbscan_args = hdbscan_args, umap_args = umap_args)
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

        subset_of_df_in_topic = df_labelled.query('topic_number == @topic_num').reset_index(drop=True)

        wordcloud = td.wordcloud_generator(subset_of_df_in_topic, topic_num, ngram_value, title_or_content.lower())
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
        #fig = td.ngram_bar_visualiser(model, topic_num, red_status, ngram_value)
        #st.plotly_chart(fig)
        st.text("Number of articles in topic: " + str(model.get_topic_sizes(reduced=red_status)[0][topic_num]))
        st.markdown("#### " + "Articles from Topic " + str(topic_num + 1))

        
        gd = GridOptionsBuilder.from_dataframe(subset_of_df_in_topic)
        gd.configure_selection(selection_mode='multiple', use_checkbox=True)
        gridoptions = gd.build()

        dict_gridtable[topic_num] = AgGrid(subset_of_df_in_topic, height=400, gridOptions=gridoptions,
                        update_mode=GridUpdateMode.SELECTION_CHANGED, key = f"aggrid_{topic_num}")

        dict_checkboxes[topic_num] = st.checkbox("Tick if this topic is well-classified", key = f"checkbox_{topic_num}")
        dict_dfs[topic_num] = df_labelled[df_labelled["topic_number"]==topic_num]
        dict_topics[topic_num] = st.text_input(label = "Topic label", value = "", key = f"topic_label_{topic_num}")
        dict_subtopics[topic_num] = st.text_input(label = "Sub-topic label (if any)", value = "", key = f"subtopic_label_{topic_num}")

        
        st.text(" ")
        st.text(" ")
        st.text(" ")

ngram_size = int(
    st.sidebar.slider(
        label="How many words should be in identified phrases?", min_value=1, value=1, max_value=3, step=1
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

        combined_df = pd.concat(approved_dfs, ignore_index = True)

        st.session_state["df_after_form_completion"] = combined_df


if "df_after_form_completion" in st.session_state:

    st.dataframe(st.session_state["df_after_form_completion"])
    if "list_of_selected_dfs" not in st.session_state:
        st.session_state["list_of_selected_dfs"] = []

    if list(st.session_state["df_after_form_completion"]["id"]) not in [list(subset_df["id"]) for subset_df in st.session_state["list_of_selected_dfs"]]: #this line has issues
        st.session_state["list_of_selected_dfs"].append(st.session_state["df_after_form_completion"])

    pre_filtering_df = st.session_state['initial_dataframe']

    unlabelled_data = pre_filtering_df[~pre_filtering_df.id.isin(st.session_state["df_after_form_completion"]["id"])]
    unlabelled_data["topic"] = ""
    unlabelled_data["subtopic"] = ""
    unlabelled_data = unlabelled_data.drop(["clean_title", "clean_content"], axis = 1)

    list_of_dfs_concatenated = pd.concat(st.session_state["list_of_selected_dfs"], ignore_index = True)
    output = td.df_to_excel(pd.concat([list_of_dfs_concatenated, unlabelled_data], ignore_index = True))
    

    st.download_button("Press to Download",data = output, file_name = 'df_test.xlsx', mime="application/vnd.ms-excel")
    final_button = st.button("Save remaining data to continue topic modelling on remaining data!")

    if final_button:
        st.session_state["df_remaining"] = pre_filtering_df[~pre_filtering_df.id.isin(st.session_state["df_after_form_completion"]["id"])]
        st.dataframe(st.session_state["df_remaining"])
        st.text("Return to Dataset Filters page and continue!")
        if "df_filtered" in st.session_state:
            del st.session_state['df_filtered']
