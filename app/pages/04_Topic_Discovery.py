import streamlit as st
import time
import nltk
from nltk.corpus import stopwords
#nltk.download('stopwords')
#import sys
import pandas as pd
import numpy as np
from top2vec import Top2Vec
import matplotlib.pyplot as plt
#from app.topic_discovery.topic_discovery_script import wordcloud_generator
import topic_discovery.topic_discovery_script as td
from st_aggrid import AgGrid, GridUpdateMode
from st_aggrid.grid_options_builder import GridOptionsBuilder
#sys.path.append(str(Path(__file__).resolve().parent.parent))
import random

random.seed(10)

st.set_page_config(page_title = "Topic Discovery")
df = st.session_state["df_filtered"][['filtered_id', 'Headline', 'Summary', 'Link', 'Published', 'Domain', 'Facebook Interactions', 'id']]

if "Summary" in df.columns:
    df["full_text"] = df["Headline"] + df["Summary"]
else:
    df["full_text"] = df["Headline"]

st.title("Topic Discovery")
st.sidebar.markdown("# Settings")

start_time = time.perf_counter()

@st.cache(allow_output_mutation=True)
def load_model():
    if "model" in st.session_state:
        model = st.session_state["model"]
    else:
        list_documents = df["full_text"].tolist()
        hdbscan_args = {'min_cluster_size': 2,'metric': 'euclidean', 'cluster_selection_method': 'leaf', 'min_samples': 1}
        umap_args = {'n_neighbors': min(10, len(list_documents)), 'n_components': 3, 'metric': 'cosine'}
        model = Top2Vec(documents=list_documents, 
                        embedding_model="all-MiniLM-L6-v2", workers=8, min_count = 2,
                        hdbscan_args = hdbscan_args, umap_args = umap_args)
        st.session_state["model"] = model
    return model

model = load_model()
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

    #if keywords != "": #TODO: create an exception if any words in kw have not been learned by model
    #    tokenized_kw = nltk.word_tokenize(keywords)
    #    subset_topics = model.search_topics(keywords=tokenized_kw, num_topics=num_topics, reduced=red_status)[3]
    #else:
    #    subset_topics = model.get_topic_sizes(reduced=red_status)[1][0:num_topics]
    for topic_num in range(model.get_num_topics()):

        subset_of_df_in_topic = df_labelled.query('ranked_topic_number == @topic_num').reset_index(drop=True)
        wordcloud = td.wordcloud_generator(subset_of_df_in_topic, topic_num, ngram_value, "full_text")
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

        st.text("Number of articles in topic: " + str(len(subset_of_df_in_topic)))
        st.markdown("#### " + "Articles from Topic " + str(topic_num + 1))

        
        gd = GridOptionsBuilder.from_dataframe(subset_of_df_in_topic)
        gd.configure_selection(selection_mode='multiple', use_checkbox=True)
        gridoptions = gd.build()
        gridoptions['columnDefs'][0]['checkboxSelection']=True
        dict_gridtable[topic_num] = AgGrid(subset_of_df_in_topic, height=200, gridOptions=gridoptions,
                        update_mode=GridUpdateMode.SELECTION_CHANGED, key = f"aggrid_{topic_num}", reload_data = True)

        dict_checkboxes[topic_num] = st.checkbox("Tick if this topic is well-classified", key = f"checkbox_{topic_num}")
        dict_dfs[topic_num] = df_labelled[df_labelled["ranked_topic_number"]==topic_num]
        dict_topics[topic_num] = st.text_input(label = "Index label", value = "", key = f"index_label_{topic_num}")
        dict_subtopics[topic_num] = st.text_input(label = "Sub-index label (if any)", value = "", key = f"subindex_label_{topic_num}")

        
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
        removed_filtered_ids_list = []
        for topic_num in dict_dfs:
            ids_to_remove = []
            if dict_checkboxes[topic_num]:
                for selected_row in dict_gridtable[topic_num]["selected_rows"]:
                    ids_to_remove.append(selected_row["id"])
                temp_df = dict_dfs[topic_num]
                temp_df = temp_df[~temp_df.id.isin(ids_to_remove)]
                temp_df["Index"] = dict_topics[topic_num]
                temp_df["Sub-Index"] = dict_subtopics[topic_num]
                approved_dfs.append(temp_df)

        combined_df = pd.concat(approved_dfs, ignore_index = True)

        st.session_state["df_after_form_completion"] = combined_df


if "df_after_form_completion" in st.session_state:

    st.dataframe(st.session_state["df_after_form_completion"])
    if "list_of_selected_dfs" not in st.session_state:
        st.session_state["list_of_selected_dfs"] = []

    if list(st.session_state["df_after_form_completion"]["id"]) not in [list(subset_df["id"]) for subset_df in st.session_state["list_of_selected_dfs"]]:
        st.session_state["list_of_selected_dfs"].append(st.session_state["df_after_form_completion"])

    pre_filtering_df = st.session_state['initial_dataframe']

    unlabelled_data = pre_filtering_df[~pre_filtering_df.id.isin(st.session_state["df_after_form_completion"]["id"])]
    unlabelled_data["Index"] = ""
    unlabelled_data["Sub_Index"] = ""
    unlabelled_data = unlabelled_data.drop(["clean_Headline", "clean_Summary"], axis = 1)

    list_of_dfs_concatenated = pd.concat(st.session_state["list_of_selected_dfs"], ignore_index = True)
    st.session_state["intermediate_labelled_topics_df"] = list_of_dfs_concatenated
    intermediate_topics_and_unlabelled_df = pd.concat([list_of_dfs_concatenated, unlabelled_data], ignore_index = True)
    output = td.df_to_excel(intermediate_topics_and_unlabelled_df)
    

    st.download_button("Press to Download",data = output, file_name = 'df_test.xlsx', mime="application/vnd.ms-excel")
    manual_labelling_button = st.button("Label remaining filtered data based on model suggestions")
    final_button = st.button("Save remaining data to continue topic modelling on remaining data!")

    if final_button:
        st.session_state["df_remaining"] = pre_filtering_df[~pre_filtering_df.id.isin(st.session_state["df_after_form_completion"]["id"])]
        st.dataframe(st.session_state["df_remaining"])
        st.text("Return to Dataset Filters page and continue!")
        if "df_filtered" in st.session_state:
            del st.session_state['df_filtered']

    if manual_labelling_button:
        selected_topic_labels = list_of_dfs_concatenated["Index"].unique().tolist()
        st.write(selected_topic_labels)
        df_filtered = st.session_state['df_filtered']
        intermediate_labelled_topics_df = st.session_state["intermediate_labelled_topics_df"]
        st.session_state["leftover_filtered_df"] = df_filtered[~df_filtered.filtered_id.isin(intermediate_labelled_topics_df["filtered_id"])]
        dict_filtered_id_and_embedding = {}
        for filtered_id in st.session_state["leftover_filtered_df"]["filtered_id"]:
            id_embedding = model.document_vectors[filtered_id]
            dict_filtered_id_and_embedding[filtered_id] = id_embedding
        st.session_state["dict_filtered_id_and_embedding"] = dict_filtered_id_and_embedding
        dict_topic_label_and_mean_vector = {}
        #st.session_state["dict_topic_num_and_topic_label"] = dict_topics
        
        for topic_label in selected_topic_labels:
            list_of_ids_in_topic = intermediate_labelled_topics_df[intermediate_labelled_topics_df["Index"]==topic_label]["filtered_id"].tolist()
            st.write(list_of_ids_in_topic)
            list_of_embeddings_in_topic = []
            for id in list_of_ids_in_topic:
                id_embedding = model.document_vectors[id]
                list_of_embeddings_in_topic.append(id_embedding)
            mean_vector_of_topic = np.mean(list_of_embeddings_in_topic, axis = 0)
            dict_topic_label_and_mean_vector[topic_label] = mean_vector_of_topic
        st.session_state["dict_topic_label_and_mean_vector"] = dict_topic_label_and_mean_vector
## Parts idk how to account for:
### Adding custom text
### Ordering options