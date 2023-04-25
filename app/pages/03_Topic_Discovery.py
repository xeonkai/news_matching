import streamlit as st
import nltk
import pandas as pd
import numpy as np
from top2vec import Top2Vec
import matplotlib.pyplot as plt
import topic_discovery.topic_discovery_script as td
from st_aggrid import AgGrid, GridUpdateMode
from st_aggrid.grid_options_builder import GridOptionsBuilder
import random
from warnings import simplefilter


### code settings ###
simplefilter(action="ignore", category=FutureWarning)
random.seed(10)

### sidebar ###
st.set_page_config(page_title="Topic Discovery", layout="wide")
st.sidebar.markdown("# Settings")


### main page ###
st.title("Topic Discovery")


@st.cache(allow_output_mutation=True)
def load_model(df):
    """

    Run Top2Vec on full_text column in df and output model

    """
    if "model" in st.session_state:
        model = st.session_state["model"]
    else:
        list_documents = df["full_text"].tolist()
        hdbscan_args = {
            "min_cluster_size": 2,
            "metric": "euclidean",
            "cluster_selection_method": "leaf",
            "min_samples": 1,
        }
        umap_args = {
            "n_neighbors": min(5, len(list_documents)),
            "n_components": 3,
            "metric": "cosine",
        }
        model = Top2Vec(
            documents=list_documents,
            embedding_model="all-MiniLM-L6-v2",
            workers=8,
            min_count=1,
            hdbscan_args=hdbscan_args,
            umap_args=umap_args,
        )
        st.session_state["model"] = model
    return model


df = st.session_state["df_filtered"][
    [
        "filtered_id",
        "Headline",
        "Summary",
        "Link",
        "Published",
        "Domain",
        "Facebook Interactions",
        "id",
    ]
]
df["full_text"] = df[["Headline", "Summary"]].agg(" ".join, axis=1)
model = load_model(df)


st.header("Top News Topics")

# Initialise dictionaries for storage of user input data
dict_gridtable = {}
dict_dfs = {}
# dict_checkboxes = {}
dict_topics = {}
dict_subtopics = {}


def top_n_topics(model, df, keywords, topic_granularity):
    """
    Iterate through all topics identified by Top2Vec model, sorted by descending Facebook engagement, and print
    topic number, topic-specific phrase cloud, topic-specific dataframe, and basic details of topic

    Input:

    model - Top2Vec modelled on all articles in "df"
    df (pandas.DataFrame) - complete dataframe without labels
    keywords (string) - If non-empty, order topics based on relevance to input keywords
    topic_granularity (int) - Number of expected topics

    """
    if topic_granularity != model.get_num_topics():
        model.hierarchical_topic_reduction(num_topics=topic_granularity)
        red_status = True
    else:
        red_status = False
    df_labelled, topics_ranked_by_fb = td.generate_df_topic_labels(
        model, df, red_status
    )

    if keywords != "":
        if keywords in model.vocab:
            tokenized_kw = nltk.word_tokenize(keywords)
            subset_topics = model.search_topics(
                keywords=tokenized_kw, num_topics=topic_granularity, reduced=red_status
            )[3]
            subset_ranked_topics = [
                topics_ranked_by_fb[topic_num] for topic_num in subset_topics
            ]
        else:
            st.write(
                "Keywords not found in any article. Performing topic discovery as per normal."
            )
            subset_ranked_topics = list(range(topic_granularity))
    else:
        subset_ranked_topics = list(range(topic_granularity))

    for topic_num in subset_ranked_topics:
        subset_of_df_in_topic = df_labelled.query(
            "ranked_topic_number == @topic_num"
        ).reset_index(drop=True)

        st.markdown(f"##### Topic {topic_num + 1}")
        st.text(f"Number of articles in topic: {len(subset_of_df_in_topic)}")
        st.text(
            f"Total number of Facebook interactions: {sum(subset_of_df_in_topic['Facebook Interactions'])}"
        )
        gd = GridOptionsBuilder.from_dataframe(
            subset_of_df_in_topic[
                [
                    "Headline",
                    "Summary",
                    "Link",
                    "Published",
                    "Domain",
                    "Facebook Interactions",
                ]
            ],
            min_column_width=600,
        )
        gd.configure_selection(selection_mode="multiple", use_checkbox=True)
        
        #added
        gd.configure_column('Headline', headerCheckboxSelection=True)

        gridoptions = gd.build()
        dict_gridtable[topic_num] = AgGrid(
            subset_of_df_in_topic,
            height=300,
            gridOptions=gridoptions,
            fit_columns_on_grid_load=True,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            key=f"aggrid_{topic_num}",
            reload_data=True,
        )
        # dict_checkboxes[topic_num] = st.checkbox(
        #     "Tick if this topic is well-classified", key=f"checkbox_{topic_num}"
        # )
        dict_dfs[topic_num] = df_labelled[
            df_labelled["ranked_topic_number"] == topic_num
        ]
        dict_topics[topic_num] = st.text_input(
            label="Index label", value="", key=f"index_label_{topic_num}"
        )
        dict_subtopics[topic_num] = st.text_input(
            label="Sub-index label (if any)",
            value="",
            key=f"subindex_label_{topic_num}",
        )

        st.text(" ")
        st.text(" ")
        st.text(" ")


with st.sidebar:
    st.header("Top 10 Articles")
    df_filtered = st.session_state["df_filtered"]
    top_articles = (
        df_filtered.sort_values(by=["Facebook Interactions"], ascending=False)
        .head(10)
        .reset_index()
    )
    st.dataframe(
        top_articles[
            [
                "Headline",
                "Summary",
                "Link",
                "Published",
                "Domain",
                "Facebook Interactions",
            ]
        ]
    )

    # Number of topics expected
    topic_granularity = int(
        st.sidebar.slider(
            label="How granular should topics be?",
            min_value=1,
            value=model.get_num_topics(),
            max_value=model.get_num_topics(),
            step=1,
        )
    )

    # Orders topics by relevance to keywords
    downstream_keywords = str(
        st.sidebar.text_input("Search for keywords in topics", "")
    )

#to change this after adding in select all checkbox
# Upon submitting form,
with st.form("to_concatenate dataframes"):
    top_n_topics(model, df, downstream_keywords, topic_granularity)
    submitted = st.form_submit_button("Submit")

    if submitted:
        # approved_dfs = []
        # removed_filtered_ids_list = []
        # for topic_num in dict_dfs:
        #     ids_to_remove = []
        #     if dict_checkboxes[topic_num]:
        #         for selected_row in dict_gridtable[topic_num]["selected_rows"]:
        #             ids_to_remove.append(selected_row["id"])
        #         temp_df = dict_dfs[topic_num]
        #         # Removes articles that user manually indicated as irrelevant within a topic indicated as correctly-classified.
        #         temp_df = temp_df[~temp_df.id.isin(ids_to_remove)]
        #         # Allows user to indicate topic and sub-topic labels for all correctly-classified topics in bulk
        #         temp_df["Index"] = dict_topics[topic_num]
        #         temp_df["Sub-Index"] = dict_subtopics[topic_num]
        #         approved_dfs.append(temp_df)
        # # Concatenates dataframes that user identified as relevant topics
        # combined_df = pd.concat(approved_dfs, ignore_index=True)

        # st.session_state["df_after_form_completion"] = combined_df

        approved_dfs = []
        for topic_num in dict_dfs:
            temp_df = pd.DataFrame(dict_gridtable[topic_num]["selected_rows"])
            if temp_df.shape[0] > 0: #if more than 1 row selected
                temp_df["Index"] = dict_topics[topic_num]
                temp_df["Sub-Index"] = dict_subtopics[topic_num]
                approved_dfs.append(temp_df)

        combined_df = pd.concat(approved_dfs, ignore_index=True)

        st.session_state["df_after_form_completion"] = combined_df
    
if "df_after_form_completion" in st.session_state:
    # Preview concatenated dataframe of labelled articles
    st.dataframe(
        st.session_state["df_after_form_completion"][
            [
                "Headline",
                "Summary",
                "Link",
                "Published",
                "Domain",
                "Facebook Interactions",
                "Index",
                "Sub-Index",
            ]
        ]
    )

    filtered_df = st.session_state["df_filtered"]
    unlabelled_data = filtered_df[
        ~filtered_df.id.isin(st.session_state["df_after_form_completion"]["id"])
    ]
    unlabelled_data["Index"] = ""
    unlabelled_data["Sub-Index"] = ""
    unlabelled_data = unlabelled_data.drop(["clean_Headline", "clean_Summary"], axis=1)

    intermediate_topics_and_unlabelled_df = pd.concat(
        [st.session_state["df_after_form_completion"], unlabelled_data],
        ignore_index=True,
    )
    intermediate_topics_and_unlabelled_df = intermediate_topics_and_unlabelled_df[
        [
            "Headline",
            "Summary",
            "Link",
            "Published",
            "Domain",
            "Facebook Interactions",
            "Index",
            "Sub-Index",
        ]
    ]
    output = td.df_to_excel(intermediate_topics_and_unlabelled_df)

    st.download_button(
        "Press to Download",
        data=output,
        file_name=f'{st.session_state["file_name"]}_labelled_intermediate.xlsx',
        mime="application/vnd.ms-excel",
    )
    manual_labelling_button = st.button(
        "Label remaining filtered data based on model suggestions"
    )

    # saving specific embeddings for guided labelling
    if manual_labelling_button:
        intermediate_labelled_topics_df = st.session_state["df_after_form_completion"]
        selected_topic_labels = (
            intermediate_labelled_topics_df["Index"].unique().tolist()
        )
        df_filtered = st.session_state["df_filtered"]
        st.session_state["leftover_filtered_df"] = df_filtered[
            ~df_filtered.filtered_id.isin(
                intermediate_labelled_topics_df["filtered_id"]
            )
        ]
        dict_filtered_id_and_embedding = {}
        # save a dictionary of embeddings of unlabelled articles
        for filtered_id in st.session_state["leftover_filtered_df"]["filtered_id"]:
            id_embedding = model.document_vectors[filtered_id]
            dict_filtered_id_and_embedding[filtered_id] = id_embedding
        st.session_state[
            "dict_filtered_id_and_embedding"
        ] = dict_filtered_id_and_embedding

        # save a dictionary of topic labels and their corresponding calculated topic vector for guided labelling
        dict_topic_label_and_mean_vector = {}

        for topic_label in selected_topic_labels:
            list_of_ids_in_topic = intermediate_labelled_topics_df[
                intermediate_labelled_topics_df["Index"] == topic_label
            ]["filtered_id"].tolist()
            list_of_embeddings_in_topic = []
            for id in list_of_ids_in_topic:
                id_embedding = model.document_vectors[id]
                list_of_embeddings_in_topic.append(id_embedding)
            mean_vector_of_topic = np.mean(list_of_embeddings_in_topic, axis=0)
            dict_topic_label_and_mean_vector[topic_label] = mean_vector_of_topic
        st.session_state[
            "dict_topic_label_and_mean_vector"
        ] = dict_topic_label_and_mean_vector

        st.write("Proceed with Guided Labelling on remaining articles!")
