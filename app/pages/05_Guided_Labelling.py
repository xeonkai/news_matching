import streamlit as st
import math
import pandas as pd
import topic_discovery.topic_discovery_script as td

st.set_page_config(page_title = "Guided Labelling")

st.title("Guided Labelling")
st.sidebar.markdown("# Settings")

dict_topic_label_and_mean_vector = st.session_state["dict_topic_label_and_mean_vector"]

leftover_filtered_df = st.session_state["leftover_filtered_df"]
leftover_filtered_df = leftover_filtered_df.set_index("filtered_id")
dict_filtered_id_and_embedding = st.session_state["dict_filtered_id_and_embedding"]

intermediate_labelled_topics_df = st.session_state["intermediate_labelled_topics_df"]

if "current_headline_index" not in st.session_state:
    st.session_state["current_headline_index"] = 0


with st.form("manual_labelling_form"):
    updated_list_of_topic_labels = []
    for filtered_id in dict_filtered_id_and_embedding:
        ordered_labels_dict = {label: topic_vector 
                                    for label, topic_vector in sorted(dict_topic_label_and_mean_vector.items(), 
                                                                       key = lambda items: math.dist(items[1], dict_filtered_id_and_embedding[filtered_id]))}
        curr_headline = leftover_filtered_df.loc[filtered_id]["Headline"]
        topic_label_options = list(ordered_labels_dict.keys())
        topic_label_options.append("None of the above")
        #st.write(topic_label_options)
        st.markdown(curr_headline)
        topic_label_choice = st.radio(label = "", options= topic_label_options, index = (len(topic_label_options)-1), key = filtered_id)

        topic_label_text = st.text_input("If none of the above, input topic label:", key = filtered_id)

        if topic_label_choice == "None of the above":
            topic_label = topic_label_text
        else:
            topic_label = topic_label_choice
        updated_list_of_topic_labels.append(topic_label)
    submit_button = st.form_submit_button("Submit final dataframe!")

if submit_button:
    leftover_filtered_df["Index"] = updated_list_of_topic_labels
    #st.dataframe(leftover_filtered_df)
    st.dataframe(leftover_filtered_df)
    #leftover_filtered_df = leftover_filtered_df[leftover_filtered_df["topic"]!=""]
    final_df = pd.concat([intermediate_labelled_topics_df, leftover_filtered_df], axis = 0, ignore_index=True)
    final_df = final_df.drop(["filtered_id", "id", "clean_Summary", "clean_Headline"], axis = 1)
    output = td.df_to_excel(final_df)

    st.download_button("Press to Download", data = output, file_name = 'df_test.xlsx', mime="application/vnd.ms-excel")
