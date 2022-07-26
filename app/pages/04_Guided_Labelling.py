import streamlit as st
import math
import pandas as pd
import numpy as np
import topic_discovery.topic_discovery_script as td

### sidebar ###
st.set_page_config(page_title = "Guided Labelling", layout = 'wide')
st.sidebar.markdown("# Settings")


### main page ###
st.title("Guided Labelling")

# initialising variables saved during topic discovery
dict_topic_label_and_mean_vector = st.session_state["dict_topic_label_and_mean_vector"]
leftover_filtered_df = st.session_state["leftover_filtered_df"]
leftover_filtered_df = leftover_filtered_df.set_index("filtered_id")
dict_filtered_id_and_embedding = st.session_state["dict_filtered_id_and_embedding"]
intermediate_labelled_topics_df = st.session_state["df_after_form_completion"]

curr_index = 0

with st.form("manual_labelling_form"):
    updated_list_of_topic_labels = []
    updated_list_of_subtopic_labels = []
    for filtered_id in dict_filtered_id_and_embedding:
        #order topic options by euclidean distance of existing headline embedding to topic vector of a given topic option
        ordered_labels_dict = {label: topic_vector 
                                    for label, topic_vector in sorted(dict_topic_label_and_mean_vector.items(), 
                                                                       key = lambda items: math.dist(items[1], dict_filtered_id_and_embedding[filtered_id]))}

        topic_label_options = list(ordered_labels_dict.keys())
        topic_label_options.append("None of the above")
        preview_row = leftover_filtered_df.loc[[filtered_id]][["Headline", "Summary", "Link"]].T
        preview_row = preview_row.rename(columns={preview_row.columns[0]: curr_index + 1})

        st.markdown(f"### Article {curr_index + 1}")
        st.dataframe(preview_row, width = 800)

        topic_label_choice = st.selectbox(label = "", options = topic_label_options, index = (len(topic_label_options)-1), key = filtered_id)

        topic_label_text = st.text_input("If none of the above, input topic label:", key = f'{filtered_id}_topic')
        subtopic_label_text = st.text_input("Input sub-topic label (if any):", key = f'{filtered_id}_subtopic')
        if topic_label_choice == "None of the above":
            topic_label = topic_label_text
        else:
            topic_label = topic_label_choice
        updated_list_of_topic_labels.append(topic_label)
        updated_list_of_subtopic_labels.append(subtopic_label_text)

        curr_index +=1

    submit_button = st.form_submit_button("Submit final dataframe!")

if submit_button:
    # concatenate new columns of index & subindex labels to unlabelled articles dataframe
    leftover_filtered_df["Index"] = updated_list_of_topic_labels
    leftover_filtered_df["Sub-Index"] = updated_list_of_subtopic_labels

    # concatenate both labelled articles' dataframes together and preview. Order by index & sub-index in alphabetical order
    final_df = pd.concat([intermediate_labelled_topics_df, leftover_filtered_df], axis = 0, ignore_index=True)
    ordered_topics_df = final_df.groupby(["Index", "Sub-Index"], as_index=False).agg(fb_sum= ('Facebook Interactions', 'sum')).sort_values('fb_sum', ascending=False)
    final_df = pd.merge(final_df, ordered_topics_df, how = 'left', on=['Index', 'Sub-Index'])
    final_df = final_df.sort_values(['fb_sum', 'Headline'], ascending=False)
    final_df = final_df.drop(["filtered_id", "id", "clean_Summary", "clean_Headline", "full_text", "ranked_topic_number", "topic_number", "fb_sum"], axis = 1)
    final_df = pd.concat([final_df[final_df['Index']!=""], final_df[final_df['Index']==""]], ignore_index=False).reset_index(drop=True)

    st.dataframe(final_df)
    pivot_df = pd.pivot_table(final_df, 
                                values = ["Facebook Interactions"], 
                                index = ["Index", "Sub-Index"], aggfunc={'Facebook Interactions': [np.sum, 
                                                                                                    (lambda x: np.round(100*np.sum(x)/final_df["Facebook Interactions"].sum(), 1))]})
    pivot_df.columns.set_levels(['Percentage','Sum'],level=1,inplace=True)

    # output dataframe as excel file for download as XLSX
    output = td.df_to_excel_both(final_df, pivot_df)
    st.download_button("Press to Download", data = output, file_name = f'{st.session_state["file_name"]}_labelled.xlsx', mime="application/vnd.ms-excel")
