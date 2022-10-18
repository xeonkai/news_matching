from top2vec import Top2Vec
from bertopic import BERTopic
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
import yaml
from hdbscan import HDBSCAN

def validate_model_with_all_files(directory, model_name, text_column="", embedding_model="", hdbscan_args="", umap_args="", vectorizer_args=""):
    """
    Tests performance of input model & parameters on all files in directory

    Args:
        1. directory (str): Path to directory containing all test data in xlsx format under "Sheet1". Should contain columns: 'Date', 'Headline', 'Index', 'Sub-Index'. 'Summary' column optional.
        2. model_name (str): 'Top2Vec' or 'BERTopic'
        3. text_column (str) (optional, defaults to Headline): 'Headline' or 'Summary' or 'Both'. If both, Headline and Summary are concatenated for modelling
        4. embedding_model (str) (optional, defaults to all-MiniLM-L6-v2): Embedding model used on text_column during modelling
        5. hdbscan_args (str) (optional): Parameters of HDBSCAN for hierarchical clustering
        6. umap_args (str) (optional): Parameters of UMAP for dimensionality reduction
        7. vectorizer_args (str) (optional): Parameters for CountVectorizer for BERTopic
    
    Returns:
        1. Dataframe of performance of input model & parameters on each test set in directory
            Contains additional columns:
            - Num_Articles_Incl_Anom: Total number of articles in test set
            - Num_Articles_W_SubIndex: Number of articles in test set with non-empty Sub-Index
            - Num_Articles_Excl_Anom: Number of articles in test set with non-unique true Index
            - Pct_Correct_Index: Percentage of articles whose predicted index match true index
            - Pct_Correct_Index_and_Sub_Index: Percentage of articles whose predicted index match true index and predicted sub-index match true sub-index (amongst articles where sub-index is non-empty)
            - Pct_Correct_Index_Excl_Anom: Percentage of articles whose predicted index match true index (amongst articles in test set with non-unique true Index)

    """
    if hdbscan_args=="":
        if model_name=="Top2Vec":
            hdbscan_args = {'min_cluster_size': 2,'metric': 'euclidean', 'cluster_selection_method': 'leaf', 'min_samples': 1, 'allow_single_cluster': True}#, 'cluster_selection_epsilon': 0.2}
        elif model_name=="BERTopic":
            hdbscan_args = {'min_cluster_size': 3,'metric': 'euclidean', 'cluster_selection_method': 'leaf', 'min_samples': 1, 'allow_single_cluster': True, 'prediction_data': True}
    else:
        hdbscan_args = yaml.load(hdbscan_args)
    
    if umap_args=="":    
        umap_args = {'n_neighbors': 10,'n_components': 3, 'metric': 'cosine'}
    else:
        umap_args = yaml.load(umap_args)
    
    if vectorizer_args=="":
        if model_name=="BERTopic":
            vectorizer_args = {'ngram_range': (1, 1), 'stop_words': 'english'}
    else:
        vectorizer_args= yaml.load(vectorizer_args)
    
    if embedding_model=="":
        embedding_model = "all-MiniLM-L6-v2"

    if text_column=="":
        text_column = "Headline"

    model_validation_lst = []
    for filename in os.listdir(directory):
        if ".xlsx" in filename:
            #print(filename)
            file_path = directory + "/" + filename
            data = pd.read_excel(file_path, sheet_name = 'Sheet1')
            data = data[data["Index"].notnull()]
            if isinstance(data["Published"][0], str):
                data_date = datetime.fromisoformat(data["Published"][0]).strftime("%d-%m-%Y")
            elif isinstance(data["Published"][0], pd._libs.tslibs.timestamps.Timestamp):
                data_date = data["Published"][0].strftime("%d-%m-%Y")
        else:
            continue

        if text_column=="Both":
            if "Summary" in data.columns:
                data["Both"] = data["Headline"] + data["Summary"]
            else:
                text_column="Headline"

        if model_name=="Top2Vec":
            model = Top2Vec(documents=list(data[text_column]), workers=8,  min_count = 2,
                hdbscan_args= hdbscan_args, umap_args = umap_args, embedding_model=embedding_model, verbose=False)
            
            topic_nums = model.get_documents_topics(list(range(0,len(data))))[0]

        if model_name=="BERTopic":
            hdbscan_model = HDBSCAN(**hdbscan_args)
            umap_model = UMAP(**umap_args)
            vectorizer_model = CountVectorizer(**vectorizer_args)

            model = BERTopic(embedding_model=embedding_model, hdbscan_model=hdbscan_model, umap_model=umap_model,
                vectorizer_model= vectorizer_model, language="english", calculate_probabilities=True,verbose=False) #min_topic_size
            
            topic_nums = model.fit_transform(list(data[text_column]))[0]
            
        data["topic_num"] = topic_nums
        #METRIC 1
        topic_num_and_modal_index_df = data.groupby(['topic_num'])['Index'].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.mode()).reset_index().rename(columns={"Index":"model_index"})
        labelled_data = data.merge(topic_num_and_modal_index_df, how = 'left', on = 'topic_num')
        labelled_data["index_match_status"] = labelled_data.apply(lambda x: x.Index == x.model_index, axis= 1)
        length_df_incl_anomalies = len(labelled_data)

        #PERCENTAGE OF ARTICLES CORRECTLY CLASSIFIED INTO INDEXED TOPICS (INCLUDING ARTICLES W ONLY ONE ARTICLE IN INDEXED TOPIC)
        pct_indexed_topics_incl_anomalies = (len(labelled_data[labelled_data["index_match_status"]])*100)/length_df_incl_anomalies

        #METRIC 2
        data_w_subindex = data[data["Sub-Index"].notnull()]
        topic_num_and_modal_subindex_df = data_w_subindex.groupby(['topic_num'])['Sub-Index'].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.mode()).reset_index().rename(columns={"Sub-Index":"model_sub_index"})
        labelled_data = data_w_subindex.merge(topic_num_and_modal_subindex_df, how = 'left', on = 'topic_num')
        labelled_data["sub_index_match_status"] = labelled_data.apply(lambda x: True if x["Sub-Index"] == x["model_sub_index"] else False, axis= 1)
        length_df_w_subindex = len(labelled_data)

        #PERCENTAGE OF ARTICLES CORRECTLY CLASSIFIED INTO INDEXED TOPIC AND SUB TOPIC (INCLUDING ARTICLES W ONLY ONE ARTICLE IN INDEXED SUBTOPIC)
        pct_sub_indexed_topics_incl_anomalies = (len(labelled_data[labelled_data["sub_index_match_status"]])*100)/length_df_w_subindex

        #METRIC 3
        data_excl_anomalies = data.groupby("Index").filter(lambda x: len(x) > 1)
        topic_num_and_modal_index_df = data_excl_anomalies.groupby(['topic_num'])['Index'].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.mode()).reset_index().rename(columns={"Index":"model_index"})
        labelled_data = data_excl_anomalies.merge(topic_num_and_modal_index_df, how = 'left', on = 'topic_num')
        labelled_data["index_match_status"] = labelled_data.apply(lambda x: x.Index == x.model_index, axis= 1)
        length_df_excl_anomalies = len(labelled_data)

        #PERCENTAGE OF ARTICLES CORRECTLY CLASSIFIED INTO INDEXED TOPICS (INCLUDING ARTICLES W ONLY ONE ARTICLE IN INDEXED TOPIC)
        pct_indexed_topics_excl_anomalies = (len(labelled_data[labelled_data["index_match_status"]])*100)/length_df_excl_anomalies

        one_row = [filename, data_date, length_df_incl_anomalies, length_df_w_subindex, length_df_excl_anomalies, pct_indexed_topics_incl_anomalies, pct_sub_indexed_topics_incl_anomalies, pct_indexed_topics_excl_anomalies]
        model_validation_lst.append(one_row)

    model_validation_df = pd.DataFrame(model_validation_lst, columns = ["Validation_File", "Date", "Num_Articles_Incl_Anom", "Num_Articles_W_SubIndex", "Num_Articles_Excl_Anom", "Pct_Correct_Index", "Pct_Correct_Index_and_Sub_Index", "Pct_Correct_Index_Excl_Anom"])
    
    
    return model_validation_df

def validate_multiple_models(params_table, data_directory):
    """
    Tests performance of all input models & parameters in params_table on all files in data_directory

    Args:
        1. params_table: Dataframe whose rows correspond to model names and parameters to validate on test data in data_directory
            MUST contain the following column names in this order (however for some columns, cells itself may be left empty to be set to default, as described below):
            - "Model": Either "Top2Vec" or "BERTopic"
            - "Text_Column" (optional): "Headline", "Summary" (not recommended) or "Both". Defaults to "Headline"
            - "Embedding_Model" (optional): Defaults to "all-MiniLM-L6-v2"
            - "HDBSCAN_args" (optional)
            - "UMAP_args"
            - "vectorizer_args"
        2. data_directory (str): Path to directory containing all test data in xlsx format under "Sheet1". Should contain columns: 'Date', 'Headline', 'Index', 'Sub-Index'. 'Summary' column optional.

    Returns:
        1. Dataframe containing columns as follows:
            - "Model"
            - "Text_Column"
            - "Embedding_Model"
            - "HDBSCAN_args"
            - "UMAP_args"
            - "vectorizer_args" 
            - Pct_Correct_Index: Mean percentage of articles whose predicted index match true index
            - Pct_Correct_Index_and_Sub_Index: Mean percentage of articles whose predicted index match true index and predicted sub-index match true sub-index (amongst articles where sub-index is non-empty)
            - Pct_Correct_Index_Excl_Anom: Mean percentage of articles whose predicted index match true index (amongst articles in test set with non-unique true Index)

    """
    params_table = params_table.fillna('')
    params_table_extended = params_table.apply(lambda x: pd.Series(((validate_model_with_all_files(data_directory, x.Model, x.Text_Column, x.Embedding_Model, x.HDBSCAN_args, x.UMAP_args, x.vectorizer_args)).mean()).tolist()[3:6], 
                                                index = ["Pct_Correct_Index", 
                                                        "Pct_Correct_Index_and_Sub_Index", 
                                                        "Pct_Correct_Topics_Excl_Anom"]),
                                       result_type='expand', axis=1)
    params_table = pd.concat([params_table, params_table_extended], axis = 1)
    params_table["Text_Column"] = params_table["Text_Column"].apply(lambda x: "Headline" if x=="" else x)
    params_table["Embedding_Model"] = params_table["Embedding_Model"].apply(lambda x: "all-MiniLM-L6-v2" if x=="" else x)
    params_table["HDBSCAN_args"] = params_table.apply(lambda x: 
                                                        "{'min_cluster_size': 2,'metric': 'euclidean', 'cluster_selection_method': 'leaf', 'min_samples': 1, 'allow_single_cluster': True}" if (x["Model"] == "Top2Vec" and x["HDBSCAN_args"]=="")
                                                            else "{'min_cluster_size': 3,'metric': 'euclidean', 'cluster_selection_method': 'leaf', 'min_samples': 1, 'allow_single_cluster': True, 'prediction_data': True}" if (x["Model"] == "BERTopic" and x["HDBSCAN_args"] == "")
                                                            else x["HDBSCAN_args"], axis = 1)
    params_table["UMAP_args"] = params_table["UMAP_args"].apply(lambda x: "{'n_neighbors': 10,'n_components': 3, 'metric': 'cosine'}" if x=="" else x)
    params_table["vectorizer_args"] = params_table.apply(lambda x: "{'ngram_range': (1, 1), 'stop_words': 'english'}" if (x["vectorizer_args"]=="" and x["Model"]=="BERTopic")
                                                            else x["vectorizer_args"], axis = 1)
    return params_table
