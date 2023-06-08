import streamlit as st
import pandas as pd
import time
import json
from sentence_transformers import SentenceTransformer
from setfit import SetFitModel
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import datetime
import os
from pathlib import Path

RAW_DATA_DIR = Path("data", "raw")
PROCESSED_DATA_DIR = Path("data", "processed")


def csv_file_uploader():
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")

    if uploaded_file is not None:
        st.write("You selected `%s`" % uploaded_file.name)
        if st.button("Upload CSV"):
            # Storing CSV data to ST cache
            cache_object(pd.read_csv(uploaded_file), "csv_file")
            customDisppearingMsg(
                "CSV file uploaded successfully!", wait=3, type_="success", icon=None
            )
            customDisppearingMsg(
                "You may now navigate to the other tabs for analysis!",
                wait=-1,
                type_="info",
                icon=None,
            )


def json_file_uploader():
    uploaded_file = st.file_uploader("Upload JSON file", type="json")

    if uploaded_file is not None:
        st.write("You selected `%s`" % uploaded_file.name)
        if st.button("Upload JSON"):
            # Storing JSON data to ST cache
            cache_object(pd.read_csv(uploaded_file), "csv_file")
            customDisppearingMsg(
                "JSON file uploaded successfully!", wait=3, type_="success", icon=None
            )
            customDisppearingMsg(
                "You may now navigate to the other tabs for analysis!",
                wait=-1,
                type_="info",
                icon=None,
            )


def convert_df(df):
    return df.to_csv(index=False).encode("utf-8")


def cache_object(object, key):
    """Function to cache objects in Streamlit Session State
    Args:
        object (object): Object to be cached
        key (str): Key to be used to cache the object

    Returns:
        object: Cached object
    """

    if key not in st.session_state:
        # print("INFO: Key does not exist in session state. Creating new key.")
        st.session_state[key] = object
    else:
        # print("INFO: Key exists in session state. Overwriting key.")
        st.session_state[key] = object
    return st.session_state[key]


def get_cached_object(key):
    """Function to get cached objects from Streamlit Session State
    Args:
        key (str): Key to be used to get the object

    Returns:
        object: Cached object
    """

    if key in st.session_state:
        # print("INFO: Key exists in session state. Returning cached object.")
        return st.session_state[key]
    else:
        # print("INFO: Key does not exist in session state. Returning None.")
        return None


def check_session_state_key(key):
    """Function to check if a key exists in Streamlit Session State
    Args:
        key (str): Key to be checked

    Returns:
        bool: True if key exists, False otherwise
    """

    return key in st.session_state


def customDisppearingMsg(msg, wait=3, type_="success", icon=None):
    """Function to display a custom disappearing message
    Args:
        msg (str): Message to be displayed
        wait (int, optional): Time to wait before disappearing. Defaults to 3.
        type_ (str, optional): Type of message. Defaults to 'success'.
        icon (str, optional): Icon to be displayed. Defaults to None.

    Returns:
        object: Placeholder object
    """

    placeholder = st.empty()
    if type_ == "success":
        placeholder.success(msg, icon=icon)
    elif type_ == "warning":
        placeholder.warning(msg, icon=icon)
    elif type_ == "info":
        placeholder.info(msg, icon=icon)
    if wait > 0:
        time.sleep(wait)
        placeholder.empty()
    return placeholder


def no_file_uploaded():
    """Function to display a message when no file is uploaded
    Args:
        None

    Returns:
        None
    """

    customDisppearingMsg(
        "No file uploaded yet! Please upload your CSV file in the 'Home' page!",
        wait=-1,
        type_="warning",
        icon="⚠️",
    )


def load_embedding_model() -> SentenceTransformer:
    model = SentenceTransformer(
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        cache_folder="cached_models",
    )
    return model


def load_classification_model(model_path=None) -> SetFitModel:
    data_folder = Path("trained_models")
    data_folder_date_sorted = sorted(data_folder.iterdir(), key=os.path.getmtime)
    latest_model_path = str(data_folder_date_sorted[-1])

    if model_path is None:
        model_path = latest_model_path

    model = SetFitModel.from_pretrained(model_path)
    return model


class FileHandler:
    def __init__(self, raw_data_dir, processed_data_dir):
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir

        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def preprocess_daily_scan(file, source: str = "") -> pd.DataFrame:
        df = (
            pd.read_csv(
                file,
                usecols=[
                    "Published",
                    "Headline",
                    "Summary",
                    "Link",
                    "Domain",
                    "Facebook Interactions",
                ],
                dtype={
                    "Headline": "string",
                    "Summary": "string",
                    "Link": "string",
                    "Domain": "string",
                    "Facebook Interactions": "int",
                },
                parse_dates=["Published"],
            )
            .assign(
                timestamp=lambda df: df["Published"].astype("int64") // 10**9,
                source=source,
            )
            .rename(lambda col_name: col_name.lower().replace(" ", "_"), axis="columns")
        )
        return df

    @staticmethod
    def label_df(df: pd.DataFrame, model: SetFitModel, column: str) -> pd.DataFrame:
        y_score = model.predict_proba(df[column])

        label_order = np.argsort(y_score, axis=1, kind="stable").numpy()[:, ::-1]
        label_scores_df = pd.DataFrame(y_score, columns=model.model_head.classes_)

        sorted_label_list = []
        sorted_scores_list = []
        for (idx, row) in label_scores_df.iterrows():
            sorted_label = row.iloc[label_order[idx]]
            sorted_label_list.append(sorted_label.index.to_list())
            sorted_scores_list.append(sorted_label.to_list())

        labelled_df = df.assign(
            predicted_indexes=sorted_label_list, prediction_prob=sorted_scores_list
        )

        labelled_df = df.assign(suggested_labels=sorted_label_list, suggested_labels_score=sorted_scores_list)
        return labelled_df

    @staticmethod
    def embed_df(df: pd.DataFrame, model: SentenceTransformer, column: str) -> pa.Table:
        embeddings = model.encode(df[column])
        # Pyarrow table keeps schema
        embedded_table = pa.Table.from_pandas(df, preserve_index=False).append_column(
            "vector",
            pa.FixedSizeListArray.from_arrays(
                embeddings.ravel(),
                list_size=embeddings.shape[-1],
            ),
        )
        return embedded_table

    def write_csv(self, file):
        filepath = self.raw_data_dir / file.name
        filepath.write_bytes(file.getbuffer())
        return filepath

    def write_processed_parquet(self, file, embedding_model, classification_model):
        processed_table = (
            self.preprocess_daily_scan(file, source=file.name)
            .pipe(self.label_df, model=classification_model, column="headline")
            .pipe(self.embed_df, model=embedding_model, column="headline")
        )
        # Save processed data
        filepath = self.processed_data_dir / file.name.replace(".csv", ".parquet")
        pq.write_table(
            processed_table,
            filepath,
        )
        return filepath

    def list_csv_filenames(self):
        return [file.name for file in self.raw_data_dir.iterdir()]

    def list_csv_files(self):
        return list(self.raw_data_dir.iterdir())

    def list_csv_files_df(self):
        raw_files_info = [
            {
                "filename": file.name,
                "modified": datetime.datetime.fromtimestamp(file.stat().st_mtime),
                # "filesize": f"{file.stat().st_size / 1000 / 1000:.2f} MB",
            }
            for file in self.raw_data_dir.iterdir()
        ]
        return pd.DataFrame(raw_files_info)

    def remove_files(self, filenames):
        for filename in filenames:
            (self.raw_data_dir / filename).unlink(missing_ok=True)
            (self.processed_data_dir / filename.replace(".csv", ".parquet")).unlink(
                missing_ok=True
            )
        return [self.raw_data_dir / filename for filename in filenames]

    def download_csv_files(self):
        # Convert folder to zip, return zip file
        raise NotImplementedError

    def __str__(self):
        return f"{[file.name for file in self.raw_data_dir.iterdir()]}"

    def __len__(self):
        return len(self.raw_data_dir.iterdir())
