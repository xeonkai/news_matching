
import streamlit as st
import pandas as pd
import time
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
        "No data selected yet! Please select the required data from the Data Explorer page!",
        wait=-1,
        type_="warning",
        icon="⚠️",
    )
