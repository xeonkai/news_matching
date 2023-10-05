import pandas as pd
import streamlit as st
import numpy as np
from utils import core

st.set_page_config(
    page_title="Upload Indexed Data", page_icon="📰", layout="wide"
)

st.title("🖥️ Upload Indexed Data")
st.markdown("""---""")
st.markdown(
    """
    Upload an excel file of the indexed articles to update the main database.
    """
)
st.markdown("""---""")

file_handler = core.FileHandler(core.DATA_DIR)

st.subheader("Uploaded files")
files_table = st.empty()
files_table.write(file_handler.list_csv_files_df())

io_mode = st.selectbox("Upload or Delete files", ("Upload", "Delete"))

if io_mode == "Upload":
    uploaded_files = st.file_uploader(
        "Upload excel file here:", type=["xlsx"], accept_multiple_files=True
    )
    if uploaded_files:
        dup_filenames = [
            file.name
            for file in uploaded_files
            if file.name in file_handler.list_csv_filenames()
        ]
        new_files = [
            file for file in uploaded_files if (file.name not in dup_filenames)
        ]
        # Check for dup files, overwrite if selected
        if len(dup_filenames) > 0:
            files_to_overwrite = st.multiselect(
                "Duplicate files detected, choose which to overwrite",
                dup_filenames,
            )
            new_files.extend(
                [file for file in uploaded_files if file.name in files_to_overwrite]
            )
        save_btn = st.button("Save")
        if save_btn:
            # save file name for future output name
            num_uploaded_files = 0
            progress_text = "File upload {num_uploaded_files} / {num_new_files} in progress. Please wait."
            progress_bar = st.progress(
                0,
                text=progress_text.format(
                    num_uploaded_files=num_uploaded_files,
                    num_new_files=len(new_files),
                ),
            )

            for daily_news in new_files:
                try:
                    # Processed file first for schema validation
                    file_handler.write_labelled_articles_db(daily_news)
                    # Raw file if processing ok
                    file_handler.write_csv(daily_news)

                    num_uploaded_files += 1
                    progress_bar.progress(
                        num_uploaded_files / len(new_files),
                        text=progress_text.format(
                            num_uploaded_files=num_uploaded_files,
                            num_new_files=len(new_files),
                        ),
                    )
                except Exception as err:
                    st.warning(
                        f"Failed to write{daily_news.name}, check if file is valid"
                    )
                    st.error(err)
            if num_uploaded_files == len(new_files):
                st.success(f"{num_uploaded_files} files uploaded successfully!")

elif io_mode == "Delete":
    files_to_delete = st.multiselect(
        "Files to Delete", file_handler.list_csv_filenames()
    )
    delete_btn = st.button("Delete")

    if delete_btn:
        file_handler.remove_files(files_to_delete)

# Update table on each action
files_table.write(file_handler.list_csv_files_df())

if file_handler.list_csv_files_df().empty:
    st.warning("There is no data. Please upload a csv file before continuing.")


