import pandas as pd
import streamlit as st
import numpy as np
from utils import core
from st_pages import add_page_title

add_page_title(layout="wide")

# st.set_page_config(
#     page_title="Upload Data to Central Tool", page_icon="📰", layout="wide"
# )
# st.title("🖥️ Upload Data to Central Tool")
st.markdown(
    """
    - If you have done the article indexing on your internet laptop, please upload **2 FILES** on this page:
        - Indexed Data (from the previous section) 
        - Weekly News File (from the content aggregator)
    - Please disregard this page if the article indexing was done on the central tool.
    
    """
)
st.markdown("---")
st.markdown("## 1. Upload Indexed Data")
st.markdown(
    """
    - Upload an excel file of the indexed articles to update the main database.\n
    - Please ensure that the name of the uploaded file follows the format below:\n
        `Labelled_Articles_YYYY_MM_DD.xlsx`
    """
)

daily_file_handler = core.FileHandler(core.DATA_DIR)

st.subheader("Uploaded files")
files_table = st.empty()
files_table.write(daily_file_handler.list_csv_files_df())

io_mode_indexed = st.selectbox("Upload or Delete files", ("Upload", "Delete"), key="Indexed_Articles_io")

if io_mode_indexed == "Upload":
    uploaded_files = st.file_uploader(
        "Upload excel file here:", type=["xlsx"], accept_multiple_files=True, key="Indexed_Articles_files"
    )
    if uploaded_files:
        dup_filenames = [
            file.name
            for file in uploaded_files
            if file.name in daily_file_handler.list_csv_filenames()
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
                    daily_file_handler.write_labelled_articles_db(daily_news)
                    # Raw file if processing ok
                    daily_file_handler.write_csv(daily_news)

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

elif io_mode_indexed == "Delete":
    files_to_delete = st.multiselect(
        "Files to Delete", daily_file_handler.list_csv_filenames()
    )
    delete_btn = st.button("Delete")

    if delete_btn:
        daily_file_handler.remove_files(files_to_delete)

# Update table on each action
files_table.write(daily_file_handler.list_csv_files_df())

if daily_file_handler.list_csv_files_df().empty:
    st.warning("There is no data. Please upload an excel file before continuing.")

st.markdown("---")
st.markdown("## 2. Upload Weekly News File")
st.markdown(
    """
    - Upload weekly csv file from content aggregator.
    - Please ensure that the name of the uploaded file follows the format below:\n
        `test_-_for_indexing-facebook_posts-<MM_DD_YY-HH_MM>.csv`.
    """
)

weekly_file_handler = core.WeeklyFileHandler(core.DATA_DIR)

st.subheader("Uploaded files")
files_table = st.empty()
files_table.write(weekly_file_handler.list_csv_files_df())

io_mode_weekly = st.selectbox("Upload or Delete files", ("Upload", "Delete"), key="Weekly_Articles_io")

if io_mode_weekly == "Upload":
    uploaded_files = st.file_uploader(
        "Upload excel file here:", type=["csv"], accept_multiple_files=True, key="Weekly_Articles_files"
    )
    if uploaded_files:
        dup_filenames = [
            file.name
            for file in uploaded_files
            if file.name in weekly_file_handler.list_csv_filenames()
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

            for weekly_news in new_files:
                try:
                    # Processed file first for schema validation
                    weekly_file_handler.write_db(weekly_news)
                    # Raw file if processing ok
                    weekly_file_handler.write_csv(weekly_news)

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
                        f"Failed to write{weekly_news.name}, check if file is valid"
                    )
                    st.error(err)
            if num_uploaded_files == len(new_files):
                st.success(f"{num_uploaded_files} files uploaded successfully!")
            st.dataframe(weekly_file_handler.full_query())
            # test = file_handler.full_query()
            # test = test.loc[test['link'] == 'https://sg.news.yahoo.com/90-old-employee-mcdonalds-japan-225430598.html']
            # st.dataframe(test)


elif io_mode_weekly == "Delete":
    files_to_delete = st.multiselect(
        "Files to Delete", weekly_file_handler.list_csv_filenames()
    )
    delete_btn = st.button("Delete")

    if delete_btn:
        weekly_file_handler.remove_files(files_to_delete)

# Update table on each action
files_table.write(weekly_file_handler.list_csv_files_df())
# st.dataframe(file_handler.full_query())

if weekly_file_handler.list_csv_files_df().empty:
    st.warning("There is no data. Please upload a csv file before continuing.")

