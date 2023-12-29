import streamlit as st
from utils import core
from st_pages import add_page_title

add_page_title(layout="wide")

# st.set_page_config(page_title="Topic Discovery Tool", page_icon="📰", layout="wide")
# st.title("🖥️ Topic Discovery Tool")

def run():
    st.markdown("""---""")
    st.subheader("Welcome!")
    st.markdown(
        """
        - Upload a CSV file of the WEEKLY NEWS FILE downloaded from the content aggregator. \n
        - Please ensure that the name of the uploaded file follows the format below: \n
            `<filename>-<MM_DD_YY-HH_MM>.csv`. \n
        - If you have done the article indexing on your internet laptop, additionally on this page please upload INDEXED DATA FILE\n
        """
    )
    st.markdown("""---""")
    file_handler = core.FileHandler(core.DATA_DIR)

    st.subheader("Uploaded files")
    files_table = st.empty()

    # TODO: Maybe add download mode
    io_mode = st.selectbox("Action type", ("Upload Weekly News Scan", "Upload Indexed Data", "Delete"))

    if io_mode == "Upload Weekly News Scan":
        st.markdown(
            """
            - Upload a csv file of the news scan to add them to the database.
            """
        )
        uploaded_files = st.file_uploader(
            "Upload new weekly news scan here:", type=["csv"], accept_multiple_files=True
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

                for news in new_files:
                    try:
                        # Processed file first for schema validation
                        file_handler.write_daily_scan(news)
                        # Raw file if processing ok
                        file_handler.write_csv(news)

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
                            f"Failed to write{news.name}, check if file is valid"
                        )
                        st.error(err)

                if num_uploaded_files == len(new_files):
                    st.success(f"{num_uploaded_files} files uploaded successfully!")
    
    elif io_mode == "Upload Indexed Data":
        st.markdown(
            """
            - Upload a csv file of the indexed articles to update the database.
            """
        )
        uploaded_index_files = st.file_uploader(
            "Upload indexed data here:", type=["csv"], accept_multiple_files=True
        )
        if uploaded_index_files:
            save_index_btn = st.button("Save")
            if save_index_btn:
                # save file name for future output name
                num_uploaded_files = 0
                progress_text = "Index file upload {num_uploaded_files} / {num_new_files} in progress. Please wait."
                progress_bar = st.progress(
                    0,
                    text=progress_text.format(
                        num_uploaded_files=uploaded_index_files,
                        num_new_files=len(uploaded_index_files),
                    ),
                )
                for indexed_file in uploaded_index_files:
                    try:
                        # Processed file first for schema validation
                        file_handler.write_labelled_articles(indexed_file)

                        num_uploaded_files += 1
                        progress_bar.progress(
                            num_uploaded_files / len(uploaded_index_files),
                            text=progress_text.format(
                                num_uploaded_files=num_uploaded_files,
                                num_new_files=len(uploaded_index_files),
                            ),
                        )
                    except Exception as err:
                        st.warning(
                            f"Failed to write{indexed_file.name}, check if file is valid"
                        )
                        st.error(err)
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

    st.markdown("""---""")
    st.cache_data.clear()


if __name__ == "__main__":
    run()
