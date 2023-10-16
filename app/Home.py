import streamlit as st
from utils import core
import os
from st_pages import add_page_title

add_page_title(layout="wide")

# st.set_page_config(page_title="Topic Discovery Tool", page_icon="📰", layout="wide")
# st.title("🖥️ Topic Discovery Tool")

st.markdown("""---""")
st.subheader("Welcome!")
st.markdown(
    """
    - Upload a CSV file of the weekly news file downloaded from the content aggregator. \n
    - Please ensure that the name of the uploaded file follows the format below:\n
        `test_-_for_indexing-facebook_posts-<MM_DD_YY-HH_MM -  D MMM> weekly`, where `MMM` are the first 3 letters of the month.
    """
)
st.markdown("""---""")


def run():
    daily_file_handler = core.FileHandler(core.DATA_DIR)
    weekly_file_handler = core.WeeklyFileHandler(core.DATA_DIR)

    if weekly_file_handler.query().empty:
        os.rename('traction-analytics/may_june_data_filtered.xlsx', 'data/weekly/may_june_data_filtered.xlsx')
        weekly_file_handler.write_may_june_db()

    st.subheader("Uploaded files")
    files_table = st.empty()

    # TODO: Maybe add download mode
    io_mode = st.selectbox("Upload or Delete files", ("Upload", "Delete"))

    if io_mode == "Upload":
        uploaded_files = st.file_uploader(
            "Upload new data here:", type=["xlsx"], accept_multiple_files=True
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

                for news in new_files:
                    try:
                        # Processed file first for schema validation
                        daily_file_handler.write_db(news)
                        # Raw file if processing ok
                        daily_file_handler.write_csv(news)
                        # Processed file first for schema validation
                        weekly_file_handler.write_db(news)
                        # Raw file if processing ok
                        weekly_file_handler.write_csv(news)

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

    elif io_mode == "Delete":
        files_to_delete = st.multiselect(
            "Files to Delete", daily_file_handler.list_csv_filenames()
        )
        delete_btn = st.button("Delete")

        if delete_btn:
            daily_file_handler.remove_files(files_to_delete)
            weekly_file_handler.remove_files(files_to_delete)

    # Update table on each action
    files_table.write(daily_file_handler.list_csv_files_df())

    if daily_file_handler.list_csv_files_df().empty:
        st.warning("There is no data. Please upload a csv file before continuing.")

    st.markdown("""---""")


if __name__ == "__main__":
    run()
