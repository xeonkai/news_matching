import pandas as pd
import streamlit as st
from functions.taxonomy_reader import read_taxonomy, convert_chain_to_list
import utils.utils as utils
import utils.design_format as format
from st_aggrid import AgGrid, GridUpdateMode, ColumnsAutoSizeMode, JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder
import time
import json
import ast


# Function to process dataframe - extract top theme, index and subindex
@st.cache_data
def process_table(df):
    df["suggested_labels"] = df["Predicted_Theme_Chains"].apply(
        lambda x: [chain for chain in list(x.keys())[:5]]
    )

    df["suggested_themes"] = df["Predicted_Theme_Chains"].apply(
        lambda x: [convert_chain_to_list(chain)[0] for chain in list(x.keys())[:5]]
    )
    df["suggested_themes"] = df["suggested_themes"].apply(
        lambda x: list(dict.fromkeys(x))
    )
    df["theme"] = df["Predicted_Theme_Chains"].apply(
        lambda x: [convert_chain_to_list(chain)[0] for chain in list(x.keys())][0]
    )
    df["theme_prob"] = df["Predicted_Theme_Chains"].apply(lambda x: list(x.values())[0])

    df["suggested_indexes"] = df["Predicted_Theme_Chains"].apply(
        lambda x: [convert_chain_to_list(chain)[1] for chain in list(x.keys())[:5]]
    )
    df["suggested_indexes"] = df["suggested_indexes"].apply(
        lambda x: list(dict.fromkeys(x))
    )
    df["index"] = df["Predicted_Theme_Chains"].apply(
        lambda x: [convert_chain_to_list(chain)[1] for chain in list(x.keys())][0]
    )
    df["index_prob"] = df["Predicted_Theme_Chains"].apply(lambda x: list(x.values())[0])

    df["suggested_subindexes"] = df["Predicted_Theme_Chains"].apply(
        lambda x: [convert_chain_to_list(chain)[2] for chain in list(x.keys())[:5]]
    )
    df["suggested_subindexes"] = df["suggested_subindexes"].apply(
        lambda x: list(dict.fromkeys(x))
    )
    df["subindex"] = df["Predicted_Theme_Chains"].apply(
        lambda x: [convert_chain_to_list(chain)[2] for chain in list(x.keys())][0]
    )
    df["subindex_prob"] = df["Predicted_Theme_Chains"].apply(
        lambda x: list(x.values())[0]
    )

    # st.write(df)
    return df


# Function to generate json file for URL and top suggested themes, indexes and subindexes
def generate_json(df):
    output = {}
    df.apply(
        lambda x: output.update(
            {
                x["Link"]: {
                    "themes": x["suggested_themes"],
                    "indexes": x["suggested_indexes"],
                    "subindexes": x["suggested_subindexes"],
                }
            }
        ),
        axis=1,
    )
    # save to json

    return output


# Function to slice table based on top theme and sort by top_index
@st.cache_data
def slice_table(df):
    top_themes = get_top_themes(df)
    df_collection = {}
    for theme in top_themes:
        df_slice = df[df["theme"] == theme]
        df_slice = df_slice.sort_values(
            by=["index", "subindex", "index_prob"], ascending=[True, True, False]
        )
        df_collection[theme] = df_slice
    return df_collection


# Function to get top themes based on facebook interactions
@st.cache_data
def get_top_themes(df):
    df_sum = df.groupby(["theme"]).agg({"facebook_interactions": "sum"})

    df_sum = df_sum.sort_values(
        by=["facebook_interactions"], ascending=False
    ).reset_index()

    top_themes = df_sum["theme"].unique()

    return top_themes


# Function to display statistics
def display_stats(df, title=True, show_themes=True, show_theme_count=True):
    if title:
        st.subheader("Overall Summary Statistics")

    n_articles = df.shape[0]
    n_themes = len(df["theme"].unique())
    n_index = len(df["index"].unique())
    n_subindex = len(df["subindex"].unique())
    n_fb_interactions = df["facebook_interactions"].sum()

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(label="Total Articles", value=n_articles)

    with col2:
        st.metric(label="Total Facebook Interactions", value=n_fb_interactions)

    if show_theme_count:
        with col3:
            st.metric(label="Total themes", value=n_themes)

        with col4:
            st.metric(label="Total Index", value=n_index)

        with col5:
            st.metric(label="Total Subindex", value=n_subindex)

    else:
        with col3:
            st.metric(label="Total Index", value=n_index)

        with col4:
            st.metric(label="Total Subindex", value=n_subindex)

    if show_themes:
        st.write("")
        themes = ", ".join(get_top_themes(df))
        st.markdown(f"Themes present: **{themes}**")


# Function to display aggrid by themes
@st.cache_resource(experimental_allow_widgets=True, show_spinner=False)
def display_aggrid_by_theme(df_collection, current_theme_index):
    current_theme = list(df_collection.keys())[current_theme_index]
    n_themes = len(df_collection.keys())
    df = df_collection[current_theme]
    st.subheader(f"{current_theme}")

    # Display stats
    display_stats(df, title=False, show_themes=False, show_theme_count=False)

    m1, m2 = st.columns([2, 1])
    with m1:
        theme_jumper(df_collection)
    with m2:
        table_pagination_menu()

    # load grid responses from cache
    if utils.check_session_state_key("grid_responses"):
        grid_responses = utils.get_cached_object("grid_responses")
    else:
        grid_responses = {}
        utils.cache_object(grid_responses, "grid_responses")

    with st.form("aggrid_form"):
        load_state = False
        selected_rows = []
        if current_theme in grid_responses:
            load_state = True
            df = grid_responses[current_theme]["data"]
            selected_rows = grid_responses[current_theme]["selected_rows"]

        current_response = display_aggrid(df, load_state, selected_rows)

        if st.form_submit_button("Confirm"):
            valid_submission = validate_current_response(current_response)

            grid_responses_validation = {}
            grid_responses_validation[current_theme] = valid_submission
            utils.cache_object(grid_responses_validation, "grid_responses_validation")

            grid_responses[current_theme] = current_response
            utils.cache_object(grid_responses, "grid_responses")

            if valid_submission:

                st.success(
                    f"Article Labels Confirmed for {current_theme}! You may overwrite by pressing 'Confirm' again!",
                    icon="✅",
                )
                # st.experimental_rerun() 

            else:
                st.warning(
                    f"Please enter blank fields that require new inputs!",
                    icon="⚠️",
                )
            st.experimental_rerun() 
        elif (
            load_state
            and utils.check_session_state_key("grid_responses_validation")
            and current_theme in utils.get_cached_object("grid_responses_validation")
            and utils.get_cached_object("grid_responses_validation")[current_theme]
        ):
            st.success(
                f"Article Labels Confirmed for {current_theme}! You may overwrite by pressing 'Confirm' again!",
                icon="✅",
            )
        elif (
            load_state
            and utils.check_session_state_key("grid_responses_validation")
            and current_theme in utils.get_cached_object("grid_responses_validation")
            and not utils.get_cached_object("grid_responses_validation")[current_theme]
        ):
            st.warning(
                f"Please enter blank fields that require new inputs!",
                icon="⚠️",
            )

    # Buttons
    nav_buttons(current_theme_index, n_themes)

    return


# Function to display navigation buttons
def nav_buttons(current_theme_index, n_themes):
    b1, b2, b3, b4, b5 = st.columns([7, 1, 1, 1, 7])
    with b2:
        if st.button("Prev"):
            current_theme_index = max(current_theme_index - 1, 0)
            utils.cache_object(current_theme_index, "current_theme_index")
            st.experimental_rerun()
    with b3:
        format.align_text(f"{current_theme_index + 1} of {n_themes}", "left")

    with b4:
        if st.button("Next"):
            current_theme_index = min(current_theme_index + 1, n_themes - 1)
            utils.cache_object(current_theme_index, "current_theme_index")
            st.experimental_rerun()


# Theme jumper dropdown
def theme_jumper(df_collection):
    current_theme_index = utils.get_cached_object("current_theme_index")
    current_theme = list(df_collection.keys())[current_theme_index]
    theme_list = list(df_collection.keys())
    theme_index = theme_list.index(current_theme)

    with st.form("theme_jumper_form"):
        theme_index = st.selectbox(
            "Jump to Theme",
            theme_list,
            index=theme_index,
        )

        if st.form_submit_button("Jump"):
            current_theme_index = theme_list.index(theme_index)
            utils.cache_object(current_theme_index, "current_theme_index")
            st.experimental_rerun()


# Table interface menu
def table_pagination_menu():
    # TODO: Menu to toggle pagination
    if utils.check_session_state_key("n_articles_per_page"):
        n_articles_per_page = utils.get_cached_object("n_articles_per_page")
    else:
        n_articles_per_page = 20

    with st.form("pagination_form"):
        n_articles_per_page = st.number_input(
            "Number of Articles per Page",
            min_value=1,
            max_value=999,
            value=n_articles_per_page,
            step=1,
        )

        if st.form_submit_button("Submit"):
            utils.cache_object(n_articles_per_page, "n_articles_per_page")
            st.experimental_rerun()
    return


# Function to validate current response of aggrid table
def validate_current_response(current_response):
    # st.write(current_response["selected_rows"])
    incomplete_theme = []
    incomplete_index = []
    incomplete_subindex = []
    for row in current_response["selected_rows"]:
        if row["theme"] == "-Enter New Theme" and row["new theme"] == "":
            incomplete_theme.append(row["_selectedRowNodeInfo"]["nodeRowIndex"])
        if row["index"] == "-Enter New Index" and row["new index"] == "":
            incomplete_index.append(row["_selectedRowNodeInfo"]["nodeRowIndex"])
        if row["subindex"] == "-Enter New Subindex" and row["new subindex"] == "":
            incomplete_subindex.append(row["_selectedRowNodeInfo"]["nodeRowIndex"])

    if len(incomplete_theme) or len(incomplete_index) or len(incomplete_subindex):
        return False
    else:
        return True


# Function to display aggrid table
@st.cache_resource(experimental_allow_widgets=True, show_spinner=False)
def display_aggrid(df, load_state, selected_rows):
    # Initialising columns for new input
    if not load_state:
        df["new theme"] = ""
        df["new index"] = ""
        df["new subindex"] = ""
        cols = df.columns.tolist()
        cols = cols[1:] + cols[:1]
        df = df[cols]
    else:
        df["suggested_themes"] = df["suggested_themes"].apply(
            lambda x: ast.literal_eval(x) if type(x) == str else x
        )
        df["suggested_indexes"] = df["suggested_indexes"].apply(
            lambda x: ast.literal_eval(x) if type(x) == str else x
        )
        df["suggested_subindexes"] = df["suggested_subindexes"].apply(
            lambda x: ast.literal_eval(x) if type(x) == str else x
        )

    # loading taxonomy from cache
    if utils.check_session_state_key("taxonomy"):
        taxonomy = utils.get_cached_object("taxonomy")
    else:
        taxonomy = read_taxonomy()
        utils.cache_object(taxonomy, "taxonomy")

    columns_to_show = [
        "headline",
        "facebook_interactions",
        "domain",
        "theme",
        "new theme",
        "index",
        "new index",
        "subindex",
        "new subindex",
        # "suggested_labels",
        # "suggested_themes",
        # "suggested_indexes",
        # "suggested_subindexes",
    ]

    gb = GridOptionsBuilder.from_dataframe(df)

    gb.configure_default_column(
        groupable=False,
        value=True,
        enableRowGroup=False,
        aggFunc="count",
        filterable=False,
        sortable=False,
    )

    # Configure individual columns
    gb.configure_column("Headline", headerCheckboxSelection=True, width=1200)

    tooltipjs = JsCode(
        """ function(params) { return '<span title="' + params.value + '">'+params.value+'</span>';  }; """
    )  # if using with cellRenderer

    gb.configure_column("suggested_labels", width=900, cellRenderer=tooltipjs)

    themejs = JsCode(
        """
        function(params) {
        const predictedThemes = params.data.suggested_themes;
        predictedThemes.indexOf("-Enter New Theme") === -1 ? predictedThemes.push("-Enter New Theme") : null;
            return {
                values: predictedThemes,
                popupPosition: "under",
                cellHeight: 30,
            }
        }
        
        """
    )

    gb.configure_column(
        "theme",
        editable=True,
        cellEditor="agRichSelectCellEditor",
        cellEditorParams=themejs,
    )

    indexesjs = JsCode(
        """
        function(params) {
        const predictedIndexes = params.data.suggested_indexes;
        predictedIndexes.indexOf("-Enter New Index") === -1 ? predictedIndexes.push("-Enter New Index") : null;
            return {
                values: predictedIndexes,
                popupPosition: "under",
                cellHeight: 30,
            }
        }
        
        """
    )

    gb.configure_column(
        "index",
        editable=True,
        cellEditor="agRichSelectCellEditor",
        cellEditorParams=indexesjs,
    )

    subindexesjs = JsCode(
        """
        function(params) {
        const predictedSubIndexes = params.data.suggested_subindexes;
        predictedSubIndexes.indexOf("-Enter New Subindex") === -1 ? predictedSubIndexes.push("-Enter New Subindex") : null;
            return {
                values: predictedSubIndexes,
                popupPosition: "under",
                cellHeight: 30,
            }
        }
        """
    )

    gb.configure_column(
        "subindex",
        editable=True,
        cellEditor="agRichSelectCellEditor",
        cellEditorParams=subindexesjs,
    )

    gb.configure_column("new theme", editable=True)
    gb.configure_column("new index", editable=True)
    gb.configure_column("new subindex", editable=True)

    # gb.configure_grid_options(domLayout="normal")
    # st.write(selected_rows)
    selected_rows_id = [
        row["_selectedRowNodeInfo"]["nodeRowIndex"] for row in selected_rows
    ]
    gb.configure_selection(
        selection_mode="multiple",
        use_checkbox=True,
        pre_selected_rows=selected_rows_id,
    )

    # Pagination

    if utils.check_session_state_key("n_articles_per_page"):
        n_articles_per_page = utils.get_cached_object("n_articles_per_page")
    else:
        n_articles_per_page = 20
        utils.cache_object(n_articles_per_page, "n_articles_per_page")

    gb.configure_pagination(
        paginationAutoPageSize=False,
        paginationPageSize=n_articles_per_page,
    )

    gridOptions = gb.build()

    columns_to_hide = [i for i in df.columns if i not in columns_to_show]
    column_defs = gridOptions["columnDefs"]
    for col in column_defs:
        if col["headerName"] in columns_to_hide:
            col["hide"] = True

    grid_response = AgGrid(
        df,
        gridOptions=gridOptions,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        columns_auto_size_mode=ColumnsAutoSizeMode.FIT_ALL_COLUMNS_TO_VIEW,
        allow_unsafe_jscode=True,
    )

    return grid_response
