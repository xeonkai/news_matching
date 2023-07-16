import ast
import copy

import streamlit as st
from functions.taxonomy_reader import convert_chain_to_list
from st_aggrid import AgGrid, ColumnsAutoSizeMode, GridUpdateMode, JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder

# Function to process dataframe - extract top index, index and subindex


@st.cache_data
def process_table(df, taxonomy):
    """Function to process dataframe - extract top index, index and subindex

    Args:
        df (pandas.core.frame.DataFrame): Dataframe to be processed

    Returns:
        pandas.core.frame.DataFrame: Processed dataframe
    """

    k = 5

    df["Predicted_theme_and_index"] = df["suggested_labels"].apply(lambda x: x[0])

    df["suggested_labels"] = df["suggested_labels"].apply(lambda x: x[:k])

    df["suggested_themes"] = df["suggested_labels"].apply(
        lambda chain_list: [convert_chain_to_list(chain)[0] for chain in chain_list][:k]
    )
    df["theme_ref"] = df["suggested_themes"].str[0]

    df["theme"] = ""

    df["theme_prob"] = df["suggested_labels_score"].str[0]

    df["suggested_indexes"] = df["suggested_labels"].apply(
        lambda chain_list: [convert_chain_to_list(chain)[1] for chain in chain_list][:k]
    )
    df["index_ref"] = df["suggested_indexes"].str[0]

    df["index"] = ""

    df["index_prob"] = df["suggested_labels_score"].str[0]

    df["taxonomy"] = df["suggested_labels"].apply(lambda x: taxonomy)

    df["subindex"] = ""

    return df


@st.cache_data
def slice_table(df):
    """Function to slice dataframe

    Args:
        df (pandas.core.frame.DataFrame): Dataframe to be sliced

    Returns:
        dict: Sliced dataframe
    """

    df_theme_grouped = df.copy().groupby("theme_ref")

    top_themes = (
        df_theme_grouped["facebook_interactions"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )

    # top_themes = get_top_themes(df)
    df_collection = {}
    for theme in top_themes:
        # df_slice = df[df["theme_ref"] == theme]
        # df_slice = df_slice.sort_values(by=["facebook_interactions"], ascending=False)
        # df_collection[theme] = df_slice
        df_collection[theme] = df_theme_grouped.get_group(theme).sort_values(
            by=["facebook_interactions"], ascending=False
        )
    return df_collection


@st.cache_data
def get_top_themes(df):
    """Function to get top themes

    Args:
        df (pandas.core.frame.DataFrame): Dataframe

    Returns:
        numpy.ndarray: Top themes
    """

    df_sum = df.groupby(["theme_ref"]).agg({"facebook_interactions": "sum"})

    df_sum = df_sum.sort_values(
        by=["facebook_interactions"], ascending=False
    ).reset_index()

    top_themes = df_sum["theme_ref"].unique()

    return top_themes


def display_stats(df, title=True, show_themes=True, show_theme_count=True):
    """Function to display stats

    Args:
        df (pandas.core.frame.DataFrame): Dataframe
        title (bool, optional): Whether to display title. Defaults to True.
        show_themes (bool, optional): Whether to display themes. Defaults to True.
        show_theme_count (bool, optional): Whether to display theme count. Defaults to True.
    """

    if title:
        st.subheader("Overall Summary Statistics")

    n_articles = df.shape[0]
    n_theme = len(df["theme_ref"].unique())
    n_index = len(df["index_ref"].unique())
    n_fb_interactions = df["facebook_interactions"].sum()

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(label="Total Articles", value=n_articles)

    with col2:
        st.metric(label="Total Facebook Interactions", value=n_fb_interactions)

    if show_theme_count:
        # with col3:
        #     st.metric(label="Total indexes", value=n_indexes)

        with col3:
            st.metric(label="Total Themes", value=n_theme)

        with col4:
            st.metric(label="Total Indexes", value=n_index)

    else:
        with col3:
            st.metric(label="Total Indexes", value=n_index)

    if show_themes:
        st.write("")
        themes = ", ".join(get_top_themes(df))
        st.markdown(f"Themes present: **{themes}**")


# @st.cache_resource(experimental_allow_widgets=True, show_spinner=False)
def display_aggrid_by_theme(df_collection, current_theme_index):
    """Function to display aggrid by theme

    Args:
        df_collection (dict): Dataframe collection
        current_theme_index (int): Current theme index
    """

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
    if "grid_responses" not in st.session_state:
        grid_responses = {}
    else:
        grid_responses = st.session_state["grid_responses"]

    with st.form("aggrid_form"):
        load_state = False

        selected_rows = []

        if current_theme in grid_responses:
            load_state = True
            df = grid_responses[current_theme]["data"].copy()
            selected_rows = copy.deepcopy(
                grid_responses[current_theme]["selected_rows"]
            )

        current_response = display_aggrid(df, load_state, selected_rows)

        if st.form_submit_button("Confirm"):
            grid_responses[current_theme] = current_response

            st.session_state["grid_responses"] = grid_responses

            valid_submission = validate_current_response(current_response)

            if "grid_responses_validation" not in st.session_state:
                grid_responses_validation = {}
            else:
                grid_responses_validation = st.session_state[
                    "grid_responses_validation"
                ]

            grid_responses_validation[current_theme] = valid_submission
            st.session_state["grid_responses_validation"] = grid_responses_validation

            if valid_submission:
                st.success(
                    f"Article Labels Confirmed for {current_theme}!",
                    icon="✅",
                )
                st.experimental_rerun()

            else:
                st.warning(
                    f"Please enter blank fields that require new inputs!",
                    icon="⚠️",
                )
                st.experimental_rerun()
        elif (
            load_state
            and (
                "grid_responses_validation"
                in st.session_state["grid_responses_validation"]
            )
            and current_theme in st.session_state["grid_responses_validation"]
            and st.session_state["grid_responses_validation"][current_theme]
        ):
            st.success(
                f"Article Labels Confirmed for {current_theme}!",
                icon="✅",
            )
        elif (
            load_state
            and (
                "grid_responses_validation"
                in st.session_state["grid_responses_validation"]
            )
            and current_theme in st.session_state["grid_responses_validation"]
            and not st.session_state["grid_responses_validation"][current_theme]
        ):
            st.warning(
                f"Please enter blank fields that require new inputs!",
                icon="⚠️",
            )

    # Buttons
    nav_buttons(current_theme_index, n_themes)

    return


def nav_buttons(current_theme_index, n_themes):
    """Function to display navigation buttons

    Args:
        current_theme_index (int): Current theme index
        n_themes (int): Total number of themes
    """

    b1, b2, b3, b4, b5 = st.columns([7, 1, 1, 1, 7])
    with b2:
        if st.button("Prev"):
            current_theme_index = max(current_theme_index - 1, 0)
            st.session_state["current_theme_index"] = current_theme_index
            st.experimental_rerun()
    with b3:
        st.markdown(
            f'<div style="text-align: left;">{current_theme_index + 1} of {n_themes}</div>',
            unsafe_allow_html=True,
        )

    with b4:
        if st.button("Next"):
            current_theme_index = min(current_theme_index + 1, n_themes - 1)
            st.session_state["current_theme_index"] = current_theme_index
            st.experimental_rerun()


def theme_jumper(df_collection):
    """Function to display theme jumper dropdown

    Args:
        df_collection (dict): Dataframe collection
    """

    current_theme_index = st.session_state["current_theme_index"]
    current_theme = list(df_collection.keys())[current_theme_index]
    theme_list = list(df_collection.keys())
    theme_index = theme_list.index(current_theme)

    with st.form("index_jumper_form"):
        theme_index = st.selectbox(
            "Jump to Theme",
            theme_list,
            index=theme_index,
        )

        if st.form_submit_button("Jump"):
            current_theme_index = theme_list.index(theme_index)
            st.session_state["current_theme_index"] = current_theme_index
            st.experimental_rerun()


def table_pagination_menu():
    """Function to display table pagination menu

    Returns:
        int: Number of articles per page
    """

    if "n_articles_per_page" in st.session_state:
        n_articles_per_page = st.session_state["n_articles_per_page"]
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
            st.session_state["n_articles_per_page"] = n_articles_per_page
            st.experimental_rerun()
    return


def validate_current_response(current_response):
    """Function to validate current response of aggrid table

    Args:
        current_response (dict): Current response of aggrid table

    Returns:
        bool: True if valid, False otherwise
    """

    incomplete_theme = []
    incomplete_index = []
    incomplete_label = []

    for row in current_response["selected_rows"]:
        if row["Predicted_theme_and_index"] == "-Enter New Label" and (
            row["theme"] == "" or row["index"] == ""
        ):
            incomplete_label.append(row["_selectedRowNodeInfo"]["nodeRowIndex"])
        if row["theme"] == "-Enter New Theme" and row["new theme"] == "":
            incomplete_theme.append(row["_selectedRowNodeInfo"]["nodeRowIndex"])
        if row["index"] == "-Enter New Index" and row["new index"] == "":
            incomplete_index.append(row["_selectedRowNodeInfo"]["nodeRowIndex"])

    if len(incomplete_label) or len(incomplete_theme) or len(incomplete_index):
        return False
    else:
        return True


# @st.cache_resource(experimental_allow_widgets=True, show_spinner=False)
def display_aggrid(df, load_state, selected_rows):
    """Function to display aggrid table

    Args:
        df (pandas.DataFrame): Dataframe to be displayed
        load_state (bool): True if loading from cache, False otherwise
        selected_rows (list): List of selected rows

    Returns:
        dict: Current response of aggrid table
    """

    # Initialising columns for new input
    if not load_state:
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
        df["suggested_labels"] = df["suggested_labels"].apply(
            lambda x: ast.literal_eval(x) if type(x) == str else x
        )
        df["taxonomy"] = df["taxonomy"].apply(
            lambda x: ast.literal_eval(x) if type(x) == str else x
        )

    columns_to_show = [
        "headline",
        "facebook_interactions",
        "domain",
        "theme",
        "index",
        "Predicted_theme_and_index",
        "subindex",
    ]

    gb = GridOptionsBuilder.from_dataframe(df)

    gb.configure_default_column(
        groupable=True,
        value=True,
        enableRowGroup=True,
        aggFunc="count",
        filterable=True,
        sortable=False,
        suppressMenu=True,
    )

    # Configure individual columns
    headlinejs = JsCode(
        """function(params) {return `<a href=${params.data.link} target="_blank" style="text-decoration: none; color: white"> ${params.data.headline} </a>`}"""
    )

    gb.configure_column(
        "headline",
        # headerCheckboxSelection=True,
        # headerCheckboxSelectionCurrentPageOnly=True,
        width=1200,
        cellRenderer=headlinejs,
    )

    labeljs = JsCode(
        """
        function(params) {
        const predictedLabels = params.data.suggested_labels;
        predictedLabels.indexOf("-Enter New Label") === -1 ? predictedLabels.push("-Enter New Label") : null;
            return {
                values: predictedLabels,
                popupPosition: "under",
                cellHeight: 30,
            }
        }
        
        """
    )

    gb.configure_column(
        "Predicted_theme_and_index",
        editable=True,
        cellEditor="agRichSelectCellEditor",
        cellEditorParams=labeljs,
        width=900,
    )

    themejs = JsCode(
        """
        function(params) {
        const themes = Object.keys(params.data.taxonomy);
        themes.indexOf("") === -1 ? themes.push("") : null;
            return {
                value: "",
                values: themes.sort(),
                popupPosition: "under",
                cellHeight: 30,
            }
        }
        
        """
    )

    editableCelljs = JsCode(
        """
        function(params) {
        const label = params.data.Predicted_theme_and_index;
        var editable_cell = label == "-Enter New Label";
        return editable_cell;
        }
        """
    )

    indexesjs = JsCode(
        """
        function(params) {
        const theme = params.data.theme;
        const indexes = params.data.taxonomy[theme];
        indexes.indexOf("") === -1 ? indexes.push("") : null;
            return {
                values: indexes.sort(),
                popupPosition: "under",
                cellHeight: 30,
            }
        }
        
        """
    )

    gb.configure_column(
        "theme",
        editable=editableCelljs,
        cellEditor="agRichSelectCellEditor",
        cellEditorParams=themejs,
    )

    gb.configure_column(
        "index",
        editable=editableCelljs,
        cellEditor="agRichSelectCellEditor",
        cellEditorParams=indexesjs,
    )

    gb.configure_column("subindex", editable=True)

    # Pagination

    if "n_articles_per_page" in st.session_state:
        n_articles_per_page = st.session_state["n_articles_per_page"]
    else:
        n_articles_per_page = 20
        st.session_state["n_articles_per_page"] = n_articles_per_page

    gb.configure_pagination(
        paginationAutoPageSize=False,
        paginationPageSize=n_articles_per_page,
    )

    selected_rows_id = [
        row["_selectedRowNodeInfo"]["nodeRowIndex"] for row in selected_rows
    ]
    gb.configure_selection(
        selection_mode="multiple",
        use_checkbox=True,
        pre_selected_rows=selected_rows_id,
        # select current page only
        suppressRowClickSelection=True,
        suppressRowDeselection=True,
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
        update_mode=GridUpdateMode.MODEL_CHANGED | GridUpdateMode.VALUE_CHANGED,
        columns_auto_size_mode=ColumnsAutoSizeMode.FIT_ALL_COLUMNS_TO_VIEW,
        allow_unsafe_jscode=True,
        # defaultColGroupDef=selectAlljs,
    )

    return grid_response
