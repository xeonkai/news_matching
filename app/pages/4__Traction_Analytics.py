import streamlit as st
from streamlit_extras.dataframe_explorer import dataframe_explorer
import pandas as pd
import ast
from plotly_calplot import calplot
import plotly.graph_objects as go
from st_aggrid import AgGrid, GridUpdateMode, ColumnsAutoSizeMode, JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder
import altair as alt
import random
from utils import core
from st_pages import add_page_title

add_page_title(layout="wide")

# st.set_page_config(
#     page_title="Traction Analytics Interface Demo", page_icon="📰", layout="wide"
# )

# st.title("🖥️ Traction Analytics Interface Demo")
st.markdown("""---""")
st.subheader("Welcome!")

st.markdown(
    """
    This is a demo of the Traction Analytics Interface. Upload a CSV file of the news daily scans below to begin.
    """
)

st.markdown("""---""")

st.write("")
daily_file_handler = core.FileHandler(core.DATA_DIR)

def upload_data():
    # Upload file
    # uploaded_file = st.file_uploader("Upload Excel file", type="xlsx")
    # if uploaded_file is not None:
    #     st.write(f"You selected {uploaded_file.name}")
    #     if "traction_data" in st.session_state:
    #         if st.button("Overwrite previous traction data"):
    #             pass
    #         else:
    #             return
    st.session_state["traction_data"] = daily_file_handler.full_query()

    st.write("")

    st.markdown("""---""")


def analyze_uploaded_data():
    st.title("🖥️ Theme and Index Analysis")
    st.markdown("""---""")
    st.markdown(
        """
        In this page, you can view the time series data analysis of the themes and index.
        """
    )
    st.markdown("""---""")

    if "traction_data" in st.session_state:
        uploaded_data = st.session_state["traction_data"]
        uploaded_data = process_data(uploaded_data)
        with st.expander("View Raw Data"):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.metric(label="Number of Rows", value=uploaded_data.shape[0])
            st.dataframe(dataframe_explorer(uploaded_data), use_container_width=True)

        max_date = uploaded_data["date"].max()
        min_date = uploaded_data["date"].min()
        max_interactions = uploaded_data["facebook_interactions"].max()
        min_interactions = uploaded_data["facebook_interactions"].min()

        with st.sidebar:
            with st.form("min_interactions"):
                st.subheader("Filters")
                st.write("Input filters for data analysis.")

                min_interactions = st.slider(
                    "Minimum Number of Facebook Interactions per article",
                    min_value=0,
                    max_value=10000,
                    value=500,
                )
                date_range = st.date_input(
                    "Date range of articles",
                    value=(
                        min_date,
                        max_date,
                    ),
                    min_value=min_date,
                    max_value=max_date,
                )

                aggregate_by = st.selectbox(
                    "Aggregate by",
                    options=["day", "week", "month"],
                    index=0,
                )

                selected_themes = st.multiselect(
                    "Select Themes to Exclude",
                    options=sorted(uploaded_data["theme"].unique().tolist()),
                    default=["general"],
                )

                selected_index = st.multiselect(
                    "Select Index to Exclude",
                    options=sorted(uploaded_data["index"].unique().tolist()),
                    default=["others"],
                )

                submit_button = st.form_submit_button(label="Submit")

        uploaded_data_filtered = uploaded_data.loc[
            lambda df: (df["facebook_interactions"] >= min_interactions)
            & (df["date"].between(date_range[0], date_range[1]))
            & (~df["theme"].isin(selected_themes))
            & (~df["index"].isin(selected_index))
        ]

        period_mapping = {"day": "D", "week": "W", "month": "M"}

        uploaded_data_filtered["date_extracted"] = (
            pd.to_datetime(uploaded_data_filtered["date_extracted"])
            .dt.to_period(period_mapping[aggregate_by])
            .dt.start_time
        )

        tab1, tab2, tab3 = st.tabs(
            ["Summary Analysis", "Theme Analysis", "Index Analysis"]
        )

        with tab1:
            run_summary_tab(uploaded_data_filtered)

        with tab2:
            run_theme_tab(uploaded_data_filtered)

        with tab3:
            run_index_tab(uploaded_data_filtered)


def run_summary_tab(df):
    st.write()

    n_themes = df["theme"].nunique()
    n_indexes = df["index"].nunique()
    n_articles = df.shape[0]
    sum_interactions = df["facebook_interactions"].sum()
    mean_interactions = round(df["facebook_interactions"].mean(), 2)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(label="Number of Themes", value="{0:,}".format(n_themes))
    with col2:
        st.metric(label="Number of Indexes", value="{0:,}".format(n_indexes))
    with col3:
        st.metric(label="Number of Articles", value="{0:,}".format(n_articles))
    with col4:
        st.metric(
            label="Sum of Facebook Interactions", value="{0:,}".format(sum_interactions)
        )
    with col5:
        st.metric(
            label="Mean of Facebook Interactions",
            value="{0:,.2f}".format(mean_interactions),
        )

    st.markdown("""---""")

    df["theme_index"] = df["theme"] + " > " + df["index"]

    chart = (
        alt.Chart(df)
        .mark_rect()
        .encode(
            x=alt.X(
                "monthdate(date_extracted):T",
                title="Date Extracted",
                axis=alt.Axis(labelAngle=-45),
            ),
            y=alt.Y("theme_index", title="Theme + Index"),
            color=alt.Color(
                "mean(facebook_interactions)",
                title="Mean Facebook Interactions",
                scale=alt.Scale(
                    scheme="greens"
                ),  # , bins=alt.BinParams(step=700), nice=True),
            ),
            tooltip=["date_extracted", "theme_index", "mean(facebook_interactions)"],
        )
        .properties(width=1600)
    ).configure_axis(
        labelLimit=500,
    )

    st.altair_chart(chart, use_container_width=True)


def show_theme_metrics(df):
    n_themes = df["theme"].nunique()
    n_articles = df.shape[0]
    sum_interactions = df["facebook_interactions"].sum()
    mean_interactions = round(df["facebook_interactions"].mean(), 2)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="Number of Themes", value="{0:,}".format(n_themes))
    with col2:
        st.metric(label="Number of Articles", value="{0:,}".format(n_articles))
    with col3:
        st.metric(
            label="Sum of Facebook Interactions", value="{0:,}".format(sum_interactions)
        )
    with col4:
        st.metric(
            label="Mean of Facebook Interactions",
            value="{0:,.2f}".format(mean_interactions),
        )


def plot_theme_calplot(df):
    # df[df['theme'] == theme]
    # # aggregate facebook_interactions by day
    df = (
        df.groupby(["date_extracted"])
        .agg({"facebook_interactions": "mean"})
        .reset_index()
    )

    # convert date_extracted to datetime
    df["date_extracted"] = pd.to_datetime(df["date_extracted"])

    fig = calplot(
        df,
        x="date_extracted",
        y="facebook_interactions",
        dark_theme=True,
        month_lines_color="#fff",
    )
    fig.update_layout(
        height=400,
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
    )

    return fig


def show_colour_scale(df):
    df = (
        df.groupby(["date_extracted"])
        .agg({"facebook_interactions": "mean"})
        .reset_index()
    )
    min_y = df["facebook_interactions"].min()
    max_y = df["facebook_interactions"].max()

    dummy_trace = go.Scatter(
        y=[min_y, max_y],
        mode="markers",
        marker=dict(
            size=0.0001,
            color=[min_y, max_y],
            colorscale="greens",
            colorbar=dict(
                thickness=20,
                x=-1,
                y=1,
                title="Mean Facebook Interactions",
                dtick=1000,  # adjust here
            ),
            showscale=True,
        ),
        hoverinfo="none",
    )
    layout = dict(xaxis=dict(visible=False), yaxis=dict(visible=False))
    fig = go.Figure([dummy_trace], layout)

    # disable graph interactivity
    fig.update_layout(
        clickmode=None,
        hovermode=False,
        dragmode=False,
        selectdirection=None,
        modebar=dict(
            remove=[
                "zoom2d",
                "pan2d",
                "select2d",
                "lasso2d",
                "zoomIn2d",
                "zoomOut2d",
                "autoScale2d",
                "resetScale2d",
                "hoverClosestCartesian",
                "hoverCompareCartesian",
                "zoom3d",
                "pan3d",
                "orbitRotation",
                "tableRotation",
                "handleDrag3d",
                "resetCameraDefault3d",
                "resetCameraLastSave3d",
                "hoverClosest3d",
                "hoverClosestGl2d",
                "hoverClosestPie",
                "toggleHover",
                "resetViews",
                "toggleSpikelines",
                "resetViewMapbox",
                "toImage",
            ]
        ),
        margin=dict(l=0, r=0, t=0, b=0),
    )

    # fig.update_layout(
    #     title=dict(text="Mean Facebook Interactions", font=dict(size=12), automargin=True, yref='paper'),
    # )

    return fig


def plot_theme_heatmap(df):
    chart = (
        alt.Chart(df)
        .mark_rect()
        .encode(
            x=alt.X(
                "monthdate(date_extracted):T",
                title="Date Extracted",
                axis=alt.Axis(labelAngle=-45),
            ),
            y=alt.Y("theme", title="Theme"),
            color=alt.Color(
                "mean(facebook_interactions)",
                title="Mean Facebook Interactions",
                scale=alt.Scale(scheme="greens"),
            ),
            tooltip=["date_extracted", "theme", "mean(facebook_interactions)"],
        )
        .properties(width=1600)
    ).configure_axis(
        labelLimit=500,
    )

    return chart


def plot_theme_timeseries(df, y, title):
    unique_themes = sorted(df["theme"].unique().tolist())
    labels = [theme + " " for theme in unique_themes]

    selection = alt.selection_point(fields=["theme"], bind="legend")

    chart = (
        (
            alt.Chart(df)
            .mark_line(point=True)
            .encode(
                x=alt.X(
                    "monthdate(date_extracted):T",
                    title="Date Extracted",
                    axis=alt.Axis(labelAngle=-45),
                ),
                y=alt.Y(y, title=title),
                color=alt.Color("theme", title="Theme").scale(
                    domain=unique_themes, scheme="category20"
                ),
                tooltip=["date_extracted", "theme", y],
            )
            .properties(width=800, height=600)
        )
        .add_params(selection)
        .transform_filter(selection)
    )

    return chart


@st.cache_resource(experimental_allow_widgets=True)
def display_aggrid(df, theme=True):
    if theme:
        columns_to_show = [
            "published",
            "headline",
            "theme",
            "facebook_interactions",
        ]
    else:
        columns_to_show = [
            "published",
            "headline",
            "index",
            "facebook_interactions",
        ]
    gb = GridOptionsBuilder.from_dataframe(df)

    gb.configure_default_column(
        groupable=False,
        value=True,
        enableRowGroup=False,
        aggFunc="count",
        filterable=True,
        sortable=True,
        suppressMenu=False,
    )

    headlinejs = JsCode(
        """function(params) {return `<a href=${params.data.link} target="_blank" style="text-decoration: none; color: white"> <span title="${params.data.headline}"> ${params.data.headline} </span> </a>`}; """
    )

    tooltipjs = JsCode(
        """ function(params) { return '<span title="' + params.value + '">'+params.value+'</span>';  }; """
    )  # if using with cellRenderer

    gb.configure_column(
        "headline",
        width=600,
        cellRenderer=headlinejs,
    )

    if theme:
        gb.configure_column("theme", width=100, cellRenderer=tooltipjs)
    else:
        gb.configure_column("index", width=100, cellRenderer=tooltipjs)

    gb.configure_column("facebook_interactions", width=90, cellRenderer=tooltipjs)

    gb.configure_column("published", width=150, cellRenderer=tooltipjs)

    gridOptions = gb.build()

    columns_to_hide = [i for i in df.columns if i not in columns_to_show]
    column_defs = gridOptions["columnDefs"]
    for col in column_defs:
        if col["headerName"] in columns_to_hide:
            col["hide"] = True
    grid_response = AgGrid(
        df,
        height=400,
        gridOptions=gridOptions,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        columns_auto_size_mode=ColumnsAutoSizeMode.FIT_ALL_COLUMNS_TO_VIEW,
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=True,
        key=random.randint(0, 10000000),
    )
    return grid_response


def show_articles_exceeding_threshold_theme(df_agg, df, criteria, threshold):
    df_mean_threshold = df_agg[df_agg[criteria] >= threshold]
    df_mean_filtered = df[df["theme"].isin(df_mean_threshold["theme"].unique())]
    df_mean_filtered = df_mean_filtered[
        df_mean_filtered["date_extracted"].isin(
            df_mean_threshold["date_extracted"].unique()
        )
    ]
    # df_mean_filtered = make_clickable_df(df_mean_filtered)
    df_mean_filtered = df_mean_filtered[
        ["published", "headline", "theme", "facebook_interactions", "link"]
    ].drop_duplicates(subset=["published", "headline", "theme"])
    df_mean_filtered = df_mean_filtered.sort_values(
        by=["theme", "facebook_interactions"], ascending=[True, False]
    ).reset_index(drop=True)
    # st.dataframe(df_mean_filtered, height=400)
    display_aggrid(df_mean_filtered, theme=True)


def aggregate_pct_change(df, groupby_col, agg_col, agg_func):
    # groub by the groupby_col and aggregate the agg_col by agg_func, then calculate the pct_change
    df = df.groupby(groupby_col)[agg_col].agg(agg_func).reset_index()
    df["pct_change"] = df.groupby(groupby_col[0])[agg_col].pct_change().fillna(0) * 100
    df["abs_change"] = df.groupby(groupby_col[0])[agg_col].diff().fillna(0)
    # st.write(df)

    return df


def run_theme_tab(uploaded_data_filtered):
    st.write()

    show_theme_metrics(uploaded_data_filtered)

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Calendar", "Heatmap", "Count Analysis", "Mean Analysis", "Sum Analysis"]
    )

    with tab1:
        selected_theme = st.multiselect(
            "Select Theme",
            uploaded_data_filtered["theme"].unique(),
            default=uploaded_data_filtered["theme"].unique()[0],
        )

        # filter based on selected theme
        uploaded_data_filtered_cal = uploaded_data_filtered[
            uploaded_data_filtered["theme"].isin(selected_theme)
        ]

        subcol1, subcol2 = st.columns([6, 1])
        with subcol1:
            # st.markdown("##### Mean Facebook Interactions")
            st.plotly_chart(
                plot_theme_calplot(uploaded_data_filtered_cal),
                use_container_width=True,
            )
        with subcol2:
            st.plotly_chart(
                show_colour_scale(uploaded_data_filtered_cal),
                use_container_width=True,
            )

    with tab2:
        st.altair_chart(
            plot_theme_heatmap(uploaded_data_filtered),
            use_container_width=True,
        )
        # visualisation.plot_theme_calplot(uploaded_data_filtered, theme="general")

    with tab3:
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            st.subheader("Count Analysis")

        st.markdown("###### Count of Articles by Theme")
        theme_count_chart = plot_theme_timeseries(
            uploaded_data_filtered, "count()", "Number of Articles"
        )
        st.altair_chart(theme_count_chart, use_container_width=True, theme="streamlit")

        st.markdown("---")

    with tab4:
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            st.subheader("Mean Analysis")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("###### Mean of Facebook Interactions by Theme")
            theme_mean_chart = plot_theme_timeseries(
                uploaded_data_filtered,
                "mean(facebook_interactions)",
                "Mean of Facebook Interactions",
            )
            st.altair_chart(theme_mean_chart, use_container_width=True)

        with col2:
            df_mean = (
                uploaded_data_filtered.groupby(["theme", "date_extracted"])[
                    "facebook_interactions"
                ]
                .mean()
                .sort_values(ascending=False)
                .reset_index()
            )
            with st.form("threshold_form_mean"):
                col1, col2 = st.columns([2, 1])
                threshold = st.slider(
                    "Mean number of Facebook Interaction Threshold",
                    min_value=0,
                    max_value=int(df_mean.facebook_interactions.max()),
                    value=1000,
                    step=100,
                )
                st.form_submit_button(
                    "View Articles Exceeding Threshold",
                    on_click=show_articles_exceeding_threshold_theme(
                        df_mean,
                        uploaded_data_filtered,
                        "facebook_interactions",
                        threshold,
                    ),
                )

        # altair chart for pct change in mean of interactions
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(
                "###### Percent Change in Mean of Facebook Interactions by Theme"
            )
            df_mean_agg = aggregate_pct_change(
                uploaded_data_filtered,
                ["theme", "date_extracted"],
                "facebook_interactions",
                "mean",
            )
            theme_pct_change_chart = plot_theme_timeseries(
                df_mean_agg,
                "pct_change",
                "Percent Change in Mean of Facebook Interactions %",
            )
            st.altair_chart(theme_pct_change_chart, use_container_width=True)
        with col2:
            with st.form("threshold_form_mean_pct_change"):
                threshold = st.slider(
                    "Percent Change Threshold",
                    min_value=0,
                    max_value=int(df_mean_agg["pct_change"].max()),
                    value=50,
                    step=10,
                )
                submit_threshold = st.form_submit_button(
                    "View Articles Exceeding Threshold",
                    on_click=show_articles_exceeding_threshold_theme(
                        df_mean_agg, uploaded_data_filtered, "pct_change", threshold
                    ),
                )

        st.markdown("---")

    with tab5:
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            st.subheader("Sum Analysis")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("###### Sum of Facebook Interactions by Theme")
            theme_sum_chart = plot_theme_timeseries(
                uploaded_data_filtered,
                "sum(facebook_interactions)",
                "Sum of Facebook Interactions",
            )
            st.altair_chart(theme_sum_chart, use_container_width=True)
        with col2:
            df_sum = (
                uploaded_data_filtered.groupby(["theme", "date_extracted"])[
                    "facebook_interactions"
                ]
                .sum()
                .sort_values(ascending=False)
                .reset_index()
            )
            with st.form("threshold_form_sum"):
                threshold = st.slider(
                    "Sum of Facebook Interaction Threshold",
                    min_value=0,
                    max_value=int(df_sum.facebook_interactions.max()),
                    value=1000,
                    step=100,
                )
                submit_threshold_sum = st.form_submit_button(
                    "View Articles Exceeding Threshold",
                    on_click=show_articles_exceeding_threshold_theme(
                        df_sum,
                        uploaded_data_filtered,
                        "facebook_interactions",
                        threshold,
                    ),
                )

        # altair chart for pct change in sum of interactions
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(
                "###### Percent Change in Sum of Facebook Interactions by Theme"
            )
            df_sum_agg = aggregate_pct_change(
                uploaded_data_filtered,
                ["theme", "date_extracted"],
                "facebook_interactions",
                "sum",
            )
            theme_pct_change_chart = plot_theme_timeseries(
                df_sum_agg,
                "pct_change",
                "Percent Change in Sum of Facebook Interactions %",
            )
            st.altair_chart(theme_pct_change_chart, use_container_width=True)
        with col2:
            with st.form("threshold_form_sum_pct_change"):
                threshold = st.slider(
                    "Percent Change Threshold",
                    min_value=0,
                    max_value=int(df_sum_agg["pct_change"].max()),
                    value=50,
                    step=10,
                )
                submit_threshold_sum_pct = st.form_submit_button(
                    "View Articles Exceeding Threshold",
                    on_click=show_articles_exceeding_threshold_theme(
                        df_sum_agg, uploaded_data_filtered, "pct_change", threshold
                    ),
                )


def show_index_metrics(df):
    n_index = df["index"].nunique()
    n_articles = df.shape[0]
    sum_interactions = df["facebook_interactions"].sum()
    mean_interactions = round(df["facebook_interactions"].mean(), 2)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="Number of Indexes", value="{0:,}".format(n_index))
    with col2:
        st.metric(label="Number of Articles", value="{0:,}".format(n_articles))
    with col3:
        st.metric(
            label="Sum of Facebook Interactions", value="{0:,}".format(sum_interactions)
        )
    with col4:
        st.metric(
            label="Mean of Facebook Interactions",
            value="{0:,.2f}".format(mean_interactions),
        )


# Function to plot calplot for index
def plot_index_calplot(df):
    # df[df['index'] == index]
    # # aggregate facebook_interactions by day
    df = (
        df.groupby(["date_extracted"])
        .agg({"facebook_interactions": "mean"})
        .reset_index()
    )

    # convert date_extracted to datetime
    df["date_extracted"] = pd.to_datetime(df["date_extracted"])

    fig = calplot(
        df,
        x="date_extracted",
        y="facebook_interactions",
        dark_theme=True,
        month_lines_color="#fff",
    )
    fig.update_layout(
        height=400,
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
    )

    return fig


def plot_index_heatmap(df):
    chart = (
        alt.Chart(df)
        .mark_rect()
        .encode(
            x=alt.X(
                "monthdate(date_extracted):T",
                title="Date Extracted",
                axis=alt.Axis(labelAngle=-45),
            ),
            y=alt.Y("index", title="Index"),
            color=alt.Color(
                "mean(facebook_interactions)",
                title="Mean Facebook Interactions",
                scale=alt.Scale(scheme="greens"),
            ),
            tooltip=["date_extracted", "index", "mean(facebook_interactions)"],
        )
        .properties(width=1600)
    ).configure_axis(
        labelLimit=500,
    )

    return chart


def plot_index_timeseries(df, y, title):
    unique_indexes = sorted(df["index"].unique().tolist())
    labels = [theme + " " for theme in unique_indexes]

    selection = alt.selection_point(fields=["index"], bind="legend")

    chart = (
        (
            alt.Chart(df)
            .mark_line(point=True)
            .encode(
                x=alt.X(
                    "monthdate(date_extracted):T",
                    title="Date Extracted",
                    axis=alt.Axis(labelAngle=-45),
                ),
                y=alt.Y(y, title=title),
                color=alt.Color("index", title="Index").scale(
                    domain=unique_indexes, scheme="category20"
                ),
                tooltip=["date_extracted", "index", y],
            )
            .properties(width=800, height=600)
        )
        .add_params(selection)
        .transform_filter(selection)
    )
    return chart


def show_articles_exceeding_threshold_index(df_agg, df, criteria, threshold):
    df_mean_threshold = df_agg[df_agg[criteria] >= threshold]
    df_mean_filtered = df[df["index"].isin(df_mean_threshold["index"].unique())]
    df_mean_filtered = df_mean_filtered[
        df_mean_filtered["date_extracted"].isin(
            df_mean_threshold["date_extracted"].unique()
        )
    ]
    df_mean_filtered = df_mean_filtered[
        ["published", "headline", "index", "facebook_interactions", "link"]
    ].drop_duplicates(subset=["published", "headline", "index"])
    df_mean_filtered = df_mean_filtered.sort_values(
        by=["index", "facebook_interactions"], ascending=[True, False]
    ).reset_index(drop=True)
    response = display_aggrid(df_mean_filtered, theme=False)


def run_index_tab(uploaded_data_filtered):
    # get list of themes sorted by sum of interactions
    themes_sorted = uploaded_data_filtered.groupby("theme")[
        "facebook_interactions"
    ].sum()
    themes_sorted = themes_sorted.sort_values(ascending=False)
    themes_sorted = themes_sorted.index.tolist()

    selected_theme = st.selectbox("Select Theme", options=themes_sorted)

    theme_data = uploaded_data_filtered[lambda df: df["theme"] == selected_theme]

    show_index_metrics(theme_data)

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Calendar", "Heatmap", "Count Analysis", "Mean Analysis", "Sum Analysis"]
    )

    with tab1:
        selected_index = st.multiselect(
            "Select Index",
            theme_data["index"].unique(),
            default=theme_data["index"].unique()[0],
        )

        # filter based on selected index
        theme_data_cal = theme_data[theme_data["index"].isin(selected_index)]

        subcol1, subcol2 = st.columns([10, 1])
        with subcol1:
            # st.markdown("##### Mean Facebook Interactions")
            st.plotly_chart(
                plot_index_calplot(theme_data_cal),
                use_container_width=True,
            )
        with subcol2:
            st.plotly_chart(
                show_colour_scale(theme_data_cal),
                use_container_width=True,
            )
    with tab2:
        st.altair_chart(
            plot_index_heatmap(theme_data),
            use_container_width=True,
        )

    with tab3:
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            st.subheader("Count Analysis")

        st.markdown("###### Count of Articles by Index")
        index_count_chart = plot_index_timeseries(
            theme_data, "count()", "Number of Articles"
        )

        st.altair_chart(index_count_chart, use_container_width=True)

        st.markdown("---")

    with tab4:
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            st.subheader("Mean Analysis")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("###### Mean of Facebook Interactions by Index")
            index_mean_chart = plot_index_timeseries(
                theme_data,
                "mean(facebook_interactions)",
                "Mean of Facebook Interactions",
            )

            st.altair_chart(index_mean_chart, use_container_width=True)
        with col2:
            df_mean = (
                theme_data.groupby(["index", "date_extracted"])["facebook_interactions"]
                .mean()
                .sort_values(ascending=False)
                .reset_index()
            )
            with st.form("threshold_form_mean_index"):
                threshold = st.slider(
                    "Mean of Facebook Interaction Threshold",
                    min_value=0,
                    max_value=int(df_mean["facebook_interactions"].max()),
                    value=100,
                    step=10,
                )
                submit_threshold_mean = st.form_submit_button(
                    "View Articles Exceeding Threshold",
                    on_click=show_articles_exceeding_threshold_index(
                        df_mean, theme_data, "facebook_interactions", threshold
                    ),
                )

        # pct change of mean

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(
                "###### Percent Change in Mean of Facebook Interactions by Index"
            )
            df_mean_agg = aggregate_pct_change(
                theme_data, ["index", "date_extracted"], "facebook_interactions", "mean"
            )
            index_pct_change_chart = plot_index_timeseries(
                df_mean_agg,
                "pct_change",
                "Percent Change in Mean of Facebook Interactions %",
            )
            st.altair_chart(index_pct_change_chart, use_container_width=True)
        with col2:
            with st.form("threshold_form_pct_change_index"):
                threshold = st.slider(
                    "Percent Change Threshold",
                    min_value=0,
                    max_value=int(df_mean_agg["pct_change"].max()),
                    value=50,
                    step=10,
                )
                submit_threshold_pct_change = st.form_submit_button(
                    "View Articles Exceeding Threshold",
                    on_click=show_articles_exceeding_threshold_index(
                        df_mean_agg, theme_data, "pct_change", threshold
                    ),
                )

        st.markdown("---")

    with tab5:
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            st.subheader("Sum Analysis")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("###### Sum of Facebook Interactions by Index")
            index_sum_chart = plot_index_timeseries(
                theme_data, "sum(facebook_interactions)", "Sum of Facebook Interactions"
            )

            st.altair_chart(index_sum_chart, use_container_width=True)
        with col2:
            df_sum = (
                theme_data.groupby(["index", "date_extracted"])["facebook_interactions"]
                .sum()
                .sort_values(ascending=False)
                .reset_index()
            )
            with st.form("threshold_form_sum_index"):
                threshold = st.slider(
                    "Sum of Facebook Interaction Threshold",
                    min_value=0,
                    max_value=int(df_sum.facebook_interactions.max()),
                    value=1000,
                    step=100,
                )
                submit_threshold_sum = st.form_submit_button(
                    "View Articles Exceeding Threshold",
                    on_click=show_articles_exceeding_threshold_index(
                        df_sum, theme_data, "facebook_interactions", threshold
                    ),
                )

        # pct change of sum

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(
                "###### Percent Change in Sum of Facebook Interactions by Index"
            )
            df_sum_agg = aggregate_pct_change(
                theme_data, ["index", "date_extracted"], "facebook_interactions", "sum"
            )
            index_pct_change_chart = plot_index_timeseries(
                df_sum_agg,
                "pct_change",
                "Percent Change in Sum of Facebook Interactions %",
            )
            st.altair_chart(index_pct_change_chart, use_container_width=True)
        with col2:
            with st.form("threshold_form_pct_change_sum_index"):
                threshold = st.slider(
                    "Percent Change Threshold",
                    min_value=0,
                    max_value=int(df_sum_agg["pct_change"].max()),
                    value=50,
                    step=10,
                )
                submit_threshold_pct_change = st.form_submit_button(
                    "View Articles Exceeding Threshold",
                    on_click=show_articles_exceeding_threshold_index(
                        df_sum_agg, theme_data, "pct_change", threshold
                    ),
                )


def process_data(df):
    df["suggested_labels"] = df["suggested_labels"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    df["theme"] = df["suggested_labels"].apply(lambda x: x[0].split(">")[0].strip())
    df["index"] = df["suggested_labels"].apply(lambda x: x[0].split(">")[1].strip())
    df["date"] = df["published"].apply(lambda x: pd.to_datetime(x).date())
    df["time"] = df["published"].apply(lambda x: pd.to_datetime(x).time())
    df["date_extracted"] = df["date_extracted"].apply(
        lambda x: pd.to_datetime(x).date()
    )
    # df["date_extracted"] = pd.to_datetime(df["date_extracted"])

    # Sorting by headline
    df = df.sort_values(
        by=["headline", "published", "date_extracted"], ascending=[False, True, True]
    ).reset_index(drop=True)

    # calculating difference between consecutive rows for each article
    df["facebook_interactions_abs_change"] = (
        df.groupby("link")["facebook_interactions"].diff().fillna(0)
    )

    # calculating percentage change between consecutive rows for each article
    df["facebook_interactions_pct_change"] = (
        df.groupby("link")["facebook_interactions"].pct_change().fillna(0)
    )

    # filtering to only include the relavent columns
    df = df[
        [
            "published",
            "date",
            "time",
            "date_extracted",
            "headline",
            "summary",
            "link",
            "domain",
            "facebook_interactions",
            "facebook_interactions_abs_change",
            "facebook_interactions_pct_change",
            "theme",
            "index",
        ]
    ]

    return df


if __name__ == "__main__":
    upload_data()
    analyze_uploaded_data()
