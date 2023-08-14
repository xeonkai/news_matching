import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
from plotly_calplot import calplot
import plotly.graph_objects as go
from st_aggrid import AgGrid, GridUpdateMode, ColumnsAutoSizeMode, JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder
import random
import io
import utils.utils as utils


# abstract function to plot altair line chart for theme


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


# abstraction function to plot altair line chart for index


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


# function to plot altair heatmap for theme and index


def plot_heatmap(df):
    df["theme_index"] = df["theme"] + " > " + df["index"]

    if utils.check_session_state_key("aggregate_by"):
        agg_by = utils.get_cached_object("aggregate_by")
    else:
        agg_by = "day"

    df_agg = (
        df.groupby(["date_extracted", "theme_index"])
        .agg({"facebook_interactions": "mean"})
        .reset_index()
    )

    interval = 3000
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

    return chart


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


# Function to plot calplot for theme
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


def show_summary_metrics(df):
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


# function to show key theme level metrics of the dataframe


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


# function to show key index level metrics of the dataframe


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


def make_clickable(text, link):
    return f'<a target="_blank" href="{link}">{text}</a>'


def make_clickable_df(df):
    df = df.copy()
    cols = list(df.columns)
    df["headline"] = df.apply(
        lambda x: make_clickable(x[cols.index("headline")], x[cols.index("link")]),
        axis=1,
    )
    return df


# function to show dataframe of articles that exceed the threshold


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


def to_excel(df):
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine="xlsxwriter")
    df.to_excel(writer, sheet_name="Sheet1")
    writer.save()
    processed_data = output.getvalue()
    return processed_data


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
