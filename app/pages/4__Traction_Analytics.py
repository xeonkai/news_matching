import datetime
import random

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from st_pages import add_page_title
from utils import core

st.warning("Work in Progress")

add_page_title(layout="wide")

st.title("🖥️ Theme and Index Analysis")

st.markdown("""---""")
st.markdown(
    """
    In this page, you can view the time series data analysis of the themes and index.
    """
)
st.markdown("""---""")

file_handler = core.FileHandler(core.DATA_DIR)
filter_bounds = st.cache_data(file_handler.get_filter_bounds, ttl=15)()

with st.sidebar:
    st.markdown("# Filters")

    # numerical entry filter for minimum facebook engagement
    min_engagement = int(
        st.number_input(
            label="Minimum no. of Facebook Interactions:",
            min_value=0,
            max_value=filter_bounds["max_fb_interactions"],
            value=100,
        )
    )
    # filter for date range of articles
    date_range = st.date_input(
        "Date range of articles",
        value=(
            filter_bounds["max_date"] - datetime.timedelta(days=1),
            filter_bounds["max_date"],
        ),
        min_value=filter_bounds["min_date"],
        max_value=filter_bounds["max_date"],
    )
    col_start_time, col_end_time = st.columns([1, 1])
    with col_start_time:
        start_time = st.time_input(
            "Start time on first selected day", value=datetime.time(0, 0)
        )
    with col_end_time:
        end_time = st.time_input(
            "End time on last selected day", value=datetime.time(23, 59)
        )

    # combine date and time
    start_date = datetime.datetime.combine(date_range[0], start_time)
    end_date = datetime.datetime.combine(date_range[1], end_time)
    datetime_bounds = (start_date, end_date)

    # selection-based filter for article domains to be removed
    domain_filter = st.multiselect(
        label="Article domains to exclude",
        options=filter_bounds["domain_list"],
        default=[],
    )


with st.expander("View Raw Data"):
    col1, col2 = st.columns([1, 1])
    with col1:
        df = file_handler.labelled_query()
        st.metric(label="Number of Rows", value=df.shape[0])
    st.dataframe(
        df,
        column_order=[
            "published",
            "headline",
            "summary",
            "subindex",
            "facebook_interactions",
            "link",
            "facebook_link",
            "domain",
            "themes",
            "indexes",
        ],
    )


def run_summary_tab(df):
    n_themes = df["themes"].explode().nunique()
    n_indexes = df["indexes"].explode().nunique()
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

    theme_df = (
        df[["published", "facebook_interactions", "subindex", "themes"]]
        .explode("themes")
        .dropna(subset="themes")
    )
    index_df = (
        df[["published", "facebook_interactions", "subindex", "indexes"]]
        .explode("indexes")
        .dropna(subset="indexes")
    )

    theme_heatmap_chart = (
        alt.Chart(
            # groupby month
            theme_df
            .groupby(["themes", pd.Grouper(key="published", freq="D")])
            .agg(
                facebook_interactions=pd.NamedAgg(
                    column="facebook_interactions", aggfunc="mean"
                ),
                count=pd.NamedAgg(column="facebook_interactions", aggfunc="count"),
                subindexes=pd.NamedAgg(column="subindex", aggfunc=set),
            )
            .reset_index()
        )
        .mark_rect()
        .encode(
            x=alt.X(
                "monthdate(published):T",
                title="Date Published",
                axis=alt.Axis(labelAngle=-60),
            ),
            y=alt.Y("themes", title="Theme"),
            color=alt.Color(
                "facebook_interactions",
                title="Mean Facebook Interactions",
                scale=alt.Scale(scheme="greens"),
            ),
            tooltip=["published", "themes", "facebook_interactions"],
        )
    )
    index_heatmap_chart = (
        alt.Chart(
            index_df
            .groupby(["indexes", pd.Grouper(key="published", freq="D")])
            .agg(
                facebook_interactions=pd.NamedAgg(
                    column="facebook_interactions", aggfunc="mean"
                ),
                count=pd.NamedAgg(column="facebook_interactions", aggfunc="count"),
                subindexes=pd.NamedAgg(column="subindex", aggfunc=set),
            )
            .reset_index()
        )
        .mark_rect()
        .encode(
            x=alt.X(
                "monthdate(published):T",
                title="Date Published",
                axis=alt.Axis(labelAngle=-60),
            ),
            y=alt.Y("indexes", title="Index"),
            color=alt.Color(
                "facebook_interactions",
                title="Mean Facebook Interactions",
                scale=alt.Scale(
                    scheme="greens"
                ),  # , bins=alt.BinParams(step=700), nice=True),
            ),
            tooltip=["published", "indexes", "facebook_interactions"],
        )
    )
    st.altair_chart(theme_heatmap_chart, use_container_width=True)
    st.altair_chart(index_heatmap_chart, use_container_width=True)


def run_theme_tab(df):
    theme_df = (
        df[["published", "facebook_interactions", "subindex", "themes"]]
        .explode("themes")
        .dropna(subset="themes")
    )


    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Calendar", "Heatmap", "Count Analysis", "Mean Analysis", "Sum Analysis"]
    )
    with tab1:
        # https://deepnote.com/@andrii-hazin/Altair-Calendar-View-1440f435-f1a1-4a6d-8962-a4b776be03af
        date = pd.date_range(start="2021-01-01", end="2021-12-31")
        value = np.random.normal(10, 5, size=365)
        df = pd.DataFrame(list(zip(date, value)), columns=["date", "value"])
        df["week"] = df["date"].dt.strftime("%W")
        df["weekday"] = df["date"].dt.day_name()

        chart = (
            alt.Chart(df)
            .mark_rect()
            .encode(
                x=alt.X(field="week", type="ordinal", title=None),
                y=alt.Y(
                    field="weekday",
                    type="ordinal",
                    sort=alt.Sort(
                        [
                            "Monday",
                            "Tuesday",
                            "Wednesday",
                            "Thursday",
                            "Friday",
                            "Saturday",
                            "Sunday",
                        ]
                    ),
                    axis=alt.Axis(labelExpr = "slice(datum.label,0,3)"),
                ),
                color=alt.Color(
                    field="value",
                    type="quantitative",
                    scale=alt.Scale(scheme="redblue", domainMid=0),
                ),
                # tooltip=alt.Tooltip(field='value', type='quantitative'),
                column=alt.Column(
                    field="date", type="temporal", timeUnit="month", title=None
                ),
            )
            .resolve_scale(x="independent")
            .configure_legend(orient='top', gradientLength=600 * 1.2, gradientThickness=10)
            .properties(
                width=600 / 12,

            )
            # .configure_legend(orient="top")
            .configure_axis(labelAngle=0)
        )
        st.altair_chart(chart)

    with tab3:
        unique_themes = sorted(theme_df["themes"].unique().tolist())
        labels = [theme + " " for theme in unique_themes]

        selection = alt.selection_point(fields=["theme"], bind="legend")

        chart = (
            (
                alt.Chart(theme_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X(
                        "monthdate(published):T",
                        title="Published",
                        axis=alt.Axis(labelAngle=-45),
                    ),
                    y=alt.Y("count(themes)", title="Number of Articles"),
                    color=alt.Color("themes", title="Theme").scale(
                        domain=unique_themes, scheme="category20"
                    ),
                    tooltip=["published", "themes", "count()"],
                )
                .properties(width=800, height=600)
            )
            .add_params(selection)
            .transform_filter(selection)
        )
        st.altair_chart(chart)


def run_index_tab(df):
    pass


tab1, tab2, tab3 = st.tabs(["Summary Analysis", "Theme Analysis", "Index Analysis"])

with tab1:
    run_summary_tab(df)

with tab2:
    run_theme_tab(df)

with tab3:
    run_index_tab(df)


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
        df.groupby(["date_time_extracted"])
        .agg({"facebook_interactions": "mean"})
        .reset_index()
    )

    # convert date_time_extracted to datetime
    df["date_time_extracted"] = pd.to_datetime(df["date_time_extracted"])

    fig = calplot(
        df,
        x="date_time_extracted",
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
        df.groupby(["date_time_extracted"])
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
                "monthdate(date_time_extracted):T",
                title="Date Extracted",
                axis=alt.Axis(labelAngle=-45),
            ),
            y=alt.Y("theme", title="Theme"),
            color=alt.Color(
                "mean(facebook_interactions)",
                title="Mean Facebook Interactions",
                scale=alt.Scale(scheme="greens"),
            ),
            tooltip=["date_time_extracted", "theme", "mean(facebook_interactions)"],
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
                    "monthdate(date_time_extracted):T",
                    title="Date Extracted",
                    axis=alt.Axis(labelAngle=-45),
                ),
                y=alt.Y(y, title=title),
                color=alt.Color("theme", title="Theme").scale(
                    domain=unique_themes, scheme="category20"
                ),
                tooltip=["date_time_extracted", "theme", y],
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
        df_mean_filtered["date_time_extracted"].isin(
            df_mean_threshold["date_time_extracted"].unique()
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
                uploaded_data_filtered.groupby(["theme", "date_time_extracted"])[
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
                ["theme", "date_time_extracted"],
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
                uploaded_data_filtered.groupby(["theme", "date_time_extracted"])[
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
                ["theme", "date_time_extracted"],
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
        df.groupby(["date_time_extracted"])
        .agg({"facebook_interactions": "mean"})
        .reset_index()
    )

    # convert date_time_extracted to datetime
    df["date_time_extracted"] = pd.to_datetime(df["date_time_extracted"])

    fig = calplot(
        df,
        x="date_time_extracted",
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
                "monthdate(date_time_extracted):T",
                title="Date Extracted",
                axis=alt.Axis(labelAngle=-45),
            ),
            y=alt.Y("index", title="Index"),
            color=alt.Color(
                "mean(facebook_interactions)",
                title="Mean Facebook Interactions",
                scale=alt.Scale(scheme="greens"),
            ),
            tooltip=["date_time_extracted", "index", "mean(facebook_interactions)"],
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
                    "monthdate(date_time_extracted):T",
                    title="Date Extracted",
                    axis=alt.Axis(labelAngle=-45),
                ),
                y=alt.Y(y, title=title),
                color=alt.Color("index", title="Index").scale(
                    domain=unique_indexes, scheme="category20"
                ),
                tooltip=["date_time_extracted", "index", y],
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
        df_mean_filtered["date_time_extracted"].isin(
            df_mean_threshold["date_time_extracted"].unique()
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
                theme_data.groupby(["index", "date_time_extracted"])[
                    "facebook_interactions"
                ]
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
                theme_data,
                ["index", "date_time_extracted"],
                "facebook_interactions",
                "mean",
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
                theme_data.groupby(["index", "date_time_extracted"])[
                    "facebook_interactions"
                ]
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
                theme_data,
                ["index", "date_time_extracted"],
                "facebook_interactions",
                "sum",
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


if __name__ == "__main__":
    # upload_data()
    # analyze_uploaded_data()
    pass
