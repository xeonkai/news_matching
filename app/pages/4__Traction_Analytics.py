import datetime

import altair as alt
from vega_datasets import data
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
            filter_bounds["min_date"],
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
        df = file_handler.filtered_query(domain_filter, min_engagement, datetime_bounds)
        # df = file_handler.labelled_query()
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
            theme_df.groupby(["themes", pd.Grouper(key="published", freq="D")])
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
        # .properties(width="container")
    )
    index_heatmap_chart = (
        alt.Chart(
            index_df.groupby(["indexes", pd.Grouper(key="published", freq="D")])
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
    st.altair_chart(theme_heatmap_chart)
    st.altair_chart(index_heatmap_chart, use_container_width=True)

    theme_df = (
        df[["published", "facebook_interactions", "subindex", "themes"]]
        .explode("themes")
        .dropna(subset="themes")
    )

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
                axis=alt.Axis(labelExpr="slice(datum.label,0,3)"),
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
        .configure_legend(orient="top", gradientLength=600 * 1.2, gradientThickness=10)
        .properties(
            width=600 / 12,
        )
        # .configure_legend(orient="top")
        .configure_axis(labelAngle=0)
    )
    st.altair_chart(chart)

    unique_themes = sorted(theme_df["themes"].unique().tolist())

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

    source = data.movies.url

    pts = alt.selection_point(encodings=["x"])

    rect = (
        alt.Chart(data.movies.url)
        .mark_rect()
        .encode(
            alt.X("IMDB_Rating:Q").bin(),
            alt.Y("Rotten_Tomatoes_Rating:Q").bin(),
            alt.Color("count()").scale(scheme="greenblue").title("Total Records"),
        )
    )

    circ = (
        rect.mark_point()
        .encode(
            alt.ColorValue("grey"), alt.Size("count()").title("Records in Selection")
        )
        .transform_filter(pts)
    )

    bar = (
        alt.Chart(source, width=550, height=200)
        .mark_bar()
        .encode(
            x="Major_Genre:N",
            y="count()",
            color=alt.condition(
                pts, alt.ColorValue("steelblue"), alt.ColorValue("grey")
            ),
        )
        .add_params(pts)
    )

    c = alt.vconcat(rect + circ, bar).resolve_legend(
        color="independent", size="independent"
    )
    st.altair_chart(c)

    source = data.stocks()
    source = source[source.symbol == "GOOG"]

    # define a point selection
    point = alt.selection_point(encodings=["x"])

    # define a bar chart aggregated by year
    # change the color of the bar when selected
    bar_year = (
        alt.Chart(
            data=source,
            title=["bar chart grouped by `year`", "click bar to select year"],
        )
        .mark_bar(tooltip=True)
        .encode(
            x=alt.X("date:O").timeUnit("year"),
            y=alt.Y("price:Q").aggregate("sum"),
            color=alt.condition(point, alt.value("#1FC3AA"), alt.value("#8624F5")),
        )
        .add_params(point)
    )

    # define a bar chart aggregated by yearmonth
    # change the color of all months of the selected year in the other chart.
    bar_yearmonth = (
        alt.Chart(
            data=source,
            width=alt.Step(5),
            title=[
                "bar chart grouped by `yearmonth`",
                "highlight months of selected year",
            ],
        )
        .mark_bar(tooltip=True)
        .encode(
            x=alt.X("date:O").timeUnit("yearmonth"),
            y=alt.Y("price:Q").aggregate("sum"),
            color=alt.condition(
                f"year({point.name}.year_date) === year(datum['yearmonth_date'])",
                alt.value("#1FC3AA"),
                alt.value("#8624F5"),
            ),
        )
    )

    # horizontal concatenate
    comb_bars = bar_year | bar_yearmonth
    comb_bars
    st.altair_chart(comb_bars)


def run_theme_tab(df):
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Calendar", "Heatmap", "Count Analysis", "Mean Analysis", "Sum Analysis"]
    )


tab1, tab2, tab3 = st.tabs(["Summary Analysis", "Theme Analysis", "Index Analysis"])

with tab1:
    run_summary_tab(df)

with tab2:
    run_theme_tab(df)


def plot_theme_timeseries(df, y, title):
    unique_themes = sorted(df["theme"].unique().tolist())

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


def plot_index_timeseries(df, y, title):
    unique_indexes = sorted(df["index"].unique().tolist())

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


if __name__ == "__main__":
    # upload_data()
    # analyze_uploaded_data()
    pass
