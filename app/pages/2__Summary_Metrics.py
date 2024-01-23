import streamlit as st
import pandas as pd
import datetime
from utils import core
from st_pages import add_page_title

add_page_title(layout="wide")


def mini_groupby(l):
    d = {}
    for subindex, interactions in l:
        d[subindex] = d.get(subindex, 0) + int(interactions)

    return [f"{k} ({v} interactions)" for k, v in d.items()]


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

    top_n = st.number_input("No. Themes/Indexes to Show", min_value=1, value=20, step=1)

    df = file_handler.filtered_query(domain_filter, min_engagement, datetime_bounds)
    df["themes"] = df["themes"].fillna("").apply(list)
    df["indexes"] = df["indexes"].fillna("").apply(list)
    df["themes_indexes"] = df["themes"] + df["indexes"]

    st.markdown("# Group into themes/indexes")

    prioritised_groups = st.multiselect(
        "Priority order for Theme/Index",
        df["themes_indexes"]
        .explode()
        .drop_duplicates()
        .dropna()
        .sort_values()
        .to_list(),
    )


def get_first_priority(row):
    for p in prioritised_groups:
        if p in row:
            return p


df["group"] = df["subindex"]
df["priority_groups"] = df["themes_indexes"].apply(get_first_priority)
df.loc[lambda df: df["priority_groups"].notna(), "group"] = df.loc[
    lambda df: df["priority_groups"].notna(), "priority_groups"
]

theme_metric_col, index_metric_col = st.columns([1, 1])
with theme_metric_col:
    st.subheader("Top 20 themes")
    st.write(
        df.explode("themes")
        .groupby("themes")
        .agg(
            facebook_interactions=pd.NamedAgg(
                column="facebook_interactions", aggfunc="sum"
            ),
            count=pd.NamedAgg(column="facebook_interactions", aggfunc="count"),
        )
        .sort_values("facebook_interactions", ascending=False)
        .iloc[:top_n]
    )

with index_metric_col:
    st.subheader("Top 20 indexes")
    st.write(
        df.explode("indexes")
        .groupby("indexes")
        .agg(
            facebook_interactions=pd.NamedAgg(
                column="facebook_interactions", aggfunc="sum"
            ),
            count=pd.NamedAgg(column="facebook_interactions", aggfunc="count"),
        )
        .sort_values("facebook_interactions", ascending=False)
        .iloc[:top_n]
    )

st.subheader("DOM summmary table")
st.dataframe(
    df.dropna(subset=["subindex"])
    .assign(
        subindex_counts=lambda r: r["subindex"].apply(lambda v: [v])
        + r["facebook_interactions"].apply(lambda v: [str(v)])
    )
    .groupby("group")
    .agg(
        subindex_counts=pd.NamedAgg(column="subindex_counts", aggfunc=mini_groupby),
        facebook_interactions=pd.NamedAgg(
            column="facebook_interactions", aggfunc="sum"
        ),
        count=pd.NamedAgg(column="facebook_interactions", aggfunc="count"),
        headlines=pd.NamedAgg(column="headline", aggfunc=list),
        links=pd.NamedAgg(column="link", aggfunc=list),
    )
    .sort_values(["facebook_interactions", "group"], ascending=False)
)

with st.expander("View raw data"):
    st.dataframe(
        df.dropna(subset=["subindex"])
        .sort_values(["group", "facebook_interactions"], ascending=[True, False])
        .reset_index(drop=True),
        column_config={
            "link": st.column_config.LinkColumn("link", width="small"),
            "facebook_link": st.column_config.LinkColumn("facebook_link", width="small"),
            "facebook_page_name": st.column_config.TextColumn(
                "facebook_page_name", width="small"
            ),
            "domain": st.column_config.TextColumn("domain", width="small"),
            "facebook_interactions": st.column_config.TextColumn(
                "facebook_interactions", width="small"
            ),
            "themes": st.column_config.ListColumn("themes", width="small"),
            "indexes": st.column_config.ListColumn("indexes", width="small"),
        },
        column_order=[
            "published",
            "headline",
            "link",
            "facebook_link",
            "facebook_page_name",
            "facebook_interactions",
            "themes",
            "indexes",
            "subindex",
            "group",
        ],
    )
