import streamlit as st
import altair as alt
import pandas as pd

# function to plot altair line chart of time series of sum of facebook interactions by theme

def plot_theme_sum(df):
    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("monthdate(date):T", title="Date", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("sum(facebook_interactions)", title="Sum of Facebook Interactions"),
            color=alt.Color("theme", title="Theme"),
            tooltip=["date", "theme", "sum(facebook_interactions)"],
        )
        .properties(width=800, height=500)
        .interactive()
    )
    return chart

# function to plot altair line chart of time series of mean of facebook interactions by theme


def plot_theme_mean(df):
    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("monthdate(date):T", title="Date", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("mean(facebook_interactions)", title="Mean of Facebook Interactions"),
            color=alt.Color("theme", title="Theme"),
            tooltip=["date", "theme", "mean(facebook_interactions)"],
        )
        .properties(width=800, height=500)
        .interactive()
    )
    return chart

def plot_theme_count(df):
    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("monthdate(date):T", title="Date", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("count()", title="Number of Articles"),
            color=alt.Color("theme", title="Theme"),
            tooltip=["date", "theme", "count()"],
        )
        .properties(width=800, height=500)
        .interactive()
    )
    return chart


# function to plot altair line chart of time series of sum of facebook interactions by index

def plot_index_sum(df):
    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("monthdate(date):T", title="Date", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("sum(facebook_interactions)", title="Sum of Facebook Interactions"),
            color=alt.Color("index", title="Index"),
            tooltip=["date", "index", "sum(facebook_interactions)"],
        )
        .properties(width=800, height=500)
        .interactive()
    )
    return chart

# function to plot altair line chart of time series of mean of facebook interactions by index

def plot_index_mean(df):
    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("monthdate(date):T", title="Date", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("mean(facebook_interactions)", title="Mean of Facebook Interactions"),
            color=alt.Color("index", title="Index"),
            tooltip=["date", "index", "mean(facebook_interactions)"],
        )
        .properties(width=800, height=500)
        .interactive()
    )
    return chart

def plot_index_count(df):
    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("monthdate(date):T", title="Date", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("count()", title="Number of Articles"),
            color=alt.Color("index", title="Index"),
            tooltip=["date", "index", "count()"],
        )
        .properties(width=800, height=500)
        .interactive()
    )
    return chart


# function to show key theme level metrics of the dataframe

def show_theme_metrics(df):
    n_themes = df["theme"].nunique()
    n_articles = df.shape[0]
    sum_interactions = df["facebook_interactions"].sum()
    mean_interactions = round(df["facebook_interactions"].mean(),2)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="Number of Themes", value='{0:,}'.format(n_themes))
    with col2:
        st.metric(label="Number of Articles", value='{0:,}'.format(n_articles))
    with col3:
        st.metric(label="Sum of Facebook Interactions", value='{0:,}'.format(sum_interactions))
    with col4:
        st.metric(label="Mean of Facebook Interactions", value='{0:,.2f}'.format(mean_interactions))

    
# function to show key index level metrics of the dataframe

def show_index_metrics(df):
    n_index = df["index"].nunique()
    n_articles = df.shape[0]
    sum_interactions = df["facebook_interactions"].sum()
    mean_interactions = round(df["facebook_interactions"].mean(),2)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="Number of Indexes", value='{0:,}'.format(n_index))
    with col2:
        st.metric(label="Number of Articles", value='{0:,}'.format(n_articles))
    with col3:
        st.metric(label="Sum of Facebook Interactions", value='{0:,}'.format(sum_interactions))
    with col4:
        st.metric(label="Mean of Facebook Interactions", value='{0:,.2f}'.format(mean_interactions))

    