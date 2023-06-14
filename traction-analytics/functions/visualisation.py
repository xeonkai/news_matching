import streamlit as st
import altair as alt
import pandas as pd

# abstract function to plot altair line chart for theme

def plot_theme_timeseries(df, y, title):
    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("monthdate(date_extracted):T", title="Date Extracted", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y(y, title=title),
            color=alt.Color("theme", title="Theme"),
            tooltip=["date_extracted", "theme", y],
        )
        .properties(width=800, height=500)
        .interactive()
    )
    return chart

# abstraction function to plot altair line chart for index

def plot_index_timeseries(df, y, title):
    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("monthdate(date_extracted):T", title="Date Extracted", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y(y, title=title),
            color=alt.Color("index", title="Index"),
            tooltip=["date_extracted", "index", y],
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

    