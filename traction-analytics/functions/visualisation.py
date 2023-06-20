import streamlit as st
import altair as alt
import pandas as pd

# abstract function to plot altair line chart for theme

def plot_theme_timeseries(df, y, title):
    unique_themes = sorted(df["theme"].unique().tolist())
    labels = [theme + ' ' for theme in unique_themes]

    input_dropdown = alt.binding_select(
        # Add the empty selection which shows all when clicked
        options=unique_themes + [None],
        labels=labels + ['All'],
        name='Choose Theme: '
    )
    selection = alt.selection_point(
        fields=['theme'],
        bind=input_dropdown,
    )

    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("monthdate(date_extracted):T", title="Date Extracted", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y(y, title=title),
            color=alt.Color("theme", title="Theme").scale(domain=unique_themes,scheme="category20"),
            tooltip=["date_extracted", "theme", y],
        )
        .properties(width=800, height=500)
        .interactive()
    ).add_params(
        selection
    ).transform_filter(
        selection
    )

    return chart



# abstraction function to plot altair line chart for index

def plot_index_timeseries(df, y, title):
    unique_indexes = sorted(df["index"].unique().tolist())
    labels = [theme + ' ' for theme in unique_indexes]

    input_dropdown = alt.binding_select(
        # Add the empty selection which shows all when clicked
        options=unique_indexes + [None],
        labels=labels + ['All'],
        name='Choose Index: '
    )
    selection = alt.selection_point(
        fields=['index'],
        bind=input_dropdown,
    )
    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("monthdate(date_extracted):T", title="Date Extracted", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y(y, title=title),
            color=alt.Color("index", title="Index").scale(domain=unique_indexes,scheme="category20"),
            tooltip=["date_extracted", "index", y],
        )
        .properties(width=800, height=500)
        .interactive()
    ).add_params(
        selection
    ).transform_filter(
        selection
    )
    return chart

# function to plot altair heatmap for theme and index

def plot_heatmap(df):
    df["theme_index"] = df["theme"] + " > " + df["index"]

    chart = (
        alt.Chart(df).mark_rect().encode(
            x=alt.X("monthdate(date_extracted):T", title="Date Extracted", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("theme_index", title="Theme + Index"),
            color=alt.Color("mean(facebook_interactions)", title="Mean Facebook Interactions"),
            tooltip=["date_extracted", "theme_index", "mean(facebook_interactions)"],
        ).properties(width=1600, height=700)
    ).configure_axis(
        labelLimit=500,

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


def make_clickable(text, link):
    return f'<a target="_blank" href="{link}">{text}</a>'

def make_clickable_df(df):
    df = df.copy()
    cols = list(df.columns)
    df["headline"] = df.apply(lambda x: make_clickable(x[cols.index("headline")], x[cols.index("link")]), axis=1)
    return df

# function to show dataframe of articles that exceed the threshold

def show_articles_exceeding_threshold_theme(df_agg, df, criteria, threshold):
    df_mean_threshold = df_agg[df_agg[criteria] > threshold]
    df_mean_filtered = df[df["theme"].isin(df_mean_threshold["theme"].unique())]
    df_mean_filtered = df_mean_filtered[df_mean_filtered["date_extracted"].isin(df_mean_threshold["date_extracted"].unique())]
    # df_mean_filtered = make_clickable_df(df_mean_filtered)
    df_mean_filtered = df_mean_filtered[["published", "headline", "theme", "facebook_interactions"]].drop_duplicates(subset=["published", "headline", "theme"])
    df_mean_filtered = df_mean_filtered.sort_values(by=["theme", "facebook_interactions"], ascending=[True, False]).reset_index(drop=True)

    # st.write(df_mean_filtered.to_html(escape=False, index=False), unsafe_allow_html=True, height=400)
    
    st.dataframe(df_mean_filtered, height=400)

def show_articles_exceeding_threshold_index(df_agg, df, criteria, threshold):
    df_mean_threshold = df_agg[df_agg[criteria] > threshold]
    df_mean_filtered = df[df["index"].isin(df_mean_threshold["index"].unique())]
    df_mean_filtered = df_mean_filtered[df_mean_filtered["date_extracted"].isin(df_mean_threshold["date_extracted"].unique())]
    df_mean_filtered = df_mean_filtered[["published", "headline", "index", "facebook_interactions"]].drop_duplicates(subset=["published", "headline", "index"])
    df_mean_filtered = df_mean_filtered.sort_values(by=["index", "facebook_interactions"], ascending=[True, False]).reset_index(drop=True)
    st.dataframe(df_mean_filtered, height=400)

