import pandas as pd
import ast
import streamlit as st

# function to process the uploaded csv data

def process_data(df):
    df["suggested_labels"] = df["suggested_labels"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df["theme"] = df["suggested_labels"].apply(lambda x: x[0].split('>')[0].strip())
    df["index"] = df["suggested_labels"].apply(lambda x: x[0].split('>')[1].strip())
    df["date"] = df["published"].apply(lambda x: pd.to_datetime(x).date())
    df["time"] = df["published"].apply(lambda x: pd.to_datetime(x).time())
    df["date_extracted"] = df["date_extracted"].apply(lambda x: pd.to_datetime(x).date())
    # df["date_extracted"] = pd.to_datetime(df["date_extracted"])

    # Sorting by headline
    df = df.sort_values(by=["headline", "published", "date_extracted"], ascending=[False, True, True]).reset_index(drop=True)

    # calculating difference between consecutive rows for each article
    df["facebook_interactions_abs_change"] = df.groupby("link")["facebook_interactions"].diff().fillna(0)

    # calculating percentage change between consecutive rows for each article
    df["facebook_interactions_pct_change"] = df.groupby("link")["facebook_interactions"].pct_change().fillna(0)


    # filtering to only include the relavent columns
    df = df[["published", "date", "time", "date_extracted", "headline", "summary", "link", "domain", "facebook_interactions","facebook_interactions_abs_change","facebook_interactions_pct_change", "theme", "index"]]
    

    return df

def aggregate_pct_change(df, groupby_col, agg_col, agg_func):
    # groub by the groupby_col and aggregate the agg_col by agg_func, then calculate the pct_change
    df = df.groupby(groupby_col)[agg_col].agg(agg_func).reset_index()
    df["pct_change"] = df.groupby(groupby_col[0])[agg_col].pct_change().fillna(0) * 100
    df["abs_change"] = df.groupby(groupby_col[0])[agg_col].diff().fillna(0)
    # st.write(df)

    return df

def filter_data(df, min_interactions, date_range, selected_themes, selected_index):
    df = df[df["facebook_interactions"] >= min_interactions]
    df = df[df["date"] >= date_range[0]]
    df = df[df["date"] <= date_range[1]]

    df = df[~df["theme"].isin(selected_themes)]
    df = df[~df["index"].isin(selected_index)]

    return df

def filter_data_by_theme(df, theme):
    df = df[df["theme"] == theme]

    return df

