import pandas as pd
import ast

# function to process the uploaded csv data

def process_data(df):
    df["suggested_labels"] = df["suggested_labels"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df["theme"] = df["suggested_labels"].apply(lambda x: x[0].split('>')[0].strip())
    df["index"] = df["suggested_labels"].apply(lambda x: x[0].split('>')[1].strip())
    df["date"] = df["published"].apply(lambda x: pd.to_datetime(x).date())
    df["time"] = df["published"].apply(lambda x: pd.to_datetime(x).time())


    # filtering to only include the relavent columns

    df = df[["published", "date", "time", "headline", "summary", "link", "domain", "facebook_interactions", "theme", "index"]]
    
    df = df.sort_values(by=["date", "headline"], ascending=False)

    return df

def filter_data(df, min_interactions, date_range, selected_themes):
    df = df[df["facebook_interactions"] >= min_interactions]
    df = df[df["date"] >= date_range[0]]
    df = df[df["date"] <= date_range[1]]

    df = df[~df["theme"].isin(selected_themes)]

    return df

def filter_data_by_theme(df, theme):
    df = df[df["theme"] == theme]

    return df

