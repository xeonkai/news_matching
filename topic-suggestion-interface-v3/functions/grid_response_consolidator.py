import pandas as pd
import streamlit as st

# Function to consolidate grid responses into a dataframe


def consolidate_grid_responses(grid_responses, columns):
    consolidated_df = pd.DataFrame()

    for theme in grid_responses:
        sub_df = pd.DataFrame(grid_responses[theme]["selected_rows"])
        consolidated_df = pd.concat([consolidated_df, sub_df])

    consolidated_columns = consolidated_df.columns.tolist()

    # replacing theme values with new theme values if new theme is not empty
    consolidated_df = consolidated_df.apply(
        lambda x: x.replace(
            x[consolidated_columns.index("theme")],
            x[consolidated_columns.index("new theme")],
        )
        if x[consolidated_columns.index("new theme")] != ""
        and x[consolidated_columns.index("theme")] == "-Enter New Theme"
        else x,
        axis=1,
    )

    consolidated_df = consolidated_df.apply(
        lambda x: x.replace(
            x[consolidated_columns.index("index")],
            x[consolidated_columns.index("new index")],
        )
        if x[consolidated_columns.index("new index")] != ""
        and x[consolidated_columns.index("index")] == "-Enter New Index"
        else x,
        axis=1,
    )

    consolidated_df = consolidated_df.apply(
        lambda x: x.replace(
            x[consolidated_columns.index("subindex")],
            x[consolidated_columns.index("new subindex")],
        )
        if x[consolidated_columns.index("new subindex")] != ""
        and x[consolidated_columns.index("subindex")] == "-Enter New Subindex"
        else x,
        axis=1,
    )

    # Filtering out rows without new theme/index/subindex

    consolidated_df = consolidated_df[consolidated_df["theme"] != "-Enter New Theme"]
    consolidated_df = consolidated_df[consolidated_df["index"] != "-Enter New Index"]
    consolidated_df = consolidated_df[
        consolidated_df["subindex"] != "-Enter New Subindex"
    ]


    # selecting relevent columns
    consolidated_df = consolidated_df[columns.tolist() + ["theme", "index", "subindex"]]

    consolidated_df = consolidated_df.reset_index(drop=True)

    return consolidated_df
