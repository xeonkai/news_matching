import pandas as pd
import streamlit as st
from functions.taxonomy_reader import convert_chain_to_list

# Function to consolidate grid responses into a dataframe


def consolidate_grid_responses(grid_responses, columns):
    consolidated_df = pd.DataFrame()

    for theme in grid_responses:
        sub_df = pd.DataFrame(grid_responses[theme]["selected_rows"])
        consolidated_df = pd.concat([consolidated_df, sub_df])

    consolidated_columns = consolidated_df.columns.tolist()

    def row_processer(row, columns):

        # by label
        if row[columns.index("suggested_label")] != "-Enter New Label":
            chain_list = convert_chain_to_list(
                row[columns.index("suggested_label")])
            # row[columns.index("theme")] = chain_list[0]
            row[columns.index("theme")] = chain_list[0]
            row[columns.index("index")] = chain_list[1]

        elif row[columns.index("suggested_label")] == "-Enter New Label":
            # if row[columns.index("theme")] == "-Enter New Theme" and row[columns.index("new theme")] != "":
            #     row.replace(row[columns.index("theme")],
            #                 row[columns.index("new theme")], inplace=True)
            if row[columns.index("theme")] == "-Enter New Theme" and row[columns.index("new theme")] != "":
                row.replace(row[columns.index("theme")],
                            row[columns.index("new theme")], inplace=True)
            if row[columns.index("index")] == "-Enter New Index" and row[columns.index("new index")] != "":
                row.replace(
                    row[columns.index("index")], row[columns.index(
                        "new index")], inplace=True
                )

        return row

    consolidated_df = consolidated_df.apply(
        lambda x: row_processer(x, consolidated_columns), axis=1)

    # Filtering out rows without new theme/index/subindex

    # consolidated_df = consolidated_df[consolidated_df["theme"]
    #                                   != "-Enter New Theme"]
    consolidated_df = consolidated_df[consolidated_df["theme"]
                                      != "-Enter New Theme"]
    consolidated_df = consolidated_df[
        consolidated_df["index"] != "-Enter New Index"
    ]

    # Filtering out blank theme/index/subindex
    # consolidated_df = consolidated_df[consolidated_df["theme"]
    #                                   != ""]
    consolidated_df = consolidated_df[consolidated_df["theme"]
                                      != ""]
    consolidated_df = consolidated_df[
        consolidated_df["index"] != ""
    ]

    # selecting relevent columns
    consolidated_df = consolidated_df[["headline", "summary", "link", "domain", "facebook_interactions", "theme", "index", "subindex"]]

    consolidated_df = consolidated_df.reset_index(drop=True)

    return consolidated_df
