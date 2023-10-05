import pandas as pd
from functions.taxonomy_reader import convert_chain_to_list


def row_processer(row, columns):
    """Function to process a row of a dataframe

    Args:
        row (pandas.core.series.Series): Row of a dataframe
        columns (list): List of column names of the dataframe

    Returns:
        pandas.core.series.Series: Processed row of a dataframe

    """

    if row[columns.index("Predicted_theme_and_index")] != "-Enter New Label":
        chain_list = convert_chain_to_list(
            row[columns.index("Predicted_theme_and_index")]
        )
        row[columns.index("theme")] = chain_list[0]
        row[columns.index("index")] = chain_list[1]
    return row


# Function to consolidate grid responses into a dataframe


def consolidate_grid_responses(grid_responses):
    """Function to consolidate grid responses into a dataframe

    Args:
        grid_responses (dict): Dictionary of grid responses

    Returns:
        pandas.core.frame.DataFrame: Dataframe of consolidated grid responses

    """
    consolidated_df = pd.DataFrame()

    if grid_responses != {}:
        for theme in grid_responses:
            sub_df = pd.DataFrame(grid_responses[theme]["selected_rows"])
            consolidated_df = pd.concat([consolidated_df, sub_df])

        consolidated_columns = consolidated_df.columns.tolist()

        consolidated_df = consolidated_df.apply(
            lambda x: row_processer(x, consolidated_columns), axis=1
        )

        # Filtering out rows without new theme/index/subindex
        consolidated_df = consolidated_df[
            consolidated_df["theme"] != "-Enter New Theme"
        ]
        consolidated_df = consolidated_df[
            consolidated_df["index"] != "-Enter New Index"
        ]

        # Filtering out blank theme/index/subindex
        consolidated_df = consolidated_df[consolidated_df["theme"] != ""]
        consolidated_df = consolidated_df[consolidated_df["index"] != ""]

        # selecting relevent columns
        consolidated_df = consolidated_df[
            [
                "published",
                "headline",
                "summary",
                "link",
                "domain",
                "facebook_interactions",
                "theme",
                "index",
                "subindex",
                "source"
            ]
        ]

        consolidated_df = consolidated_df.reset_index(drop=True)

    return consolidated_df


def extract_unlabelled_articles(labelled_df, all_data_df):
    unlabelled_articles = pd.DataFrame()

    # get rows from all_data_df that are not in labelled_df based on link using apply
    unlabelled_articles = all_data_df[~all_data_df["link"].isin(labelled_df["link"])]

    unlabelled_articles = unlabelled_articles.reset_index(drop=True)

    unlabelled_articles["theme"] = ""
    unlabelled_articles["index"] = ""
    unlabelled_articles["subindex"] = ""
    unlabelled_articles["label"] = ""

    unlabelled_articles = unlabelled_articles[
        [
            "published",
            "headline",
            "summary",
            "link",
            "domain",
            "facebook_interactions",
            "theme",
            "index",
            "subindex",
            "source",
            "label"
        ]
    ]

    # sort by facebook interactions

    unlabelled_articles = unlabelled_articles.sort_values(
        by=["facebook_interactions"], ascending=False
    )

    return unlabelled_articles
