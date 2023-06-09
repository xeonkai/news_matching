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
            row[columns.index("Predicted_theme_and_index")])
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

    for theme in grid_responses:
        sub_df = pd.DataFrame(grid_responses[theme]["selected_rows"])
        consolidated_df = pd.concat([consolidated_df, sub_df])

    consolidated_columns = consolidated_df.columns.tolist()

    consolidated_df = consolidated_df.apply(
        lambda x: row_processer(x, consolidated_columns), axis=1)

    # Filtering out rows without new theme/index/subindex
    consolidated_df = consolidated_df[consolidated_df["theme"]
                                      != "-Enter New Theme"]
    consolidated_df = consolidated_df[
        consolidated_df["index"] != "-Enter New Index"
    ]

    # Filtering out blank theme/index/subindex
    consolidated_df = consolidated_df[consolidated_df["theme"]
                                      != ""]
    consolidated_df = consolidated_df[
        consolidated_df["index"] != ""
    ]

    # selecting relevent columns
    consolidated_df = consolidated_df[["headline", "summary", "link",
                                       "domain", "facebook_interactions", "theme", "index", "subindex"]]

    consolidated_df = consolidated_df.reset_index(drop=True)

    return consolidated_df
