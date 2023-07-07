import json


def read_taxonomy():
    """Function to read taxonomy from JSON file

    Returns:
        dict: Taxonomy
    """

    with open("app/taxonomy/taxonomy_testing.json") as json_file:
        data = json.load(json_file)
        return data


def convert_chain_to_list(chain):
    """Function to convert label chain to list

    Args:
        chain (str): Label chain

    Returns:
        list: List of labels
    """

    return chain.split(" > ")
