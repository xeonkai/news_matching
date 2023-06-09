import json


def read_taxonomy():
    """Function to read taxonomy from JSON file
    
    Returns:
        dict: Taxonomy
    """

    with open("topic-suggestion-interface-v3/taxonomy/taxonomy_testing.json") as json_file:
        data = json.load(json_file)
        return data


def process_taxonomy(taxonomy):
    """Function to process taxonomy
    
    Args:
        taxonomy (dict): Taxonomy
        
    Returns:
        dict: Processed taxonomy
    """

    for index in taxonomy.keys():
        for index in taxonomy[index].keys():
            taxonomy[index][index].append("NA")
    return taxonomy


def generate_label_chains(taxonomy):
    """Function to generate label chains from taxonomy
    
    Args:
        taxonomy (dict): Taxonomy
        
    Returns:
        list: Label chains
    """

    output = []
    for indexes, subindexes in taxonomy.items():
        for subindex in subindexes:
            output.append(f"{indexes} > {subindex}")
    return output


def convert_chain_to_list(chain):
    """Function to convert label chain to list
    
    Args:
        chain (str): Label chain
        
    Returns:
        list: List of labels
    """

    return chain.split(" > ")
