import json
from random import sample


def read_taxonomy():
    with open("topic-suggestion-interface-v3/taxonomy/taxonomy.json") as json_file:
        data = json.load(json_file)
        return data


def reformat_taxonomy(taxonomy):
    themes = taxonomy["themes"]
    indexes = taxonomy["indexes"]
    subindexes = taxonomy["subindexes"]

    new_taxonomy = {}

    for theme in themes:
        new_taxonomy[theme] = {}
        for index in sample(indexes, 3):
            new_taxonomy[theme][index] = []
            new_taxonomy[theme][index].extend(sample(subindexes, 3))
            new_taxonomy[theme][index].extend(["NA"])

    return new_taxonomy


def generate_label_chains(taxonomy):
    output = []
    for theme, indexes in taxonomy.items():
        for index, subindexes in indexes.items():
            for subindex in subindexes:
                output.append(f"{theme} > {index} > {subindex}")
    return output


def convert_chain_to_list(chain):
    return chain.split(" > ")


if __name__ == "__main__":
    print(generate_label_chains(reformat_taxonomy(read_taxonomy())))
