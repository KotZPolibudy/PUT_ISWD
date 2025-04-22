from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """
    Function that loads dataset from csv file

    :param dataset_path: Path to dataset directory
    :return: Pandas Dataframe where every row represents single alternative, while every column represents single criterion
    """

    dataset = pd.read_csv(dataset_path / "dataset.csv", index_col=0)

    return dataset


def load_preference_information(dataset_path: Path) -> pd.DataFrame:
    """
    Function that loads preference information from csv file

    :param dataset_path: Path to dataset directory
    :return:
    """
    preferences = pd.read_csv(dataset_path / "preference.csv", index_col=0)

    return preferences


def merge_nodes(ranking: pd.DataFrame, first_index, second_index) -> None:
    ranking = ranking.copy()

    index_ = ranking.index.to_list()
    index_[first_index] = f"{ranking.index[first_index]}, {ranking.index[second_index]}"
    ranking.index = index_
    ranking.columns = index_

    ranking.drop(labels=ranking.index[second_index], inplace=True)
    ranking.drop(labels=ranking.index[second_index], axis=1, inplace=True)


def find_nodes_groups(ranking: pd.DataFrame) -> list[list[str]]:
    nodes = ranking.index.tolist()

    indifference_matrix = ranking & ranking.T
    edges = [
        (nodes[i], nodes[j])
        for i, j in np.stack(np.nonzero(indifference_matrix)).T.tolist()
        if i != j
    ]

    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)

    return list(nx.clique.find_cliques(g))


def display_ranking(ranking: pd.DataFrame, title: str) -> None:
    nodes_groups = find_nodes_groups(ranking)

    nodes = ranking.index.tolist()
    edges = [
        (nodes[i], nodes[j])
        for i, j in np.stack(np.nonzero(ranking)).T.tolist()
        if i != j
    ]

    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)

    names_mapping = {}

    for node_group in nodes_groups:
        first, *others = node_group

        for i in others:
            g.remove_node(i)

        names_mapping[first] = ",".join(node_group)

    g = nx.relabel_nodes(g, names_mapping)
    g = nx.transitive_reduction(g)

    layout = graphviz_layout(g, prog="dot")
    plt.title(title)
    nx.draw(
        g,
        layout,
        with_labels=True,
        arrows=False,
        node_shape="s",
        node_color="none",
        bbox=dict(facecolor="white", edgecolor="black"),
    )
    if not Path("output").exists():
        Path("output").mkdir()
    plt.savefig(f"output/{title}.png", format="png")
    plt.close()
