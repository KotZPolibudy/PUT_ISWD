from pathlib import Path

import click
import numpy as np
import pandas as pd

from promethee_pl.utils import (
    load_dataset,
    load_preference_information,
    display_ranking,
)


# TODO
def calculate_marginal_preference_matrix(
    dataset: pd.DataFrame, preference_information: pd.DataFrame
) -> np.ndarray:
    """
    Function that calculates the marginal preference matrix all alternatives pairs and criterion available in dataset

    :param dataset: difference between compared alternatives
    :param preference_information: preference information
    :return: 3D numpy array with marginal preference matrix on every parser, Consecutive indices [i, j, k] describe first alternative, second alternative, parser
    """
    n_alternatives = dataset.shape[0]
    n_criteria = dataset.shape[1]
    
    marginal_pref_matrix = np.zeros((n_alternatives, n_alternatives, n_criteria))
    
    for k, criterion in enumerate(dataset.columns):
        qk = preference_information.loc[criterion, "q"]
        pk = preference_information.loc[criterion, "p"]
        crit_type = preference_information.loc[criterion, "type"]

        for i in range(n_alternatives):
            for j in range(n_alternatives):
                if i == j:
                    marginal_pref_matrix[i, j, k] = 0
                    continue
                
                if crit_type == "gain":
                    dik = dataset.iloc[i, k] - dataset.iloc[j, k]
                elif crit_type == "cost":
                    dik = dataset.iloc[j, k] - dataset.iloc[i, k]
                else:
                    raise ValueError(f"Unknown criterion type: {crit_type}")
                    
                if dik >= pk:
                    pref = 1
                elif dik <= qk:
                    pref = 0
                else:
                    pref = (dik - qk) / (pk - qk)

                marginal_pref_matrix[i, j, k] = pref

    return marginal_pref_matrix


# TODO
def calculate_comprehensive_preference_index(
    marginal_preference_matrix: np.ndarray, preference_information: pd.DataFrame
) -> np.ndarray:
    """
    Function that calculates comprehensive preference index for the given dataset

    :param marginal_preference_matrix: 3D numpy array with marginal preference matrix on every parser, Consecutive indices [i, j, k] describe first alternative, second alternative, criterion
    :param preference_information: Padnas dataframe containing preference information
    :return: 2D numpy array with marginal preference matrix. Every entry in the matrix [i, j] represents comprehensive preference index between alternative i and alternative j
    """
    n_alternatives = marginal_preference_matrix.shape[0]
    weights = preference_information["k"].values
    weights = weights / np.sum(weights)
    
    comprehensive_index = np.zeros((n_alternatives, n_alternatives))

    for i in range(n_alternatives):
        for j in range(n_alternatives):
            if i == j:
                continue
            else:
                comprehensive_index[i, j] = np.sum(marginal_preference_matrix[i, j, :] * weights)

    return comprehensive_index


# TODO
def calculate_positive_flow(
    comprehensive_preference_matrix: np.ndarray, alternatives: pd.Index
) -> pd.Series:
    """
    Function that calculates the positive flow value for the given preference matrix and corresponding index

    :param comprehensive_preference_matrix: 2D numpy array with marginal preference matrix. Every entry in the matrix [i, j] represents comprehensive preference index between alternative i and alternative j
    :param alternatives: index representing the alternative name in the corresponding position in preference matrix
    :return: series representing positive flow values for the given preference matrix
    """
    negative_flows = comprehensive_preference_matrix.sum(axis=1)
    return pd.Series(negative_flows, index=alternatives)


# TODO
def calculate_negative_flow(
    comprehensive_preference_matrix: np.ndarray, alternatives: pd.Index
) -> pd.Series:
    """
    Function that calculates the negative flow value for the given preference matrix and corresponding index

    :param comprehensive_preference_matrix: 2D numpy array with marginal preference matrix. Every entry in the matrix [i, j] represents comprehensive preference index between alternative i and alternative j
    :param alternatives: index representing the alternative name in the corresponding position in preference matrix
    :return: series representing negative flow values for the given preference matrix
    """
    negative_flows = comprehensive_preference_matrix.sum(axis=0)
    return pd.Series(negative_flows, index=alternatives)


# TODO
def calculate_net_flow(positive_flow: pd.Series, negative_flow: pd.Series) -> pd.Series:
    """
    Function that calculates the net flow value for the given positive and negative flow

    :param positive_flow: series representing positive flow values for the given preference matrix
    :param negative_flow: series representing negative flow values for the given preference matrix
    :return: series representing net flow values for the given preference matrix
    """
    return positive_flow - negative_flow


# TODO
def create_partial_ranking(
    positive_flow: pd.Series, negative_flow: pd.Series
) -> pd.DataFrame:
    """
    Function that aggregates positive and negative flow to a partial ranking (from Promethee I)

    :param positive_flow: series representing positive flow values for the given preference matrix
    :param negative_flow: series representing negative flow values for the given preference matrix
    :return: partial ranking in a form of outranking matrix, as Dataframe where in index and columns are alternatives, i.e.
    1- if for the give pair [i, j] the alternative i is preferred over j or i is indifferent from j
    0- otherwise
    """
    alternatives = positive_flow.index
    n = len(alternatives)

    outranking_matrix = pd.DataFrame(
        np.zeros((n, n), dtype=int),
        index=alternatives,
        columns=alternatives
    )

    for i in alternatives:
        for j in alternatives:
            if i == j:
                outranking_matrix.loc[i, j] = 1
            else:
                if (positive_flow[i] > positive_flow[j] and
                    negative_flow[i] < negative_flow[j]):
                    outranking_matrix.loc[i, j] = 1
                elif (positive_flow[i] == positive_flow[j] and
                      negative_flow[i] == negative_flow[j]):
                    outranking_matrix.loc[i, j] = 1

    return outranking_matrix


# TODO
def create_complete_ranking(net_flow: pd.Series) -> pd.DataFrame:
    """
    Function that aggregates positive and negative flow to a complete ranking (from Promethee II)
    :param net_flow: series representing net flow values for the given preference matrix
    :return: complete ranking in a form of outranking matrix, as Dataframe where in index and columns are alternatives, i.e.
    1- if for the give pair [i, j] the alternative i is preferred over j or i is indifferent from j
    0- otherwise
    """
    outranking_matrix = pd.DataFrame(
        0, index=net_flow.index, columns=net_flow.index
    )
    np.fill_diagonal(outranking_matrix.values, 1)

    for i in range(len(net_flow)):
        for j in range(len(net_flow)):
            if i == j:
                continue
            if net_flow.iloc[i] > net_flow.iloc[j]:
                outranking_matrix.loc[net_flow.index[i], net_flow.index[j]] = 1
                outranking_matrix.loc[net_flow.index[j], net_flow.index[i]] = 0
            elif net_flow.iloc[i] < net_flow.iloc[j]:
                outranking_matrix.loc[net_flow.index[i], net_flow.index[j]] = 0
                outranking_matrix.loc[net_flow.index[j], net_flow.index[i]] = 1
            else:
                outranking_matrix.loc[net_flow.index[i], net_flow.index[j]] = 1
                outranking_matrix.loc[net_flow.index[j], net_flow.index[i]] = 1

    return outranking_matrix


@click.command()
@click.argument("dataset_path", type=click.Path(exists=True))
def main(dataset_path: str) -> None:
    dataset_path = Path(dataset_path)

    dataset = load_dataset(dataset_path)
    print(f"Dataset loaded:\n{dataset.head()}")

    preference_information = load_preference_information(dataset_path)
    print(f"Preference information loaded:\n{preference_information}")

    marginal_preference_matrix = calculate_marginal_preference_matrix(
        dataset, preference_information
    )
    print(f"Marginal preference matrix calculated:\n{marginal_preference_matrix}")

    comprehensive_preference_matrix = calculate_comprehensive_preference_index(
        marginal_preference_matrix, preference_information
    )
    print(f"Comprehensive preference matrix calculated:\n{comprehensive_preference_matrix}")

    positive_flow = calculate_positive_flow(
        comprehensive_preference_matrix, dataset.index
    )
    print(f"Positive flow calculated:\n{positive_flow}")

    negative_flow = calculate_negative_flow(
        comprehensive_preference_matrix, dataset.index
    )
    print(f"Negative flow calculated:\n{negative_flow}")

    assert positive_flow.index.equals(negative_flow.index)

    partial_ranking = create_partial_ranking(positive_flow, negative_flow)
    print(f"Partial ranking (Promethee I) created:\n{partial_ranking}")

    net_flow = calculate_net_flow(positive_flow, negative_flow)
    print(f"Net flow calculated:\n{net_flow}")

    complete_ranking = create_complete_ranking(net_flow)
    print(f"Complete ranking (Promethee II) created:\n{complete_ranking}")

    display_ranking(partial_ranking, "Promethee I")
    display_ranking(complete_ranking, "Promethee II")
    

if __name__ == "__main__":
    main()
