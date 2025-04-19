from pathlib import Path

import click
import numpy as np
import pandas as pd

from projekt2.electre_iii_pl.utils import (
    load_dataset,
    load_preference_information,
    display_ranking,
)


# TODO
def calculate_marginal_concordance_matrix(
    dataset: pd.DataFrame, preference_information: pd.DataFrame
) -> np.ndarray:
    """
    Function that calculates the marginal concordance matrix for all alternatives pairs and criterion available in dataset

    :param dataset: pandas dataframe representing dataset with alternatives as rows and criterion as columns
    :param preference_information: pandas dataframe with preference information
    :return: 3D numpy array with marginal concordance matrix with shape [number of alternatives, number of alternatives, number of criterion], where element with index [i, j, k] describe marginal concordance index between alternative i and alternative j on criterion k
    """
    raise NotImplementedError()


# TODO
def calculate_comprehensive_concordance_matrix(
    marginal_concordance_matrix: np.ndarray, preference_information: pd.DataFrame
) -> np.ndarray:
    """
    Function that calculates comprehensive concordance matrix for the given dataset

    :param marginal_concordance_matrix: 3D numpy array with marginal concordance matrix with shape [number of alternatives, number of alternatives, number of criterion], where element with index [i, j, k] describe marginal concordance index between alternative i and alternative j on criterion k
    :param preference_information: pandas dataframe with preference information
    :return: 2D numpy array with comprehensive concordance matrix with shape [number of alternatives, number of alternatives], where element with index [i, j] describe comprehensive concordance index between alternative i and alternative j
    """
    raise NotImplementedError()


# TODO
def calculate_marginal_discordance_matrix(
    dataset: pd.DataFrame, preference_information: pd.DataFrame
) -> np.ndarray:
    """
    Function that calculates the marginal discordance matrix for all alternatives pairs and criterion available in dataset

    :param dataset: pandas dataframe representing dataset with alternatives as rows and criterion as columns
    :param preference_information: pandas dataframe with preference information
    :return: 3D numpy array with marginal discordance matrix with shape [number of alternatives, number of alternatives, number of criterion], where element with index [i, j, k] describe marginal discordance index between alternative i and alternative j on criterion k
    """
    raise NotImplementedError()


# TODO
def calculate_credibility_index(
    comprehensive_concordance_matrix: np.ndarray,
    marginal_discordance_matrix: np.ndarray,
) -> np.ndarray:
    """
    Function that calculates the credibility index for the given comprehensive concordance matrix and marginal discordance matrix

    :param comprehensive_concordance_matrix: 2D numpy array with comprehensive concordance matrix. Every entry in the matrix [i, j] represents comprehensive concordance index between alternative i and alternative j
    :param marginal_discordance_matrix: 2D numpy array with marginal discordance matrix, Consecutive indices [i, j, k] describe first alternative, second alternative, criterion
    :return: 2D numpy array with credibility matrix with shape [number of alternatives, number of alternatives], where element with index [i, j] describe credibility index between alternative i and alternative j
    """
    raise NotImplementedError()


# TODO
def descending_distillation(
    credibility_index: np.ndarray,
    alternatives: pd.Index,
    alpha: float = -0.15,
    beta: float = 0.3,
) -> pd.DataFrame:
    """
    Function that calculates the descending distillation procedure

    :param credibility_index: 2D numpy array with credibility matrix with shape [number of alternatives, number of alternatives], where element with index [i, j] describe credibility index between alternative i and alternative j
    :param alternatives: index representing the alternative name in the corresponding position in preference matrix
    :param alpha: the parameter alpha for the s function
    :param beta: the parameter beta for the s function
    :return: descending ranking in a form of outranking matrix, as Dataframe where in index and columns are alternatives, i.e.
    1- if for the give pair [i, j] the alternative i is preferred over j or i is indifferent from j
    0- otherwise
    """
    raise NotImplementedError()


# TODO
def ascending_distillation(
    credibility_index: np.ndarray,
    alternatives: pd.Index,
    alpha: float = -0.15,
    beta: float = 0.3,
) -> pd.DataFrame:
    """
    Function that calculates the ascending distillation procedure

    :param credibility_index: 2D numpy array with credibility matrix with shape [number of alternatives, number of alternatives], where element with index [i, j] describe credibility index between alternative i and alternative j
    :param alternatives: index representing the alternative name in the corresponding position in preference matrix
    :param alpha: the parameter alpha for the s function
    :param beta: the parameter beta for the s function
    :return: ascending ranking in a form of outranking matrix, as Dataframe where in index and columns are alternatives, i.e.
    1- if for the give pair [i, j] the alternative i is preferred over j or i is indifferent from j
    0- otherwise
    """
    raise NotImplementedError()


# TODO
def create_final_ranking(
    descending_ranking: pd.DataFrame, ascending_ranking: pd.DataFrame
) -> pd.DataFrame:
    """
    Function that computes the final ranking from both ascending and descending ranking

    :param descending_ranking: dataframe representing descending ranking
    :param ascending_ranking: dataframe representing ascending ranking
    :return: final ranking in a form of outranking matrix, as Dataframe where in index and columns are alternatives, i.e.
    1- if for the give pair [i, j] the alternative i is preferred over j or i is indifferent from j
    0- otherwise
    """
    raise NotImplementedError()


@click.command()
@click.argument("dataset_path", type=click.Path(exists=True))
def main(dataset_path: str) -> None:
    dataset_path = Path(dataset_path)

    dataset = load_dataset(dataset_path)
    preference_information = load_preference_information(dataset_path)

    marginal_concordance_matrix = calculate_marginal_concordance_matrix(
        dataset, preference_information
    )
    comprehensive_concordance_matrix = calculate_comprehensive_concordance_matrix(
        marginal_concordance_matrix, preference_information
    )
    marginal_discordance_matrix = calculate_marginal_discordance_matrix(
        dataset, preference_information
    )
    credibility_index = calculate_credibility_index(
        comprehensive_concordance_matrix, marginal_discordance_matrix
    )

    descending_ranking = descending_distillation(credibility_index, dataset.index)
    display_ranking(descending_ranking, "Descending Ranking")

    ascending_ranking = ascending_distillation(credibility_index, dataset.index)
    display_ranking(ascending_ranking, "Ascending Ranking")

    final_ranking = create_final_ranking(descending_ranking, ascending_ranking)
    display_ranking(final_ranking, "Final Ranking")


if __name__ == "__main__":
    main()
