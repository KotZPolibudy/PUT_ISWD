from pathlib import Path

import click
import numpy as np
import pandas as pd

from electre_iii_pl.utils import (
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
    if preference_information.index.name != "Criterion":
        preference_information = preference_information.set_index("Criterion")
    
    num_alternatives = dataset.shape[0]
    num_criteria = dataset.shape[1]
    
    concordance_matrix = np.zeros((num_alternatives, num_alternatives, num_criteria))
    
    for k, criterion in enumerate(dataset.columns):
        qk = preference_information.loc[criterion, "q"]
        pk = preference_information.loc[criterion, "p"]
        crit_type = preference_information.loc[criterion, "type"]
        
        for i in range(num_alternatives):
            for j in range(num_alternatives):
                ga = dataset.iloc[i, k]
                gb = dataset.iloc[j, k]
                
                if crit_type == "gain":
                    dk = ga - gb
                elif crit_type == "cost":
                    dk = gb - ga
                else:
                    raise ValueError(f"Unknown criterion type: {crit_type}")
                
                if dk >= -qk:
                    ck = 1.0
                elif dk <= -pk:
                    ck = 0.0
                else:
                    ck = (pk + dk) / (pk - qk)
                
                concordance_matrix[i, j, k] = ck
    
    return concordance_matrix


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
    if preference_information.index.name != "Criterion":
        preference_information = preference_information.set_index("Criterion")
    
    num_alternatives = marginal_concordance_matrix.shape[0]
    num_criteria = marginal_concordance_matrix.shape[2]
    
    weights = preference_information["k"].values.astype(float)
    total_weight = weights.sum()

    comprehensive_matrix = np.zeros((num_alternatives, num_alternatives))

    for i in range(num_alternatives):
        for j in range(num_alternatives):
            concordance = 0.0
            for k in range(num_criteria):
                concordance += weights[k] * marginal_concordance_matrix[i, j, k]
            comprehensive_matrix[i, j] = concordance / total_weight

    return comprehensive_matrix


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
    if preference_information.index.name != "Criterion":
        preference_information = preference_information.set_index("Criterion")
    
    alternatives = dataset.index.tolist()
    criteria = dataset.columns.tolist()
    num_alternatives = len(alternatives)
    num_criteria = len(criteria)

    discordance_matrix = np.zeros((num_alternatives, num_alternatives, num_criteria))

    for k, criterion in enumerate(criteria):
        q = preference_information.loc[criterion, "q"]
        p = preference_information.loc[criterion, "p"]
        v = preference_information.loc[criterion, "v"]
        crit_type = preference_information.loc[criterion, "type"]

        for i in range(num_alternatives):
            for j in range(num_alternatives):
                a_val = dataset.iloc[i][criterion]
                b_val = dataset.iloc[j][criterion]

                if crit_type == "gain":
                    diff = b_val - a_val
                elif crit_type == "cost":
                    diff = a_val - b_val
                else:
                    raise ValueError(f"Unknown criterion type: {crit_type}")

                if diff <= -v:
                    discordance = 1.0
                elif diff >= -p:
                    discordance = 0.0
                else:
                    discordance = (-diff - p) / (v - p)

                discordance_matrix[i, j, k] = discordance

    return discordance_matrix


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
    num_alternatives = comprehensive_concordance_matrix.shape[0]
    credibility_matrix = np.zeros((num_alternatives, num_alternatives))

    for i in range(num_alternatives):
        for j in range(num_alternatives):
            C_ab = comprehensive_concordance_matrix[i, j]
            product = 1.0

            for k in range(marginal_discordance_matrix.shape[2]):
                D_ab_k = marginal_discordance_matrix[i, j, k]

                if D_ab_k > C_ab:
                    product *= (1 - D_ab_k) / (1 - C_ab)
            
            credibility_matrix[i, j] = C_ab * product
    return credibility_matrix


# TODO
def descending_distillation(
    credibility_index: np.ndarray,
    alternatives: pd.Index,
    alpha: float = -0.15,
    beta: float = 0.3
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
    A = alternatives.copy()
    P = pd.DataFrame(index=alternatives, columns=alternatives, data=0)

    while len(A) > 0:
        non_diag_elements = credibility_index[np.triu_indices_from(credibility_index, k=1)]
        lambda_k = np.max(non_diag_elements)

        sk = alpha * lambda_k + beta
        lambda_k_next = np.max(np.where(credibility_index < lambda_k - sk, credibility_index, 0))

        if lambda_k == 0:
            P.loc[A, A] = 1
            break

        strength = pd.Series(0, index=A)
        weakness = pd.Series(0, index=A)
        quality = pd.Series(0, index=A)

        for i in range(len(A)):
            for j in range(len(A)):
                if i == j:
                    continue
                a, b = A[i], A[j]
                if credibility_index[i, j] > lambda_k_next and credibility_index[i, j] > credibility_index[j, i] + alpha * credibility_index[i, j] + beta:
                    P.loc[a, b] = 1
                    strength[a] += 1
                    weakness[b] += 1

        quality = strength - weakness
        Dk1 = quality[quality == quality.max()].index

        if len(Dk1) > 1:
            lambda_kh = lambda_k_next
            Dh_prev = Dk1.copy()

            while True:
                skh = alpha * lambda_kh + beta
                matrix_indices = [alternatives.get_loc(alt) for alt in Dh_prev]
                submatrix = credibility_index[np.ix_(matrix_indices, matrix_indices)]

                non_diag = submatrix[np.triu_indices(len(Dh_prev), k=1)]
                lambda_kh1 = np.max(non_diag[non_diag < lambda_kh - skh], initial=0)

                if lambda_kh1 == 0:
                    DkF = Dh_prev
                    break

                strength = pd.Series(0, index=Dh_prev)
                weakness = pd.Series(0, index=Dh_prev)

                for i in range(len(Dh_prev)):
                    for j in range(len(Dh_prev)):
                        if i == j:
                            continue
                        a, b = Dh_prev[i], Dh_prev[j]
                        val_ab = credibility_index[alternatives.get_loc(a), alternatives.get_loc(b)]
                        val_ba = credibility_index[alternatives.get_loc(b), alternatives.get_loc(a)]

                        if val_ab > lambda_kh1 and val_ab > val_ba + alpha * val_ab + beta:
                            strength[a] += 1
                            weakness[b] += 1

                quality = strength - weakness
                Dh = quality[quality == quality.max()].index

                if len(Dh) == 1:
                    DkF = Dh
                    break

                lambda_kh = lambda_kh1
                Dh_prev = Dh
        else:
            DkF = Dk1

        P.loc[DkF, DkF] = 1
        A = A.difference(DkF)

        if len(A) == 0:
            break

        indices = [alternatives.get_loc(alt) for alt in A]
        credibility_index = credibility_index[np.ix_(indices, indices)]
        alternatives = A

    return P


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
    A = alternatives.copy()
    P = pd.DataFrame(index=alternatives, columns=alternatives, data=0)
    while len(A) > 0:
        non_diag_elements = credibility_index[np.triu_indices_from(credibility_index, k=1)]
        if non_diag_elements.size == 0:
            lambda_k = 0
        else:
            lambda_k = np.max(non_diag_elements)

        sk = alpha * lambda_k + beta
        lambda_k_next = np.max(np.where(credibility_index < lambda_k - sk, credibility_index, 0))

        if lambda_k == 0:
            P.loc[A, A] = 1
            break

        strength = pd.Series(0, index=A)
        weakness = pd.Series(0, index=A)
        quality = pd.Series(0, index=A)

        for i in range(len(A)):
            for j in range(len(A)):
                if i == j:
                    continue
                a, b = A[i], A[j]
                if credibility_index[i, j] > lambda_k_next and credibility_index[i, j] > credibility_index[j, i] + alpha * credibility_index[i, j] + beta:
                    P.loc[a, b] = 1
                    strength[a] += 1
                    weakness[b] += 1

        quality = strength - weakness
        Dk1 = quality[quality == quality.min()].index

        if len(Dk1) > 1:
            lambda_kh = lambda_k_next
            Dh_prev = Dk1.copy()

            while True:
                skh = alpha * lambda_kh + beta
                matrix_indices = [alternatives.get_loc(alt) for alt in Dh_prev]
                submatrix = credibility_index[np.ix_(matrix_indices, matrix_indices)]

                non_diag = submatrix[np.triu_indices(len(Dh_prev), k=1)]
                filtered = non_diag[non_diag < lambda_kh - skh]
                lambda_kh1 = np.max(filtered) if filtered.size > 0 else 0

                if lambda_kh1 == 0:
                    DkF = Dh_prev
                    break

                strength = pd.Series(0, index=Dh_prev)
                weakness = pd.Series(0, index=Dh_prev)

                for i in range(len(Dh_prev)):
                    for j in range(len(Dh_prev)):
                        if i == j:
                            continue
                        a, b = Dh_prev[i], Dh_prev[j]
                        val_ab = credibility_index[alternatives.get_loc(a), alternatives.get_loc(b)]
                        val_ba = credibility_index[alternatives.get_loc(b), alternatives.get_loc(a)]

                        if val_ab > lambda_kh1 and val_ab > val_ba + alpha * val_ab + beta:
                            strength[a] += 1
                            weakness[b] += 1

                quality = strength - weakness
                Dh = quality[quality == quality.min()].index

                if len(Dh) == 1:
                    DkF = Dh
                    break

                lambda_kh = lambda_kh1
                Dh_prev = Dh
        else:
            DkF = Dk1

        P.loc[DkF, DkF] = 1
        A = A.difference(DkF)

        if len(A) == 0:
            break

        indices = [alternatives.get_loc(alt) for alt in A]
        credibility_index = credibility_index[np.ix_(indices, indices)]
        alternatives = A

    return P


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
    final_ranking = descending_ranking & ascending_ranking
    return final_ranking.astype(int)


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
