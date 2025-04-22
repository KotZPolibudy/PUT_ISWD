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
                    diff = a_val - b_val
                elif crit_type == "cost":
                    diff = b_val - a_val
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
            if i == j:
                credibility_matrix[i, j] = 0
                continue
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
    P = pd.DataFrame(0, index=alternatives, columns=alternatives)
    alt_map = {alt: i for i, alt in enumerate(alternatives)}

    while len(A) > 0:
        indices = [alt_map[a] for a in A]
        submatrix = credibility_index[np.ix_(indices, indices)]

        lambda_k = np.max(submatrix[np.triu_indices(len(A), k=1)])
        sk = alpha * lambda_k + beta
        lambda_k_next = np.max(np.where(submatrix < lambda_k - sk, submatrix, 0))

        if lambda_k == 0:
            P.loc[A, A] = 1
            break

        strength = pd.Series(0, index=A)
        weakness = pd.Series(0, index=A)

        for i, ai in enumerate(A):
            for j, aj in enumerate(A):
                if i == j:
                    continue
                pij = submatrix[i, j]
                pji = submatrix[j, i]
                if pij > lambda_k_next and pij > pji + alpha * pij + beta:
                    strength[ai] += 1
                    weakness[aj] += 1

        quality = strength - weakness
        Dk1 = quality[quality == quality.max()].index

        # destylacja wewnętrzna
        if len(Dk1) > 1:
            Dh = Dk1.copy()
            lambda_kh = lambda_k_next
            while True:
                matrix_indices = [alt_map[a] for a in Dh]
                sub_submatrix = credibility_index[np.ix_(matrix_indices, matrix_indices)]
                skh = alpha * lambda_kh + beta
                lambda_kh1 = np.max(np.where(sub_submatrix < lambda_kh - skh, sub_submatrix, 0))

                strength = pd.Series(0, index=Dh)
                weakness = pd.Series(0, index=Dh)

                for i, ai in enumerate(Dh):
                    for j, aj in enumerate(Dh):
                        if i == j:
                            continue
                        pij = credibility_index[alt_map[ai], alt_map[aj]]
                        pji = credibility_index[alt_map[aj], alt_map[ai]]
                        if pij > lambda_kh1 and pij > pji + alpha * pij + beta:
                            strength[ai] += 1
                            weakness[aj] += 1

                quality = strength - weakness
                new_Dh = quality[quality == quality.max()].index

                if len(new_Dh) == 1 or lambda_kh1 == 0:
                    DkF = new_Dh
                    break

                Dh = new_Dh
                lambda_kh = lambda_kh1
        else:
            DkF = Dk1

        for a in DkF:
            for b in A:
                if a == b:
                    P.loc[a, b] = 1
                elif b not in DkF:
                    P.loc[a, b] = 1
                    P.loc[b, a] = 0
            for b in DkF:
                P.loc[a, b] = 1

        A = A.difference(DkF)

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
    P = pd.DataFrame(0, index=alternatives, columns=alternatives)
    alt_map = {alt: i for i, alt in enumerate(alternatives)}

    while len(A) > 0:
        indices = [alt_map[a] for a in A]
        submatrix = credibility_index[np.ix_(indices, indices)]

        if len(A) == 1:
            P.loc[A, A] = 1
            break
        
        lambda_k = np.max(submatrix[np.triu_indices(len(A), k=1)])
        sk = alpha * lambda_k + beta
        lambda_k_next = np.max(np.where(submatrix < lambda_k - sk, submatrix, 0))

        if lambda_k == 0:
            P.loc[A, A] = 1
            break

        strength = pd.Series(0, index=A)
        weakness = pd.Series(0, index=A)

        for i, ai in enumerate(A):
            for j, aj in enumerate(A):
                if i == j:
                    continue
                pij = submatrix[i, j]
                pji = submatrix[j, i]
                if pij > lambda_k_next and pij > pji + alpha * pij + beta:
                    strength[ai] += 1
                    weakness[aj] += 1

        quality = strength - weakness
        Dk1 = quality[quality == quality.min()].index

        # destylacja wewnętrzna
        if len(Dk1) > 1:
            Dh = Dk1.copy()
            lambda_kh = lambda_k_next
            while True:
                matrix_indices = [alt_map[a] for a in Dh]
                sub_submatrix = credibility_index[np.ix_(matrix_indices, matrix_indices)]
                skh = alpha * lambda_kh + beta
                lambda_kh1 = np.max(np.where(sub_submatrix < lambda_kh - skh, sub_submatrix, 0))

                strength = pd.Series(0, index=Dh)
                weakness = pd.Series(0, index=Dh)

                for i, ai in enumerate(Dh):
                    for j, aj in enumerate(Dh):
                        if i == j:
                            continue
                        pij = credibility_index[alt_map[ai], alt_map[aj]]
                        pji = credibility_index[alt_map[aj], alt_map[ai]]
                        if pij > lambda_kh1 and pij > pji + alpha * pij + beta:
                            strength[ai] += 1
                            weakness[aj] += 1

                quality = strength - weakness
                new_Dh = quality[quality == quality.min()].index

                if len(new_Dh) == 1 or lambda_kh1 == 0:
                    DkF = new_Dh
                    break

                Dh = new_Dh
                lambda_kh = lambda_kh1
        else:
            DkF = Dk1
            
        for a in DkF:
            for b in A:
                if a == b:
                    P.loc[a, b] = 1
                elif b not in DkF:
                    P.loc[b, a] = 1
                    P.loc[a, b] = 0
            for b in DkF:
                P.loc[a, b] = 1

        A = A.difference(DkF)

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
    print(f"Dataset loaded with {dataset.shape[0]} alternatives and {dataset.shape[1]} criteria.")
    print(f"Dataset:\n{dataset}")

    preference_information = load_preference_information(dataset_path)
    print(f"Preference information loaded with {preference_information.shape[0]} criteria.")
    print(f"Preference information:\n{preference_information}")

    marginal_concordance_matrix = calculate_marginal_concordance_matrix(dataset, preference_information)
    print(f"Marginal concordance matrix:\n{marginal_concordance_matrix}")

    comprehensive_concordance_matrix = calculate_comprehensive_concordance_matrix(marginal_concordance_matrix, preference_information)
    print(f"Comprehensive concordance matrix:\n{comprehensive_concordance_matrix}")

    marginal_discordance_matrix = calculate_marginal_discordance_matrix(dataset, preference_information)
    print(f"Marginal discordance matrix:\n{marginal_discordance_matrix}")

    credibility_index = calculate_credibility_index(comprehensive_concordance_matrix, marginal_discordance_matrix)
    print(f"Credibility index:\n{credibility_index}")

    descending_ranking = descending_distillation(credibility_index, dataset.index)
    print(f"Descending ranking:\n{descending_ranking}")
    
    ascending_ranking = ascending_distillation(credibility_index, dataset.index)
    print(f"Ascending ranking:\n{ascending_ranking}")

    final_ranking = create_final_ranking(descending_ranking, ascending_ranking)
    print(f"Final ranking:\n{final_ranking}")
    
    display_ranking(descending_ranking, "Descending Ranking")
    display_ranking(ascending_ranking, "Ascending Ranking")
    display_ranking(final_ranking, "Final Ranking")


if __name__ == "__main__":
    main()
