import numpy as np
from pulp import LpProblem, LpVariable, LpMaximize, LpMinimize, lpSum, value, LpStatus
import random
import matplotlib.pyplot as plt
import networkx as nx
from copy import deepcopy

R1 = [i for i in range(1, 28) if (i - 1) % 9 < 3]
R2 = [i for i in range(1, 28) if 3 <= (i - 1) % 9 < 6]
R3 = [i for i in range(1, 28) if 6 <= (i - 1) % 9 < 9]

S1 = list(range(1,10))
S2 = list(range(10,19))
S3 = list(range(19,28))

F1 = [i for i in range(1, 28) if i%3 == 1]
F2 = [i for i in range(1, 28) if i%3 == 2]
F3 = [i for i in range(1, 28) if i%3 == 0]

def read_data(file_path: str) -> list[str]:
    with open(file_path, 'r') as file:
        return file.readlines()

def process_data(data: list[str]) -> tuple[list[str], dict[str, list[float]], list[list[str]]]:
    data = [line.strip().split(',') for line in data]
    criterions = data[0][1:]
    alternatives = data[1:]
    data = np.array([line[1:] for line in alternatives]).T
    data = {criterion: list(map(float, values)) for criterion, values in zip(criterions, data)}
    return criterions, data, alternatives

def get_data(file_path: str) -> tuple[list[str], dict[str, list[float]], list[list[str]]]:
    return process_data(read_data(file_path))

def get_extremes(data: dict[str, list[float]]) -> dict[str, tuple[float, float]]:
    return {criterion: (min(values), max(values)) for criterion, values in data.items()}

def prepare_alternatives(alternatives: list[list[str]], criterions: list[str]) -> dict[str, dict[str, str]]:
    return {
        f"Alternative_{alt[0]}": {criterion: value for criterion, value in zip(criterions, alt[1:])}
        for alt in alternatives
    }

def uta(data: dict[str, list[float]], alternatives: dict[str, dict[str, float]], reference_pairs) -> tuple[LpProblem, dict[str, dict[str, LpVariable]], dict[str, LpVariable]]:
    prob, criterion_vars, alternative_utilities = create_problem(data, alternatives, LpMaximize)
    epsilon = LpVariable("epsilon", lowBound=1e-6)
    prob += epsilon, "Maximize_Epsilon"
    
    for a1, a2 in reference_pairs:
        prob += alternative_utilities[a1] >= alternative_utilities[a2] + epsilon, f"Reference_{a1}_{a2}"

    return prob, criterion_vars, alternative_utilities

def uta_gms(data: dict[str, list[float]], alternatives: dict[str, dict[str, float]], reference_pairs) -> tuple[LpProblem, dict[str, dict[str, LpVariable]]]:
    necessary_prefs = []
    possible_prefs = []
    alt_keys = list(alternatives.keys())
    for i in range(len(alt_keys)):
        for j in range(len(alt_keys)):
            if i == j:
                continue
            a1, a2 = alt_keys[i], alt_keys[j]
            prob, criterion_vars, alternative_utilities = create_problem(data, alternatives, LpMinimize)
            epsilon = 0.01
            for b1, b2 in reference_pairs:
                prob += alternative_utilities[b1] >= alternative_utilities[b2] + epsilon, f"Reference_{b1}_{b2}"
            sigma = LpVariable("sigma", lowBound=-100.0)
            prob += sigma, "Minimize_Sigma"
            prob += sigma == alternative_utilities[a1] - alternative_utilities[a2], f"Diff_{a1}_{a2}"
            prob.solve()
            if value(sigma) >= 0:
                necessary_prefs.append((a1, a2))
    for i in range(len(alt_keys)):
        for j in range(len(alt_keys)):
            if i == j:
                continue
            a1, a2 = alt_keys[i], alt_keys[j]
            prob, criterion_vars, alternative_utilities = create_problem(data, alternatives, LpMaximize)
            epsilon = 0.01
            for b1, b2 in reference_pairs:
                prob += alternative_utilities[b1] >= alternative_utilities[b2] + epsilon, f"Reference_{b1}_{b2}"
            sigma = LpVariable("sigma", lowBound=-100.0)
            prob += sigma, "Maximize_Sigma"
            prob += sigma == alternative_utilities[a1] - alternative_utilities[a2], f"Diff_{a1}_{a2}"
            prob.solve()
            if value(sigma) >= 0:
                possible_prefs.append((a1, a2))
                
    return necessary_prefs, possible_prefs
    
    # prob, criterion_vars, alternative_utilities = create_problem(data, alternatives)
    # epsilon = LpVariable("epsilon", lowBound=1e-6)
    # sigma = LpVariable("sigma", lowBound=0.0)
    # prob += epsilon - sigma, "Maximize_Strong_Weak_Preference"

    # for a1, a2 in reference_pairs:
    #     prob += alternative_utilities[a1] >= alternative_utilities[a2] + epsilon, f"Necessary_Preference_{a1}_{a2}"
    #     prob += alternative_utilities[a1] >= alternative_utilities[a2] - sigma, f"Possible_Preference_{a1}_{a2}"

    # return prob, criterion_vars, alternative_utilities

def check_preference(prob_template, utility1, utility2, minimize=True):
    prob = deepcopy(prob_template)
    diff = LpVariable(f"diff_{a1}_{a2}", None, None)
    
    prob += diff == utility1 - utility2
    
    if minimize:
        prob += diff, f"Minimize_U({a1})_minus_U({a2})"
    else:
        prob += -diff, f"Maximize_U({a1})_minus_U({a2})"

    prob.solve()
    return value(diff)

def analyze_resistance(data, alternatives, reference_pairs):
    prob, criterion_vars, alternative_utilities = uta_gms(data, alternatives, reference_pairs)
    
    necessary_prefs = []
    possible_prefs = []

    alt_keys = list(alternatives.keys())

    for i in range(len(alt_keys)):
        for j in range(len(alt_keys)):
            if i == j:
                continue
            a1, a2 = alt_keys[i], alt_keys[j]
            necessary_diff = check_preference(prob, alternative_utilities[a1], alternative_utilities[a2], minimize=True)
            if necessary_diff >= 0:
                necessary_prefs.append((a1, a2))

            possible_diff = check_preference(prob, alternative_utilities[a1], alternative_utilities[a2], minimize=False)
            if possible_diff >= 0:
                possible_prefs.append((a1, a2))

    return necessary_prefs, possible_prefs

def create_problem(data: dict[str, list[float]], alternatives: dict[str, dict[str, float]], minmax: int = LpMaximize) -> tuple[LpProblem, dict[str, dict[str, LpVariable]], dict[str, LpVariable]]:
    prob = LpProblem("UTA_Method", minmax)

    criterion_vars = {c: {v: LpVariable(f"{c}_{v}", 0, 1) for v in sorted(set(data[c]))} for c in data}
    extremes = get_extremes(data)
    for c in data:
        values = sorted(set(data[c]))
        for i in range(len(values) - 1):
            prob += criterion_vars[c][values[i]] >= criterion_vars[c][values[i + 1]]
        min_val, max_val = extremes[c]
        prob += criterion_vars[c][min_val] >= 0.1
        prob += criterion_vars[c][min_val] <= 0.5
        prob += criterion_vars[c][max_val] == 0.0
    prob += lpSum(criterion_vars[c][min_val] for c, (min_val, _) in extremes.items()) == 1    

    alternative_utilities = {}
    for name, evals in alternatives.items():
        utility_var = LpVariable(f"{name}_Utility", lowBound=0)
        alternative_utilities[name] = utility_var
        prob += utility_var == lpSum(criterion_vars[c][float(evals[c])] for c in data)

    return prob, criterion_vars, alternative_utilities

def select_reference_pairs(alternatives: dict[str, dict[str, str]], k=5) -> list[tuple[str, str]]:
    alt_keys = list(alternatives.keys())
    return random.sample([(a1, a2) for a1 in alt_keys for a2 in alt_keys if a1 != a2], k)

def plot_results(criterion_vars):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for idx, (c, vars) in enumerate(criterion_vars.items()):
        row, col = divmod(idx, 2)
        x = sorted(vars.keys())
        y = [value(vars[v]) for v in x]
        axes[row, col].plot(x, y, marker='o', linestyle='-', label=f'$u(g_{{{c}}})$')
        max_y = max(y)
        yticks = axes[row, col].get_yticks()
        if max_y not in yticks:
            yticks = list(yticks) + [max_y]
            axes[row, col].set_yticks(yticks)
        axes[row, col].set_title(f'Criterion {c}')
        axes[row, col].legend()
        axes[row, col].grid(True)
        axes[row, col].set_xlabel(f'$g_{{{c}}}$')
        axes[row, col].set_ylabel(f'$u(g_{{{c}}})$')
    plt.tight_layout()
    plt.savefig('uta.png')
    # plt.show()

def plot_hasse_diagram(necessary_prefs):
    G = nx.DiGraph()
    G.add_edges_from(necessary_prefs)

    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=30, node_color="lightblue", font_size=10, edge_color="black", arrows=True)
    plt.title("Diagram Hasse'go dla preferencji koniecznych")
    plt.savefig('hasse.png')
    plt.show()

if __name__ == '__main__':
    criterions, values, alternatives = get_data('data.csv')
    alternatives = prepare_alternatives(alternatives, criterions)
    # reference_pairs = select_reference_pairs(alternatives)
    reference_pairs = [
        (f"Alternative_14", f"Alternative_11"), # 151863 (R2 > R1)
        (f"Alternative_2", f"Alternative_25"), # 151879 (R1 > R3)
        (f"Alternative_11", f"Alternative_17"), # R2 >> R3
        (f"Alternative_4", f"Alternative_5"), # F1 > F2
        (f"Alternative_4", f"Alternative_6"), # F1 > F3
    ] # + select_reference_pairs(alternatives, 3)
    
    prob, criterion_vars, alternative_utilities = uta(values, alternatives, reference_pairs)
    prob.solve()

    print(f"Status: {LpStatus[prob.status]}")
    # for var in prob.variables():
    #     print(f"{var.name} = {value(var)}")
    print("\nObjective value:", value(prob.objective))

    plot_results(criterion_vars)
    
    for name, utility in sorted(alternative_utilities.items(), key=lambda item: value(item[1]), reverse=True):
        weighted_utility = {c: value(criterion_vars[c][float(alternatives[name][c])]) for c in criterions}
        print(f"Utility of {name}: {weighted_utility}, summary: {value(utility)}")
    
    print("\nReference pairs:")
    for a1, a2 in reference_pairs:
        print(f"Utility of {a1} >= Utility of {a2}: {value(alternative_utilities[a1])} >= {value(alternative_utilities[a2])}")
    
    necessary_prefs, possible_prefs = uta_gms(values, alternatives, reference_pairs)
    print("\nNecessary Preferences:")
    for a1, a2 in necessary_prefs:
        print(f"Necessary pref: {a1} >= {a2}")
    print("\nPossible Preferences:")
    for a1, a2 in possible_prefs:
        print(f"Possible pref: {a1} >= {a2}")

    plot_hasse_diagram(necessary_prefs)
