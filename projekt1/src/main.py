import numpy as np
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, value, LpStatus
import random
import matplotlib.pyplot as plt

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

def uta(data: dict[str, list[float]], alternatives: dict[str, dict[str, float]], reference_pairs) -> tuple[LpProblem, dict[str, dict[str, LpVariable]]]:
    prob, criterion_vars, alternative_utilities = create_problem(data, alternatives)
    epsilon = LpVariable("epsilon", lowBound=1e-6)
    prob += epsilon, "Maximize_Epsilon"
    
    for a1, a2 in reference_pairs:
        prob += alternative_utilities[a1] >= alternative_utilities[a2] + epsilon, f"Reference_{a1}_{a2}"

    return prob, criterion_vars

def uta_gms(data: dict[str, list[float]], alternatives: dict[str, dict[str, float]], reference_pairs) -> tuple[LpProblem, dict[str, dict[str, LpVariable]]]:
    prob, criterion_vars, alternative_utilities = create_problem(data, alternatives)
    epsilon = LpVariable("epsilon", lowBound=1e-6)
    sigma = LpVariable("sigma", lowBound=0.0)
    prob += epsilon - sigma, "Maximize_Strong_Weak_Preference"

    for a1, a2 in reference_pairs:
        prob += alternative_utilities[a1] >= alternative_utilities[a2] + epsilon, f"Necessary_Preference_{a1}_{a2}"
        prob += alternative_utilities[a1] >= alternative_utilities[a2] - sigma, f"Possible_Preference_{a1}_{a2}"

    return prob, criterion_vars, alternative_utilities

def check_preference(prob_template, alternative_utilities, a1, a2, minimize=True):
    prob = prob_template.deepcopy()
    diff = LpVariable(f"diff_{a1}_{a2}", None, None)
    
    prob += diff == alternative_utilities[a1] - alternative_utilities[a2]
    
    if minimize:
        prob += diff, f"Minimize_U({a1})_minus_U({a2})"
    else:
        prob += -diff, f"Maximize_U({a1})_minus_U({a2})"

    prob.solve()
    return value(diff)

def analyze_resistance(prob_template, alternatives):
    necessary_prefs = []
    possible_prefs = []

    alt_keys = list(alternatives.keys())

    for i in range(len(alt_keys)):
        for j in range(i + 1, len(alt_keys)):
            a1, a2 = alt_keys[i], alt_keys[j]

            necessary_diff = check_preference(prob_template, alternatives, a1, a2, minimize=True)
            if necessary_diff >= 0:
                necessary_prefs.append((a1, a2))

            possible_diff = check_preference(prob_template, alternatives, a1, a2, minimize=False)
            if possible_diff >= 0:
                possible_prefs.append((a1, a2))

    return necessary_prefs, possible_prefs

def create_problem(data: dict[str, list[float]], alternatives: dict[str, dict[str, float]]) -> tuple[LpProblem, dict[str, dict[str, LpVariable]]]:
    prob = LpProblem("UTA_Method", LpMaximize)

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
        axes[row, col].set_title(f'Criterion {c}')
        axes[row, col].legend()
        axes[row, col].grid(True)
        axes[row, col].set_xlabel(f'$g_{{{c}}}$')
        axes[row, col].set_ylabel(f'$u(g_{{{c}}})$')
    plt.tight_layout()
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
    
    prob, criterion_vars = uta(values, alternatives, reference_pairs)
    prob.solve()

    print(f"Status: {LpStatus[prob.status]}")
    # for var in prob.variables():
    #     print(f"{var.name} = {value(var)}")
    print("\nObjective value:", value(prob.objective))

    plot_results(criterion_vars)
    
    utilities = {var.name[:-8]: value(var) for var in prob.variables() if var.name.endswith("_Utility")}
    utilities = dict(sorted(utilities.items(), key=lambda item: item[1], reverse=True))
    for name, utility in utilities.items():
        print(f"Utility of {name}: {utility}")
    
    print("\nReference pairs:")
    for a1, a2 in reference_pairs:
        print(f"Utility of {a1} >= Utility of {a2}: {utilities[a1]} >= {utilities[a2]}")

    print("\nGMS Method")
    prob, criterion_vars, alternative_utilities = uta_gms(values, alternatives, reference_pairs)
    prob.solve()

    necessary_prefs, possible_prefs = analyze_resistance(prob, alternative_utilities)
    print("\nNecessary Preferences:")
    for a1, a2 in necessary_prefs:
        print(f"Necessary pref: {a1} >= {a2}")
    print("\nPossible Preferences:")
    for a1, a2 in possible_prefs:
        print(f"Possible pref: {a1} >= {a2}")      

    print(f"Status: {LpStatus[prob.status]}")
    # for var in prob.variables():
    #     print(f"{var.name} = {value(var)}")
    print("\nObjective value:", value(prob.objective))
