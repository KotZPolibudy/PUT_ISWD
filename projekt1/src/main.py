import numpy as np
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, value, LpStatus
import random
import matplotlib.pyplot as plt

R1 = [1,2,3,10,11,12,19,20,21]
R2 = [4,5,6,13,14,15,22,23,24]
R3 = [7,8,9,16,17,18,25,26,27]

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

def create_problem(data: dict[str, list[float]], alternatives: dict[str, dict[str, float]], reference_pairs) -> tuple[LpProblem, dict[str, dict[str, LpVariable]]]:
    prob = LpProblem("UTA_Method", LpMaximize)

    criterion_vars = {c: {v: LpVariable(f"{c}_{v}", 0, 1) for v in sorted(set(data[c]))} for c in data}
    extremes = get_extremes(data)
    for c in data:
        values = sorted(set(data[c]))
        for i in range(len(values) - 1):
            prob += criterion_vars[c][values[i]] <= criterion_vars[c][values[i + 1]]
        min_val, max_val = extremes[c]
        prob += criterion_vars[c][min_val] == 0.1
        prob += criterion_vars[c][max_val] <= 0.75
    prob += lpSum(criterion_vars[c][max_val] for c, (_, max_val) in extremes.items()) == 1    

    alternative_utilities = {}
    for name, evals in alternatives.items():
        utility_var = LpVariable(f"{name}_Utility", lowBound=0)
        alternative_utilities[name] = utility_var
        prob += utility_var == lpSum(criterion_vars[c][float(evals[c])] for c in data)

    epsilon_loc = LpVariable("epsilon_loc", lowBound=0.0)
    epsilon_time = LpVariable("epsilon_time", lowBound=0.0)
    epsilon_pair = LpVariable("epsilon_pair", lowBound=0.0)
    
    prob += epsilon_loc
    prob += epsilon_time
    prob += epsilon_pair
    
    for a1, a2 in reference_pairs:
        prob += alternative_utilities[a1] >= alternative_utilities[a2] + epsilon_pair

    # Czwarta grupa: Preferowana jest lokalizacja R2 nad R1 oraz R1 nad R3.
    for alt2, alt1 in zip(R2, R1):
        prob += alternative_utilities[f"Alternative_{alt2}"] >= alternative_utilities[f"Alternative_{alt1}"] + epsilon_loc
    for alt1, alt3 in zip(R1, R3):
        prob += alternative_utilities[f"Alternative_{alt1}"] >= alternative_utilities[f"Alternative_{alt3}"] + epsilon_loc
    
    # Trzecia grupa: Inwestorzy chcą jak najdłuzej utrzymywać składowisko.    
    # preferowane opcje ze scenariusza czasowego S3 nad S1 oraz S2
    for alt3, alt12 in zip(S3, S1 + S2):
        prob += alternative_utilities[f"Alternative_{alt3}"] >= alternative_utilities[f"Alternative_{alt12}"] + epsilon_time
        
    return prob, criterion_vars

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
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    criterions, values, alternatives = get_data('data.csv')
    alternatives = prepare_alternatives(alternatives, criterions)
    reference_pairs = select_reference_pairs(alternatives)
    
    prob, criterion_vars = create_problem(values, alternatives, reference_pairs)
    prob.solve()

    print(f"Status: {LpStatus[prob.status]}")
    # for var in prob.variables():
    #     print(f"{var.name} = {value(var)}")
    # print("\nObjective value:", value(prob.objective))

    plot_results(criterion_vars)
    
    utilities = {var.name[:-8]: value(var) for var in prob.variables() if var.name.endswith("_Utility")}
    for a1, a2 in reference_pairs:
        print(f"Utility of {a1} >= Utility of {a2}: {utilities[a1]} >= {utilities[a2]}")
