import numpy as np
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, value, LpStatus
import random

def read_data(file_path: str) -> list[str]:
    with open(file_path, 'r') as file:
        data = file.readlines()
    return data

def process_data(data: list[str], k: int = 10) -> tuple[list[str], dict[str, list[float]], list[str]]:
    data = [line.replace('\n', '') for line in data]
    data = [line.split(',') for line in data]
    criterions = data[0][1:]
    alternatives = data[1:]
    # alternatives = random.sample(alternatives, k)
    data = data[1:]
    data = [line[1:] for line in data]
    data = np.array(data).T
    data = {criterion: list(map(float, values)) for criterion, values in zip(criterions, data)}
    return criterions, data, alternatives

def get_data(file_path: str) -> tuple[list[str], dict[str, list[float]], list[str]]:
    return process_data(read_data(file_path))

def get_extremes(data: dict[str, list[float]]) -> dict[str, tuple[float, float]]:
    return {criterion: (min(values), max(values)) for criterion, values in data.items()}

def prepare_alternatives(alternatives: list[str], criterions: list[str]):
    output = {}
    for i in range(len(alternatives)):
        output[f"Alternative_{alternatives[i][0]}"] = {criterion: value for criterion, value in zip(criterions, alternatives[i][1:])}
    return output
    
def create_problem(data: dict[str, list[float]], alternatives: list[dict[str, float]]) -> LpProblem:
    prob = LpProblem("Simplified_UTA_Method", LpMaximize)

    criterion1_values = sorted(list(set(data['C1'])))
    criterion2_values = sorted(list(set(data['C2'])))
    criterion3_values = sorted(list(set(data['C3'])))
    criterion4_values = sorted(list(set(data['C4'])))

    criterion1_vars = {val: LpVariable(f'c1_{val}', 0, 1, cat='Continuous') for val in criterion1_values}
    criterion2_vars = {val: LpVariable(f'c2_{val}', 0, 1, cat='Continuous') for val in criterion2_values}
    criterion3_vars = {val: LpVariable(f'c3_{val}', 0, 1, cat='Continuous') for val in criterion3_values}
    criterion4_vars = {val: LpVariable(f'c4_{val}', 0, 1, cat='Continuous') for val in criterion4_values}

    for i in range(len(criterion1_values) - 1):
        prob += criterion1_vars[criterion1_values[i]] <= criterion1_vars[criterion1_values[i + 1]], f'Monotonicity_C1_{i}'
    for i in range(len(criterion2_values) - 1):
        prob += criterion2_vars[criterion2_values[i]] <= criterion2_vars[criterion2_values[i + 1]], f'Monotonicity_C2_{i}'
    for i in range(len(criterion3_values) - 1):
        prob += criterion3_vars[criterion3_values[i]] <= criterion3_vars[criterion3_values[i + 1]], f'Monotonicity_C3_{i}'
    for i in range(len(criterion4_values) - 1):
        prob += criterion4_vars[criterion4_values[i]] <= criterion4_vars[criterion4_values[i + 1]], f'Monotonicity_C4_{i}'
    
    prob += criterion1_vars[criterion1_values[0]] == 0, "Normalization_C1_Lowest"
    prob += criterion2_vars[criterion2_values[0]] == 0, "Normalization_C2_Lowest"
    prob += criterion3_vars[criterion3_values[0]] == 0, "Normalization_C3_Lowest"
    prob += criterion4_vars[criterion4_values[0]] == 0, "Normalization_C4_Lowest"
    prob += criterion1_vars[criterion1_values[-1]] + criterion2_vars[criterion2_values[-1]] + criterion3_vars[criterion3_values[-1]] + criterion4_vars[criterion4_values[-1]] == 1, "Normalization_Criteria_Sum"

    prob += criterion1_vars[criterion1_values[-1]] <= 0.75, "Weight_C1_Limit_Up"
    prob += criterion2_vars[criterion2_values[-1]] <= 0.75, "Weight_C2_Limit_Up"
    prob += criterion3_vars[criterion3_values[-1]] <= 0.75, "Weight_C3_Limit_Up"
    prob += criterion4_vars[criterion4_values[-1]] <= 0.75, "Weight_C4_Limit_Up"

    prob += criterion1_vars[criterion1_values[0]] >= 0.1, "Weight_C1_Limit_Down"
    prob += criterion2_vars[criterion2_values[0]] >= 0.1, "Weight_C2_Limit_Down"
    prob += criterion3_vars[criterion3_values[0]] >= 0.1, "Weight_C3_Limit_Down"
    prob += criterion4_vars[criterion4_values[0]] >= 0.1, "Weight_C4_Limit_Down"

    alternative_utilities = {}
    for name, evals in alternatives.items():
        utility_var = LpVariable(f"{name}_Utility", lowBound=0)
        alternative_utilities[name] = utility_var
        prob += utility_var == lpSum(
            [
                criterion1_vars[float(evals['C1'])],
                criterion2_vars[float(evals['C2'])],
                criterion3_vars[float(evals['C3'])],
                criterion4_vars[float(evals['C4'])]
            ]
        ), f"Utility_{name}"
    epsilon = 0.01
    # prob += alternative_utilities['Alternative_1'] >= alternative_utilities['Alternative_2'] + epsilon, "Pref_A1_A2"


    return prob

if __name__ == '__main__':
    criterions, values, alternatives = get_data('data.csv')
    alternatives = prepare_alternatives(alternatives, criterions)
    prob = create_problem(values, alternatives)