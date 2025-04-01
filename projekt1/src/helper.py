import numpy as np
from pulp import *
import random
import matplotlib.pyplot as plt
import pandas as pd

# Wczytanie danych
file_path = "../data.csv"
data = pd.read_csv(file_path, index_col=0)
criteria = list(data.columns)

# Tworzenie problemu optymalizacyjnego
prob = LpProblem("UTA_Method", LpMaximize)

# Tworzenie zmiennych decyzyjnych dla każdego kryterium
criteria_vars = {
    c: {val: LpVariable(f"{c}_{val}", lowBound=0, upBound=1) for val in sorted(data[c].unique(), reverse=True)} for c in
    criteria} # reverse=True - bo są typu koszt

# Ograniczenia monotoniczności (odwrotne - bo koszty)
for c, vars_dict in criteria_vars.items():
    sorted_values = sorted(vars_dict.keys(), reverse=True)
    for i in range(len(sorted_values) - 1):
        prob += vars_dict[sorted_values[i]] >= vars_dict[sorted_values[i + 1]], f"Monotonicity_{c}_{i}"

# Ograniczenia normalizacji (0 -> highest, koszta)
for c in criteria:
    prob += criteria_vars[c][max(data[c])] == 0, f"Normalize_{c}_Highest"
prob += lpSum([criteria_vars[c][min(data[c])] for c in criteria]) == 1, "Normalize_Criteria_Sum"

# Ograniczenia wag kryteriów (min 0.1, max 0.5)
for c in criteria:
    prob += criteria_vars[c][min(data[c])] >= 0.1, f"Min_Weight_{c}"
    prob += criteria_vars[c][min(data[c])] <= 0.5, f"Max_Weight_{c}"

# Tworzenie zmiennych użytkowych dla alternatyw
table_utilities = {}
for idx, row in data.iterrows():
    utility_var = LpVariable(f"Alt{idx}_Utility", lowBound=0)
    table_utilities[idx] = utility_var
    prob += utility_var == lpSum([criteria_vars[c][row[c]] for c in criteria]), f"Utility_Alt{idx}"

# Definiowanie preferencji i optymalizacji epsilon
epsilon = LpVariable("epsilon", lowBound=0)
prob += table_utilities[14] >= table_utilities[11] + epsilon, "Pref_14_11" # ta jest od Oskara (R2 > R1)
prob += table_utilities[2] >= table_utilities[25] + epsilon, "Pref_2_25" # ta jest od Kota  (R1 > R3)
# todo -> dodać jeszcze 3 jakieś pary żeby mieć do referencyjnych
# Ponieważ obydwoje mamy Czwartą grupę informacji preferencyjnej,
# wybieramy przykłady które ją spełniają przede wszystkim. R2 > R1 > R3
prob += table_utilities[11] >= table_utilities[17] + epsilon, "Pref_11_17"  # R2 >> R3
# ten przykład daje nam kontynuację 14 > 11 > 17 (mają te samą grupę finansowania, ale różne lokacje)

# prob += table_utilities[4] >= table_utilities[1] + epsilon, "Pref_4_1"   # R2 > R1
# prob += table_utilities[1] >= table_utilities[7] + epsilon, "Pref_1_7"  # R1 > R3
# wykomentowane, bo ma być 5 par wariantów referencyjnych,
# oraz 2 grupy preferencji - więc trzeba dodać coś z drugiej grupy preferencji

# przykłady "remisu" lokalizacji, różniące sposobem finansowania (zakładamy, że F1 > F2 oraz F1 > F3)
prob += table_utilities[4] >= table_utilities[5] + epsilon, "Pref_4_5"   # F1 > F2
prob += table_utilities[4] >= table_utilities[6] + epsilon, "Pref_4_6"   # F1 > F3


# Maksymalizacja epsilon
prob += epsilon, "Maximize_Epsilon"

# Rozwiązanie problemu
prob.solve()

# Wyświetlenie wyników
print(f"\n{'='*30}\nSTATUS: {LpStatus[prob.status]}\n{'='*30}\n")
print("Optymalne rozwiązanie:")
max_var_length = max(len(var.name) for var in prob.variables())
for var in prob.variables():
    print(f"{var.name.ljust(max_var_length)} = {value(var):.4f}")
print("\nWartość funkcji celu:", f"{value(prob.objective):.4f}\n")

# Generowanie wykresów dla każdego kryterium
plt.figure(figsize=(12, 8))
for i, c in enumerate(criteria, 1):
    values = sorted(criteria_vars[c].keys(), reverse=True) # sort reverse? czy nie?
    solver_values = [value(criteria_vars[c][v]) for v in values]

    plt.subplot(2, 2, i)
    plt.plot(values, solver_values, marker='o', linestyle='-', label=f'$u(g_{{{c}}})$')
    plt.xlabel(f'$g_{{{c}}}$')
    plt.ylabel(f'$u(g_{{{c}}})$')
    plt.title(f'Funkcja użyteczności dla {c}')
    plt.xticks(values, rotation=45)
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()
