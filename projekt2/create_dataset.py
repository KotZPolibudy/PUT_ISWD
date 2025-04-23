import pandas as pd
import numpy as np
import os

def main(num_samples=10):
    SEED = 6969
    np.random.seed(SEED)

    INPUT_FILE = "Smart_Farming_Crop_Yield_2024.csv"
  
    paths = [
        "./electre_iii_pl/data/own",
        "./promethee_pl/data/own"
    ]
    for path in paths:
        os.makedirs(path, exist_ok=True)

    columns_of_interest = [
        "farm_id",                 # idx
        "sunlight_hours",          # gain
        "pesticide_usage_ml",      # cost
        "rainfall_mm",             # gain
        "yield_kg_per_hectare"     # gain
    ]
    
    df = pd.read_csv(INPUT_FILE, usecols=columns_of_interest)
    df.set_index("farm_id", inplace=True)

    print("Min-max ranges for selected criteria:")
    for col in df.columns:
        print(f"{col}: min = {df[col].min()}, max = {df[col].max()}")

    if len(df) < num_samples:
        raise ValueError(f"Dataset has fewer than {num_samples} entries.")
    sampled_df = df.sample(n=num_samples, random_state=SEED)

    file_name = "dataset.csv"
    sampled_df.to_csv(file_name)
    print(f"\nSaved {num_samples} sampled alternatives to {file_name}")

    for path in paths:
        full_path = os.path.join(path, file_name)
        sampled_df.to_csv(full_path)
        print(f"Saved {num_samples} sampled alternatives to {full_path}")
        
    preferences = []
    for a1 in sampled_df.index:
        for a2 in sampled_df.index:
            flag = True
            for criterion in sampled_df.columns:
                a1_value = sampled_df.loc[a1, criterion]
                a2_value = sampled_df.loc[a2, criterion]
                if criterion == "pesticide_usage_ml":
                    if a2_value < a1_value:
                        flag = False
                else:
                    if a1_value < a2_value:
                        flag = False
            if flag and a1 != a2:
                preferences.append((a1, a2))
    print(f"\nPreferences for sampled alternatives:{preferences}")

if __name__ == "__main__":
    main()

