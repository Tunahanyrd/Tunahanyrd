# scripts/exploratory_runner.py
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os
import pandas as pd
import inspect
import plot_exploratory as pe
import matplotlib.pyplot as plt
from tqdm import tqdm

df = pd.read_csv("../data/processed/world_bank_cleaned.csv")
countries = df["country"].unique()

special_funcs = {
    "ridge_plot": {"indicator": "GDP per capita (current US$)"},
    "boxplot_by_indicator": {"indicator": "GDP per capita (current US$)"},
    "spearman_vs_pearson": {"indicators": [
        "GDP per capita (current US$)",
        "Life expectancy at birth, total (years)",
        "Population, total"
    ]},
}

exploratory_funcs = {
    name: func for name, func in inspect.getmembers(pe, inspect.isfunction)
    if name in pe.__all__
}

out_dir = "../results/exploratory"
os.makedirs(out_dir, exist_ok=True)

for country in tqdm(countries, desc="Countries"):
    country_dir = os.path.join(out_dir, country)
    os.makedirs(country_dir, exist_ok=True)
    
    for name, func in tqdm(exploratory_funcs.items(), leave=False, desc="Functions"):
        try:
            kwargs = special_funcs.get(name, {})
            
            if name in ["boxplot_by_indicator"]:
                fig = func(df, **kwargs)
            else:
                fig = func(df, country, **kwargs)

            fig.savefig(os.path.join(country_dir, f"{name}.png"), bbox_inches="tight")
            plt.close('all') 
        except Exception as e:
            print(f"{country} - {name} → {e}")

