# scripts/analysis_runner.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import plot_trends as pt
import inspect
from tqdm import tqdm

def save_plot(fig, country_code, plot_name):
    out_dir = f"../results/plots/{country_code}"
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(f"{out_dir}/{plot_name}.png", bbox_inches="tight")
    plt.close(fig)

df = pd.read_csv("../data/processed/world_bank_cleaned.csv")
countries = df["country"].unique()

plot_jobs = [
    {"func": func, "name": name}
    for name, func in inspect.getmembers(pt, inspect.isfunction)
]

for country in tqdm(countries, desc="Countries: "):
    for job in tqdm(plot_jobs, desc="Jobs by Plot: "):
        try:
            fig = job["func"](df,country)
            save_plot(fig, country, job["name"])
        except Exception as e:
            print(f"{country} - {job['name']} → {e}")







