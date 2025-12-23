# scripts/animation_runner.py

import os
import pandas as pd
import inspect
import plot_animated as pa
from matplotlib import pyplot as plt
from tqdm import tqdm

df = pd.read_csv("../data/processed/world_bank_cleaned.csv")
countries = df["country"].unique()

animation_jobs = {
    name: func for name, func in inspect.getmembers(pa, inspect.isfunction)
    if name.startswith("anim_")
}

gif_dir = os.path.join("..", "results", "gifs")
os.makedirs(gif_dir, exist_ok=True)

plt.ioff()

for country in tqdm(countries, desc="Countries"):
    out_path = os.path.join(gif_dir, country)
    os.makedirs(out_path, exist_ok=True)

    for name, func in tqdm(animation_jobs.items(), leave=False, desc="Animations"):
        try:
            anim = func(df, country)

            gif_path = os.path.join(out_path, f"{name}.gif")
            anim.save(gif_path, writer="pillow", fps=5)

            plt.close("all")  

        except Exception as e:
            print(f"{country} - {name} → {e}")
