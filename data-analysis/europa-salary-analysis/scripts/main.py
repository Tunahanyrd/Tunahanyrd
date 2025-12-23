# scripts/main.py
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from clean_data import load_and_clean_data, fill_missing_values, save_cleaned_data
import subprocess

DATA_PATH = "../data/raw/world_bank_indicators.csv"

print(" Data is cleaning...")
df = load_and_clean_data(DATA_PATH)
df = fill_missing_values(df)
save_cleaned_data(df)

print(" Static plots are being drawn...")
subprocess.run(["python", "analysis_runner.py"])

print(" Exploratory plots are being drawn...")
subprocess.run(["python", "exploratory_runner.py"])

print(" Animated plots are being generated...")
subprocess.run(["python", "animation_runner.py"])

print(" All right.")



