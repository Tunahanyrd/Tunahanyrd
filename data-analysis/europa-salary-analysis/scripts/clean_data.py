# scripts/clean_data.py

import pandas as pd

def load_and_clean_data(path):
    df = pd.read_csv(path)

    eu_countries = [
        'aut', 'bel', 'bgr', 'hrv', 'cyp', 'cze', 'dnk', 'est', 'fin', 'fra', 'deu',
        'grc', 'hun', 'irl', 'ita', 'lva', 'ltu', 'lux', 'mlt', 'nld', 'pol', 'prt',
        'rou', 'svk', 'svn', 'esp', 'swe', 'tur'
    ]
    
    df = df[df["country"].isin(eu_countries)].copy()
    df = df[(df['year'] >= 1961) & (df['year'] <= 2022)].copy()

    nulls = df.isna().mean()
    non_null_cols = nulls[nulls < 0.2].index.tolist()
    df = df[non_null_cols]

    return df

def fill_missing_values(df):
    filled = df.copy()
    num_cols = filled.select_dtypes(include = "number").columns # only num dtype columns
    
    for country in filled["country"].unique():
        country_df = filled[filled["country"] == country].sort_values("year")
        
        filled.loc[country_df.index, num_cols] = (
            country_df[num_cols].interpolate(method='linear', limit_direction='both')
            )
    return filled
    
def save_cleaned_data(df, path="../data/processed/world_bank_cleaned.csv"):
    df.to_csv(path, index=False)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    