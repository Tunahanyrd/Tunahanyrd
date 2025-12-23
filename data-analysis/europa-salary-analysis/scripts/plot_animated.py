# scripts/plot_animated.py
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def anim_age_distribution(df, country_code):
    country_df = df[df["country"] == country_code]
    years = sorted(country_df["year"].unique())
    groups = [
        'Population ages 0-14 (% of total population)',
        'Population ages 15-64 (% of total population)',
        'Population ages 65 and above (% of total population)'
    ]
    fig, ax = plt.subplots(figsize=(8,8))
    bars = ax.bar(groups, [0,0,0])
    ax.set_ylim(0, country_df[groups].values.max() * 1.1) 
    ax.set_title(f'{country_code.upper()} - Age Distribution')
    ax.tick_params(axis='x', labelrotation=5, labelsize=10)

    def update(i):
        vals = country_df[country_df["year"] == years[i]][groups].values.flatten()
        for bar, h in zip(bars, vals):
            bar.set_height(h)
        ax.set_ylabel(f'Year: {years[i]}')
        return bars
    anim = FuncAnimation(fig, update, frames=len(years), blit=False, repeat=False)
    return anim

def anim_scatter_gdp_birth(df, country_code):
    country_df = df[df['country'] == country_code]
    years = sorted(country_df['year'].unique())

    gdp_vals = []
    birth_vals = []

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(country_df['GDP per capita (current US$)'].min(), country_df['GDP per capita (current US$)'].max())
    ax.set_ylim(country_df['Birth rate, crude (per 1,000 people)'].min(), country_df['Birth rate, crude (per 1,000 people)'].max())
    ax.set_xlabel("GDP per Capita")
    ax.set_ylabel("Birth Rate")
    ax.set_title(f"{country_code.upper()} - GDP vs Birth (Animated Line)")
    ax.tick_params(axis='x', labelsize=10)

    line, = ax.plot([], [], color="blue", linewidth=2)

    def update(frame):
        year = years[frame]
        sub = country_df[country_df['year'] == year]
        gdp_vals.append(sub['GDP per capita (current US$)'].values[0])
        birth_vals.append(sub['Birth rate, crude (per 1,000 people)'].values[0])
        line.set_data(gdp_vals, birth_vals)
        ax.set_title(f"{country_code.upper()} - GDP vs Birth ({year})")
        return line,

    anim = FuncAnimation(fig, update, frames=len(years), blit=False, repeat=False)
    return anim

