# scripts/plot_exploratory.py 
__all__ = [
    "life_expectancy_kde",
    "zscore_heatmap",
    "pca_scatter",
    "boxplot_by_indicator",
]
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import zscore, gaussian_kde

def life_expectancy_kde(df, country_code):
    country_df = df[df['country'] == country_code]
    data = country_df['Life expectancy at birth, total (years)']
    density = gaussian_kde(data)
    xs = np.linspace(data.min(), data.max(), 200)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, density(xs))
    ax.set_title(f'{country_code.upper()} - Life Expectancy KDE')
    ax.set_xlabel('Life Expectancy')
    ax.set_ylabel('Density')
    ax.grid(True)
    return fig

def zscore_heatmap(df, country_code):
    country_df = df[df['country'] == country_code]
    num = country_df.select_dtypes(include='number').drop(columns=['year'])
    
    zsq = num.apply(zscore)
    fig, ax = plt.subplots(figsize=(12, 8))
    zsq.index = country_df["year"].values
    
    sns.heatmap(zsq.T, cmap='vlag', ax=ax, cbar_kws={'label': 'Z-score'})
    ax.set_title(f'{country_code.upper()} - Z-score Heatmap')
    ax.set_xlabel('Year Index')
    ax.set_ylabel('Indicator')
    return fig

def pca_scatter(df, country_code, n_components=2):
    country_df = df[df['country'] == country_code]
    num = country_df.select_dtypes(include='number').drop(columns=['year'])
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(num.ffill().bfill())
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(transformed[:, 0], transformed[:, 1], c=country_df['year'], cmap='Spectral', edgecolor='black')
    ax.set_title(f'{country_code.upper()} - PCA Scatter')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    fig.colorbar(scatter, ax=ax, label='Year')
    ax.grid(True)
    return fig

def boxplot_by_indicator(df, indicator):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='country', y=indicator, data=df)
    ax.set_title(f'Boxplot of {indicator} by Country')
    ax.set_xlabel('Country')
    ax.set_ylabel(indicator)
    plt.xticks(rotation=45)
    fig.tight_layout()
    return fig

