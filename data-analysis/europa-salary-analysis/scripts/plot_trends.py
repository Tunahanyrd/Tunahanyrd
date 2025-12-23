# scripts/plot_trends.py
import matplotlib.pyplot as plt
import seaborn as sns

def gdp_vs_birth(df, country_code):
    country_df = df[df["country"] == country_code]
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(country_df["year"], country_df["GDP per capita (current US$)"], label="Birth Rate", color="blue")
    ax.plot(country_df["year"], country_df["Birth rate, crude (per 1,000 people)"], label="Birth Rate", color="green")
    
    ax.set_title(f"{country_code.upper()} - GDP vs Birth Rate")
    ax.set_xlabel("Year")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)
    
    return fig
    
def birth_per_population_ratio(df, country):
    country_df = df[df["country"] == country]
    birth_rate = country_df["Birth rate, crude (per 1,000 people)"]
    population = country_df["Population, total"]
    ratio = (birth_rate * 1000) / population # because the birth rate is x births per 1000 people
    
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(country_df["year"], ratio, label="Births per Person", color="orange")
    ax.set_title(f"{country.upper()} - Births per Capita")
    ax.set_ylabel("Births per Person")
    ax.set_xlabel("Year")
    ax.grid(True)
    ax.legend()
    
    return fig

def gdp_vs_lifetime(df, country):
    country_df = df[df["country"] == country]
    fig, ax = plt.subplots(figsize=(7, 7))
    x = country_df["GDP per capita (current US$)"]
    y = country_df["Life expectancy at birth, total (years)"]
    
    ax.scatter(x, y, c = country_df["year"], cmap="viridis", edgecolor = "black")
    ax.set_title(f"{country.upper()} - GDP vs Life Expectancy")
    ax.set_xlabel("GDP per Capita")
    ax.set_ylabel("Life Expectancy")
    ax.grid(True)

    return fig

def health_trends(df, country):
    country_df = df[df["country"] == country]
    health_cols = [
        "Life expectancy at birth, total (years)",
        "Life expectancy at birth, female (years)",
        "Life expectancy at birth, male (years)",
        "Mortality rate, infant (per 1,000 live births)",
        "Mortality rate, neonatal (per 1,000 live births)",
        "Mortality rate, under-5 (per 1,000 live births)"
    ]
    fig, ax = plt.subplots(figsize=(12, 6))
    for col in health_cols:
        if col in country_df.columns:
            ax.plot(country_df["year"], country_df[col], label=col)
        
        
    ax.set_title(f"{country.upper()} - Health Indicators Over Time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)

    return fig
    
def correlation_heatmap(df, country):
    country_df = df[df["country"] == country]
    num_df = country_df.select_dtypes(include="number").drop(columns=["year"])
    
    corr = num_df.corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", ax=ax, square=True, linewidths=0.5)
    ax.set_title(f"{country.upper()} - Correlation Heatmap")

    return fig

def age_distribution(df, country):
    country_df = df[df["country"] == country]
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.stackplot(
        country_df["year"],
        country_df["Population ages 0-14 (% of total population)"],
        country_df["Population ages 15-64 (% of total population)"],
        country_df["Population ages 65 and above (% of total population)"],
        labels=["0-14", "15-64", "65+"]
    )
    
    ax.set_title(f"{country.upper()} - Age Distribution Over Time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Percentage of Population")
    ax.legend(loc="upper right")
    ax.grid(True)

    return fig

def gdp_growth_vs_inflation(df, country):
    country_df = df[df["country"] == country]
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(country_df["year"], country_df["GDP growth (annual %)"], label="GDP Growth", color="blue")
    ax.plot(country_df["year"], country_df["Inflation, consumer prices (annual %)"], label="Inflation", color="red")
    
    ax.set_title(f"{country.upper()} - GDP Growth vs Inflation")
    ax.set_xlabel("Year")
    ax.set_ylabel("Percentage of Population")
    ax.legend(loc="upper right")
    ax.grid(True)
    
    return fig
    
def fertility_vs_mortality(df, country):
    country_df = df[df["country"] == country]
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(country_df["year"], country_df["Fertility rate, total (births per woman)"], label="Fertility Rate", color="green")
    ax.plot(country_df["year"], country_df["Death rate, crude (per 1,000 people)"], label="Death Rate", color="black")

    ax.set_title(f"{country.upper()} - Fertility vs Death Rate")
    ax.set_xlabel("Year")
    ax.set_ylabel("Rate")
    ax.legend()
    ax.grid(True)

    return fig

def arable_land_vs_yield(df, country):
    country_df = df[df["country"] == country]
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(country_df["year"], country_df["Arable land (% of land area)"], label="Arable Land (%)", color="sienna")
    ax.plot(country_df["year"], country_df["Cereal yield (kg per hectare)"], label="Cereal Yield", color="darkgreen")

    ax.set_title(f"{country.upper()} - Arable Land vs Cereal Yield")
    ax.set_xlabel("Year")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)

    return fig
    
def dependency_ratio(df, country):
    country_df = df[df["country"] == country]
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(country_df["year"], country_df["Age dependency ratio (% of working-age population)"], color="purple")
    ax.set_title(f"{country.upper()} - Age Dependency Ratio")
    ax.set_xlabel("Year")
    ax.set_ylabel("Dependency Ratio")
    ax.grid(True)

    return fig   

def migration_vs_growth(df, country_code):
    country_df = df[df["country"] == country_code]
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(country_df['year'], country_df['Net migration'], label='Net Migration', linestyle='--')
    ax.plot(country_df['year'], country_df['Population growth (annual %)'], label='Population Growth')
    ax.set_title(f'{country_code.upper()} - Migration vs Population Growth')
    ax.set_xlabel('Year')
    ax.set_ylabel('%')
    ax.legend()
    ax.grid(True)
    return fig

def exchange_rate_vs_gdp(df, country_code):
    country_df = df[df["country"] == country_code]
    fig, ax = plt.subplots(figsize=(7,7))
    sc = ax.scatter(
        country_df['Official exchange rate (LCU per US$, period average)'],
        country_df['GDP (current US$)'],
        c=country_df['year'], cmap='plasma', edgecolor='black'        
        )
    ax.set_title(f'{country_code.upper()} - Exchange Rate vs GDP')
    ax.set_xlabel('Exchange Rate (LCU per US$)')
    ax.set_ylabel('GDP (USD)')
    fig.colorbar(sc, ax=ax, label='Year')
    ax.grid(True)
    return fig


def volatility_plot(df, country_code, window=10):
    country_df = df[df['country'] == country_code]
    roll_std = country_df['GDP growth (annual %)'].rolling(window).std()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(country_df['year'], roll_std, color='red')
    ax.set_title(f'{country_code.upper()} - GDP Growth Volatility ({window}-Year Rolling STD)')
    ax.set_xlabel('Year')
    ax.set_ylabel('STD')
    ax.grid(True)
    return fig

def food_prod_index_trend(df, country_code):
    country_df = df[df['country'] == country_code]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(country_df['year'], country_df['Food production index (2014-2016 = 100)'], color='teal')
    ax.set_title(f'{country_code.upper()} - Food Production Index')
    ax.set_xlabel('Year')
    ax.set_ylabel('Index (2014-2016 = 100)')
    ax.grid(True)
    return fig

def fertilizer_vs_yield(df, country_code):
    country_df = df[df['country'] == country_code]
    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(
        country_df['Fertilizer consumption (kilograms per hectare of arable land)'],
        country_df['Cereal yield (kg per hectare)'],
        c=country_df['year'], cmap='magma', edgecolor='black'
    )
    ax.set_title(f'{country_code.upper()} - Fertilizer vs Cereal Yield')
    ax.set_xlabel('Fertilizer (kg/ha)')
    ax.set_ylabel('Cereal Yield (kg/ha)')
    fig.colorbar(sc, ax=ax, label='Year')
    ax.grid(True)
    return fig

def infant_vs_neonatal(df, country_code):
    country_df = df[df['country'] == country_code]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(country_df['year'], country_df['Mortality rate, infant (per 1,000 live births)'], label='Infant Mortality')
    ax.plot(country_df['year'], country_df['Mortality rate, neonatal (per 1,000 live births)'], label='Neonatal Mortality')
    ax.set_title(f'{country_code.upper()} - Infant vs Neonatal Mortality')
    ax.set_xlabel('Year')
    ax.set_ylabel('Rate (per 1,000)')
    ax.legend()
    ax.grid(True)
    return fig

def mobile_vs_life_expectancy(df, country_code):
    country_df = df[df['country'] == country_code]
    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(
        country_df['Mobile cellular subscriptions (per 100 people)'],
        country_df['Life expectancy at birth, total (years)'],
        c=country_df['year'], cmap='cividis', edgecolor='black'
    )
    ax.set_title(f'{country_code.upper()} - Mobile Subs vs Life Expectancy')
    ax.set_xlabel('Mobile Subs (per 100)')
    ax.set_ylabel('Life Expectancy')
    fig.colorbar(sc, ax=ax, label='Year')
    ax.grid(True)
    return fig   
    
