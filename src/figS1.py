import pandas as pd
import matplotlib.pyplot as plt
import os

# Global plot settings
plt.rcParams["font.family"] = "Garamond"
plt.rcParams.update({'font.size': 16})

# Color palette for disorder categories
COLOR_PALETTE = {
    'Depressive Disorders': "#5a66a3",
    'Anxiety Disorders': "#fdae61",
    'Attention-Deficit/Hyperactivity Disorders': "#e3bb46",
    'Bipolar and Related Disorders': "#d73027",
    'Obsessive-Compulsive Disorders': "#bd73f5",
    'Cluster B Personality Disorders': "#91bfdb",
    'Autism Spectrum Disorders': "#a6d96a",
    'Post-Traumatic Stress Disorders': "#fa50a0",
    'Substance-Related Disorders': "#1b9e77",
    'Social Anxiety Disorders': "#09e2ed",
}

def load_and_filter_pmdd_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] > '2012-04') & (df['date'] <= '2023-12')]
    df = df[df['subreddit'] == 'PMDD']
    return df

def load_and_filter_all_data(filepath: str, selected_conditions: list[str]) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df = df[df['subcategory'].isin(selected_conditions)]
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] > '2012-04') & (df['date'] <= '2023-12')]
    return df

def get_top_disorders(df: pd.DataFrame, top_n: int = 11) -> list[str]:
    subcategory_counts = df.groupby(['subcategory', 'user']).first().reset_index()
    top_subcategories = subcategory_counts['subcategory'].value_counts().head(top_n).index
    return top_subcategories[1:]  # Exclude the first one (PMDD)

def plot_volume_timeline(df_pmdd: pd.DataFrame, df_all: pd.DataFrame, selected_conditions: list[str], output_path: str):
    fig, ax = plt.subplots(figsize=(10, 5))

    # PMDD posts
    df_pmdd['date'].dt.to_period("M").value_counts().sort_index().plot(
        ax=ax, color='#800020', linewidth=2.5, label='r/PMDD', alpha=1, linestyle='-'
    )

    # Disorder categories
    for condition in selected_conditions:
        df_condition = df_all[df_all['subcategory'] == condition]
        df_condition['date'].dt.to_period("M").value_counts().sort_index().plot(
            ax=ax, linewidth=1.5, label=condition, alpha=0.7, linestyle='--',
            color=COLOR_PALETTE.get(condition, '#333333')
        )

    ax.set_ylabel('Number of posts (monthly)')
    ax.set_xlim('2011-12', '2023-10')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', linestyle='--', alpha=0.6, linewidth=1.5)
    plt.ylim(0, 24999)
    ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99), fontsize=14)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    pmdd_path = "data/posts_pmdd.csv"
    all_data_path = "data/posts_MHSN.csv"
    output_figure = "figures/fig_S1.png"

    df_pmdd = load_and_filter_pmdd_data(pmdd_path)
    top_disorders = get_top_disorders(df_pmdd)
    df_all = load_and_filter_all_data(all_data_path, top_disorders)

    plot_volume_timeline(df_pmdd, df_all, top_disorders, output_figure)
