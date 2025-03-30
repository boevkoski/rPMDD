import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import timedelta
from matplotlib.gridspec import GridSpec
from statsmodels.tsa.stattools import adfuller, kpss
from scipy import stats
import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning

# Plotting settings
plt.rcParams["font.family"] = "Garamond"
plt.rcParams.update({'font.size': 18})
warnings.simplefilter('ignore', InterpolationWarning)

# Color palette
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

def mean_confidence_interval(data, confidence=0.95):
    data = np.array(data)
    n = len(data)
    m, se = np.mean(data), stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h

def kpss_test(data):
    statistic, p_value, _, _ = kpss(data, regression='c')
    print(f"KPSS Statistic: {statistic}, p-value: {p_value}")
    print(f"Result: The series is {'not ' if p_value < 0.05 else ''}stationary")

def load_and_prepare_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df['subcategory'] = df['subcategory'].apply(lambda x: x + 's' if not x.endswith('s') else x)
    df['date'] = pd.to_datetime(df['date'])
    return df

def compute_sliding_statistics(df: pd.DataFrame, start='2015-04-01', end='2023-08-15', window_days=90):
    selected_conditions = df.groupby(['subcategory', 'user']).first().reset_index()['subcategory'].value_counts().head(15)[1:]
    df.set_index('date', inplace=True)
    
    stats_dict = {'subcategory': [], 'percent': [], 'date': []}
    start_date, end_date = pd.Timestamp(start) + timedelta(days=window_days), pd.Timestamp(end)

    for offset in range((end_date - start_date).days):
        center_date = start_date + timedelta(days=offset)
        from_date, to_date = center_date - timedelta(days=window_days), center_date + timedelta(days=window_days)
        df_window = df.loc[from_date:to_date]
        pmdd_users = df_window['user'].unique()

        user_counts = df_window[df_window['user'].isin(pmdd_users)]
        condition_counts = user_counts.groupby(['subcategory', 'user']).first().reset_index()['subcategory'].value_counts()
        condition_counts = condition_counts[condition_counts.index.isin(selected_conditions.index)]
        user_total = user_counts['user'].nunique()

        for condition in selected_conditions.index:
            percent = condition_counts.get(condition, 0) / user_total * 100 if user_total else 0
            stats_dict['subcategory'].append(condition)
            stats_dict['percent'].append(percent)
            stats_dict['date'].append(center_date)

    df_stats = pd.DataFrame(stats_dict)
    return df_stats, selected_conditions.index

def plot_pmdd_associations(df_stats: pd.DataFrame, selected_conditions, output_path: str):
    df_stats['date'] = pd.to_datetime(df_stats['date'])
    df_stats['subcategory'] = df_stats['subcategory'].apply(lambda x: x + 's' if not x.endswith('s') else x)

    df_stats['percent_sliding'] = df_stats.groupby('subcategory')['percent']\
                                          .transform(lambda x: x.rolling(6, min_periods=2).mean().shift(-1))

    fig = plt.figure(figsize=(12, 5))
    gs = GridSpec(1, 2, figure=fig, wspace=0.1, width_ratios=[2, 3.5])

    # Left Figure: Bar chart
    ax_total = fig.add_subplot(gs[0])
    sns.barplot(data=df_stats, x='percent', y='subcategory', ax=ax_total, gap=0.1, capsize=0.2,
                errwidth=1.2, errcolor='black', order=selected_conditions, palette=COLOR_PALETTE, alpha=0.65)
    ax_total.set_yticklabels(selected_conditions, fontsize=18)
    ax_total.set_xlabel('(%) r/PMDD users co-posting')
    ax_total.set_ylabel('')
    ax_total.spines[['top', 'right', 'left']].set_visible(False)

    # Right Figure: Timeline chart
    ax_timeline = fig.add_subplot(gs[1])
    for condition in selected_conditions:
        condition_data = df_stats[df_stats['subcategory'] == condition]
        sns.lineplot(data=condition_data, x='date', y='percent_sliding',
                     label=condition, ax=ax_timeline, color=COLOR_PALETTE[condition], alpha=0.5, linewidth=2)
        sns.scatterplot(data=condition_data, x='date', y='percent_sliding',
                        ax=ax_timeline, color=COLOR_PALETTE[condition], s=8, alpha=0.3)

    ax_timeline.set_xticks(pd.to_datetime(['2015-04-01', '2017-01-01', '2019-01-01', '2021-01-01', '2023-01-01']))
    ax_timeline.set_xticklabels(['2015', '2017', '2019', '2021', '2023'])
    ax_timeline.set_yticks(np.arange(0, 21, 5))
    ax_timeline.set_yticklabels([f'{i}%' for i in np.arange(0, 21, 5)], fontsize=18)
    ax_timeline.set_xlabel('6-month sliding window intervals')
    ax_timeline.set_ylabel('')
    ax_timeline.spines[['top', 'right']].set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def summarize_conditions(df_stats: pd.DataFrame, selected_conditions):
    for condition in selected_conditions:
        series = df_stats[df_stats['subcategory'] == condition]['percent_sliding'].dropna()
        if not series.empty:
            print(f"\nCondition: {condition}")
            mean, ci_low, ci_high = mean_confidence_interval(series)
            print(f"Mean: {mean:.2f}% | 95% CI: ({ci_low:.2f}%, {ci_high:.2f}%)")
            
            adf_result = adfuller(series, autolag='AIC')
            print("--- ADF Test ---")
            print(f"ADF Statistic: {adf_result[0]:.4f}, p-value: {adf_result[1]:.4f}")
            if adf_result[1] < 0.05:
                print("Result: Series is stationary")

            print("--- KPSS Test ---")
            kpss_test(series)

if __name__ == "__main__":
    input_file = "data/posts_pmdd.csv"
    output_csv = "data/results/associations_pmdd.csv"
    output_figure = "figures/fig1.png"

    df_pmdd = load_and_prepare_data(input_file)
    df_stats, selected_conditions = compute_sliding_statistics(df_pmdd)
    df_stats.to_csv(output_csv, index=False)

    summarize_conditions(df_stats, selected_conditions)
    plot_pmdd_associations(df_stats, selected_conditions, output_figure)
