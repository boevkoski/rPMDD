import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import timedelta
from autorank import autorank, create_report, plot_stats
import os

# --- Plot settings ---
plt.rcParams["font.family"] = "Garamond"
plt.rcParams.update({'font.size': 18})

# --- Load PMDD data ---
df_pmdd = pd.read_csv("data/posts_pmdd.csv")
df_pmdd['date'] = pd.to_datetime(df_pmdd['date'])
df_pmdd.set_index('date', inplace=True)

print("Loaded PMDD data")
print("Unique users:", df_pmdd['user'].nunique())

# --- Get most frequent conditions (top 10, excluding 1st) ---
top_conditions = df_pmdd.groupby(['subcategory', 'user'])\
                        .first().reset_index()['subcategory']\
                        .value_counts().head(11).index[1:]

# --- Robustness loop over window sizes and strides ---
results = {
    'subcategory': [],
    'window_size': [],
    'window_stride': [],
    'percent': [],
    'date': []
}

print("Running robustness checks over various window sizes and strides...")

for window_size in [360, 270, 180, 90, 60, 30]:
    for window_stride in [window_size, window_size // 2, window_size // 3, window_size // 6, window_size // 10]:

        start_date = pd.Timestamp('2015-01-01') + pd.Timedelta(days=window_size)
        end_date = pd.Timestamp('2023-12-15') - pd.Timedelta(days=window_size)
        delta = (end_date - start_date).days
        total_windows = delta // window_stride

        for offset in tqdm(range(0, delta, window_stride), total=total_windows, desc=f"{window_size=} {window_stride=}"):
            mid_date = start_date + timedelta(days=offset)
            from_date = mid_date - pd.Timedelta(days=90)
            to_date = mid_date + pd.Timedelta(days=90)

            df_window = df_pmdd.loc[(df_pmdd.index >= from_date) & (df_pmdd.index <= to_date)]
            user_ids = df_window['user'].unique()

            user_condition_counts = df_window[df_window['user'].isin(user_ids)]\
                .groupby(['subcategory', 'user'])\
                .first().reset_index()['subcategory'].value_counts()

            user_condition_counts = user_condition_counts[user_condition_counts.index.isin(top_conditions)]
            percentages = user_condition_counts / len(user_ids) * 100 if len(user_ids) > 0 else 0

            for cond in top_conditions:
                results['subcategory'].append(cond)
                results['window_size'].append(window_size)
                results['window_stride'].append(window_stride)
                results['percent'].append(percentages.get(cond, 0))
                results['date'].append(mid_date)

# --- Save and process results ---
os.makedirs("data/results", exist_ok=True)
df_results = pd.DataFrame(results)
df_results.to_csv("data/results/results_robustness_associations_windows.csv", index=False)

print("Saved bootstrap results with varying window configurations.")

# --- Prepare data for statistical ranking ---
df_avg = df_results.groupby(['date', 'subcategory']).mean(numeric_only=True).reset_index()
df_pivot = df_avg.pivot(index='date', columns='subcategory', values='percent').dropna()

# --- Run autorank statistical comparison ---
print("Running autorank comparison across subcategories...")
rank_result = autorank(data=df_pivot, alpha=0.05, verbose=True)
create_report(rank_result)

# --- Plot final robustness figure ---
plot_stats(rank_result)
plt.title("Robustness of Comorbidity Estimates (Window Variations)", fontsize=18)
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/figS2.png", dpi=300, bbox_inches="tight")
