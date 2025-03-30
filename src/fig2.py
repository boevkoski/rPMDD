import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import timedelta
from matplotlib.gridspec import GridSpec
from scipy import stats
import os

# Plot settings
plt.rcParams["font.family"] = "Garamond"
plt.rcParams.update({'font.size': 16})

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

def mean_ci(data, confidence=0.95):
    a = np.array(data)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., len(a)-1)
    return m, m - h, m + h

def compute_pmdd_user_activity(df_pmdd, user_first_dates, conditions, window=90):
    before, after, before_counts, after_counts = [], [], [], []
    for user, date in tqdm(user_first_dates.items(), desc="Computing PMDD user activity"):
        if date < pd.Timestamp('2015-06-01') or date > pd.Timestamp('2023-08-01'):
            continue

        df_user = df_pmdd[(df_pmdd['user'] == user) & (df_pmdd['subreddit'] != 'PMDD')]
        df_before = df_user[(df_user['date'] < date) & (df_user['date'] > date - pd.Timedelta(days=window))]
        df_after = df_user[(df_user['date'] > date) & (df_user['date'] < date + pd.Timedelta(days=window))]

        before_counts.append(len(df_before))
        after_counts.append(len(df_after))

        before_vec = df_before['subcategory'].value_counts().reindex(conditions, fill_value=0)
        after_vec = df_after['subcategory'].value_counts().reindex(conditions, fill_value=0)

        before.append(before_vec)
        after.append(after_vec)

    df_before = pd.DataFrame(before).fillna(0)
    df_after = pd.DataFrame(after).fillna(0)
    df_counts = pd.DataFrame({
        'counts': before_counts + after_counts,
        'type': ['Before r/PMDD'] * len(before_counts) + ['After r/PMDD'] * len(after_counts)
    })

    return df_before, df_after, df_counts, dict(zip(user_first_dates.keys(), zip(before_counts, after_counts)))

def simulate_null(df_all, user_date_volumes, conditions, n_runs=1000, window=90):
    df_all['date'] = pd.to_datetime(df_all['date'])
    df_all = df_all[df_all['subcategory'].isin(conditions)]
    df_all = df_all[df_all['subreddit'] != 'PMDD']

    null_results = []

    for _ in tqdm(range(n_runs), desc="Simulating null model"):
        sim_diffs = []

        for user, (first_date, (n_before, n_after)) in user_date_volumes.items():
            if first_date < pd.Timestamp('2015-06-01') or first_date > pd.Timestamp('2023-08-01'):
                continue

            pre_window = (first_date - timedelta(days=window), first_date)
            post_window = (first_date, first_date + timedelta(days=window))

            pre_pool = df_all[(df_all['date'] > pre_window[0]) & (df_all['date'] < pre_window[1])]
            post_pool = df_all[(df_all['date'] > post_window[0]) & (df_all['date'] < post_window[1])]

            pre_sample = pre_pool.sample(n=n_before, replace=True) if n_before > 0 else pd.DataFrame(columns=df_all.columns)
            post_sample = post_pool.sample(n=n_after, replace=True) if n_after > 0 else pd.DataFrame(columns=df_all.columns)

            pre_vec = pre_sample['subcategory'].value_counts().reindex(conditions, fill_value=0)
            post_vec = post_sample['subcategory'].value_counts().reindex(conditions, fill_value=0)

            pre_rate = pre_vec / n_before if n_before > 0 else pd.Series(0, index=conditions)
            post_rate = post_vec / n_after if n_after > 0 else pd.Series(0, index=conditions)

            sim_diffs.append((post_rate - pre_rate) * 100)

        if sim_diffs:
            null_results.append(pd.DataFrame(sim_diffs).mean())

    return pd.DataFrame(null_results)

def plot_did(diff_pmdd, diff_null_mean, ci_pmdd, pvals, sorted_conditions, df_counts, outpath):
    fig = plt.figure(figsize=(8, 9.5))
    gs = GridSpec(2, 1, figure=fig, hspace=0.2, height_ratios=[1, 11])

    # Post counts bar chart
    ax_counts = fig.add_subplot(gs[0])
    sns.barplot(data=df_counts, x='counts', y='type', ax=ax_counts, palette=['black'] * 2,
                capsize=0.5, errwidth=1.2, errcolor='gray', alpha=0.95)
    ax_counts.spines[['top', 'right', 'left']].set_visible(False)
    ax_counts.set_xlabel('posts per user')
    ax_counts.set_ylabel('')
    ax_counts.grid(True, linestyle='--', alpha=0.7)

    # DiD main chart
    ax_main = fig.add_subplot(gs[1])
    y_pos = np.arange(len(sorted_conditions))
    ax_main.barh(y_pos, diff_null_mean, alpha=0.3, color='gray', edgecolor='black', height=0.75)
    ax_main.barh(y_pos, diff_pmdd[sorted_conditions], xerr=ci_pmdd[sorted_conditions],
                 color=[COLOR_PALETTE[c] for c in sorted_conditions], height=0.5, capsize=5)
    ax_main.set_yticks(y_pos)
    ax_main.set_yticklabels(sorted_conditions)
    ax_main.set_xlim(-0.85, 0.85)
    ax_main.set_xlabel('(%) difference in co-posting (90 days before vs. after first post on r/PMDD)')
    ax_main.axvline(0, color='black', linewidth=0.5)
    ax_main.grid(True, linestyle='--', alpha=0.7)

    for i, cond in enumerate(sorted_conditions):
        val = diff_pmdd[cond]
        p = pvals[cond]
        ax_main.text(val + 0.2 if val > 0 else val - 0.2, i, f"{val:+.3f}%", ha='center', va='center', fontsize=14)
        ax_main.text(0.9, i, f"$p<{0.001}$" if p < 0.001 else f"$p={p:.3f}$", ha='left', va='center', fontsize=11)

    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    df_pmdd = pd.read_csv("data/posts_pmdd.csv")
    df_all = pd.read_csv("data/posts_MHSN.csv")

    df_pmdd['subcategory'] = df_pmdd['subcategory'].apply(lambda x: x + 's' if not x.endswith('s') else x)
    df_pmdd['date'] = pd.to_datetime(df_pmdd['date'])

    selected_conditions = df_pmdd.groupby(['subcategory', 'user']).first().reset_index()['subcategory'] \
                                 .value_counts().head(11).index[1:]

    user_first_dates = df_pmdd[df_pmdd['subreddit'] == 'PMDD'].groupby('user')['date'].min().to_dict()

    # Get PMDD user activity and volume dict
    df_before, df_after, df_counts, user_date_volumes = compute_pmdd_user_activity(
        df_pmdd, user_first_dates, selected_conditions)

    # Compute % difference for PMDD
    before_rates = df_before.sum() / df_before.count()
    after_rates = df_after.sum() / df_after.count()
    diff_pmdd = (after_rates - before_rates) * 100
    ci_pmdd = (df_after / df_after.count() - df_before / df_before.count()).apply(mean_ci).apply(lambda x: (x[2] - x[1]) / 2)

    # Simulate null model using full MHSN
    null_diffs_df = simulate_null(df_all, {u: (d, user_date_volumes[u]) for u, d in user_first_dates.items() if u in user_date_volumes}, selected_conditions)
    diff_null_mean = null_diffs_df.mean()

    sorted_conditions = diff_pmdd.abs().sort_values().index.tolist()

    # DiD p-values
    pvals = {
        cond: stats.ttest_1samp(null_diffs_df[cond] - diff_pmdd[cond], 0, alternative='two-sided')[1]
        for cond in selected_conditions
    }

    os.makedirs("figures", exist_ok=True)
    plot_did(diff_pmdd, diff_null_mean[sorted_conditions], ci_pmdd, pvals, sorted_conditions, df_counts,
             "figures/fig2.png")

if __name__ == "__main__":
    main()
