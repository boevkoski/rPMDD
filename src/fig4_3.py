import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Plot settings ---
plt.rcParams["font.family"] = "Garamond"
plt.rcParams.update({'font.size': 20})

# --- Load Data ---
df = pd.read_csv("data/posts_pmdd_enriched.csv")
df = df[df['subreddit'] == 'PMDD']
df = df[(df['post_date'] >= '2016-01-01') & (df['post_date'] <= '2022-12-01')]
df['lemma'] = df['lemma'].str.lower()

# --- Load Symptom Type Mapping ---
symptoms_dict = pd.read_csv("data/raw/symptoms_keywords.csv").set_index('keyword')['label'].dropna().to_dict()

# --- Filter for Symptom Mentions ---
df_symptoms = df[df['ner'] == 'SIGN_SYMPTOM'].copy()
df_symptoms['symptom_type'] = df_symptoms['lemma'].map(symptoms_dict)
df_symptoms = df_symptoms.dropna(subset=['symptom_type'])

# --- Prepare Monthly Data ---
df_symptoms = df_symptoms.drop_duplicates(subset=['post_id', 'symptom_type'])
df_symptoms['month'] = pd.to_datetime(df_symptoms['post_date'].str[:7])
df_symptoms['time'] = df_symptoms['month'].dt.to_period('M')

monthly_counts = df_symptoms.groupby(['time', 'symptom_type']).size().reset_index(name='count')

# --- Pivot: time as rows, symptom types as columns ---
pivot_df = monthly_counts.pivot(index='time', columns='symptom_type', values='count').fillna(0)

# --- Normalize: percentage share of each symptom type per month ---
percent_df = pivot_df.div(pivot_df.sum(axis=1), axis=0)

# --- Focus on top 6 most discussed types ---
total_counts = pivot_df.sum().sort_values(ascending=False)
top_6_symptom_types = total_counts.index[:6]  # top 6, not 7
percent_df_top6 = percent_df[top_6_symptom_types]

# --- Apply 6-month rolling average ---
percent_df_smooth = percent_df_top6.rolling(window=6, min_periods=1).mean()
percent_df_smooth = percent_df_smooth[6:]  # skip initial short window

# --- Plotting ---
fig, ax = plt.subplots(figsize=(15, 7.5))
for column in percent_df_smooth.columns:
    ax.plot(percent_df_smooth.index.to_timestamp(), percent_df_smooth[column], label=column, linewidth=2)

ax.set_title('Smoothed Monthly Dynamics of Top 6 Symptom Type Shares (6-month window)', fontsize=18)
ax.set_ylabel('Share of Symptom Types')
ax.set_xlabel('Month')

ax.legend(title='', loc='upper right', fontsize=15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/fig4d_symptom_dynamics_smoothed.png", dpi=300)
