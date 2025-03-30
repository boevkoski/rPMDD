import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import scipy
import os

# Plot settings
plt.rcParams["font.family"] = "Garamond"
plt.rcParams.update({'font.size': 18})

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h

# ---------- Load Data ----------
df = pd.read_csv('data/posts_pmdd_enriched.csv')
df = df[df['subreddit'] == 'PMDD']
df['post_date'] = pd.to_datetime(df['post_date'])
df['lemma'] = df['lemma'].str.lower()

# Load symptom type dictionary of the manually annotated symptoms
symptoms_dict = pd.read_csv('data/raw/symptoms_keywords.csv').set_index('keyword')['label'].dropna().to_dict()

# Filter for symptom entities
df_symptoms = df[df['ner'] == 'SIGN_SYMPTOM'].copy()
df_symptoms['symptom_type'] = df_symptoms['lemma'].apply(lambda x: symptoms_dict.get(x, np.nan))
df_symptoms.dropna(subset=['symptom_type'], inplace=True)

# Remove duplicate symptom mentions per post
df_symptoms = df_symptoms.drop_duplicates(subset=['post_id', 'lemma'])

# ---------- Sliding Window Aggregation ----------
start_date = pd.Timestamp('2015-04-01') + pd.Timedelta(days=90)
end_date = pd.Timestamp('2023-08-15')
delta_days = (end_date - start_date).days

results_type = []
results_detail = []

print("Calculating 6-month symptom prevalence over time...")
for offset in tqdm(range(delta_days)):
    date = start_date + pd.Timedelta(days=offset)
    from_date = date - pd.Timedelta(days=180)
    to_date = date

    window_posts = df[(df['post_date'] >= from_date) & (df['post_date'] < to_date)]
    symptom_window = df_symptoms[(df_symptoms['post_date'] >= from_date) & (df_symptoms['post_date'] < to_date)]
    num_users = window_posts['user'].nunique()

    # Symptom types
    type_counts = symptom_window.groupby('symptom_type')['user'].nunique() / num_users * 100
    for stype, percent in type_counts.items():
        results_type.append({
            'post_date': date,
            'symptom_type': stype,
            'num_users_type': symptom_window[symptom_window['symptom_type'] == stype]['user'].nunique(),
            'num_users': num_users,
            'percent': percent
        })

    # Specific symptoms
    symptom_counts = symptom_window.groupby('lemma')['user'].nunique() / num_users * 100
    for symptom, percent in symptom_counts.items():
        results_detail.append({
            'post_date': date,
            'symptom': symptom,
            'num_users_symptom': symptom_window[symptom_window['lemma'] == symptom]['user'].nunique(),
            'num_users': num_users,
            'percent': percent
        })

df_results_type = pd.DataFrame(results_type)
df_results_detail = pd.DataFrame(results_detail)

# ---------- Summary Stats ----------
print("\nSymptom Type Averages & CI:")
for symptom_type in df_results_type['symptom_type'].unique():
    subset = df_results_type[df_results_type['symptom_type'] == symptom_type]['percent']
    mean_val = subset.mean()
    ci = mean_confidence_interval(subset)
    print(f"{symptom_type}: Mean = {mean_val:.2f}%, CI width = {ci[2] - ci[1]:.2f}%")

print("\nTop Symptom Averages & CI:")
for symptom in df_results_detail['symptom'].unique():
    subset = df_results_detail[df_results_detail['symptom'] == symptom]['percent']
    mean_val = subset.mean()
    ci = mean_confidence_interval(subset)
    print(f"{symptom}: Mean = {mean_val:.2f}%, CI width = {ci[2] - ci[1]:.2f}%")

# ---------- Plotting ----------
fig = plt.figure(figsize=(16, 6))
gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 6], hspace=0.1, wspace=0.1, width_ratios=[2.5, 1])

# (A) Text box with DSM-5 criteria
ax_text = fig.add_subplot(gs[:, 0])
ax_text.axis('off')
ax_text.text(0.5, 1.1, 'Diagnostic Criteria for PMDD in DSM-5',
             ha='center', va='top', fontsize=20, fontweight='bold', color='black', transform=ax_text.transAxes)

dsm_lines = [
    "(B1): Marked affective lability (e.g., mood swings, feeling suddenly sad or tearful)",
    "(B2): Marked irritability or anger or increased interpersonal conflicts",
    "(B3): Marked depressed mood, feelings of hopelessness, or self-deprecating thoughts",
    "(B4): Marked anxiety, tension, and/or feelings of being keyed up or on edge",
    """(CX): One (or more) of the following symptoms must additionally be present
         to reach a total of 5 symptoms when combined with symtpoms from B1-B4:
    (C1): decreased interest in usual activities
    (C2): subjective difficulty in concentration
    (C3): lethargy, easy fatigability, or marked lack of energy
    (C4): change in appetite, overeating, or specific food cravings
    (C5): hypersomnia or insomnia (sleep disturbance)
    (C6): a sense of being overwhelmed or out of control
    (C7): physical symptoms (e.g., breast tenderness, joint/muscle pain, bloating, weight gain)"""
]

for i, line in enumerate(dsm_lines):
    ax_text.text(0.01, 0.95 - i * 0.11, line, ha='left', va='top', fontsize=17.5, color='black', transform=ax_text.transAxes)

# (B1) Barplot of symptom types
ax_type = fig.add_subplot(gs[0, 1])
type_order = ['B', 'C']
sns.barplot(data=df_results_type, y='symptom_type', x='percent', order=type_order,
            color='black', ax=ax_type, alpha=0.8, capsize=0.2, errwidth=1.2, errcolor='lightgray', orient='h')
ax_type.set_ylabel('')
ax_type.set_xlabel('')
ax_type.set_xlim(0, 58)
ax_type.set_xticks([])
ax_type.spines[['top', 'right']].set_visible(False)

# (B2) Barplot of specific symptoms
ax_symptoms = fig.add_subplot(gs[1, 1])
top_order = df_results_detail.groupby('symptom')['percent'].mean().sort_values(ascending=False).index
sns.barplot(data=df_results_detail, y='symptom', x='percent', order=top_order,
            color='darkslateblue', ax=ax_symptoms, alpha=0.7, capsize=0.2, errwidth=1.2, errcolor='black', orient='h')
ax_symptoms.set_ylabel('')
ax_symptoms.set_xlabel('(%) of r/PMDD users')
ax_symptoms.set_xlim(0, 58)
ax_symptoms.spines[['top', 'right']].set_visible(False)

# ---------- Save Output ----------
os.makedirs("figures", exist_ok=True)
fig.tight_layout()
fig.savefig("figures/fig3_symptoms_summary.png", dpi=300, bbox_inches='tight')
