import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
import scipy

# --- Plot settings ---
plt.rcParams["font.family"] = "Garamond"
plt.rcParams.update({'font.size': 16})

# --- Confidence Interval Helper ---
def mean_confidence_interval(data, confidence=0.95):
    a = np.array(data)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., len(a)-1)
    return m, m - h, m + h

# --- Load Data ---
df = pd.read_csv("data/posts_pmdd_enriched.csv")
df = df[df['subreddit'] == 'PMDD']
df['post_date'] = pd.to_datetime(df['post_date'])
df['lemma'] = df['lemma'].str.lower()
pmdd_users = df['user'].nunique()

print("Loaded and filtered enriched PMDD post data...")

# --- Load Medication Mapping Dictionary (Manual Annotations) ---
medications_dict = pd.read_csv("data/raw/medications_keywords.csv").set_index('keyword')['label'].dropna().to_dict()

# --- Extract Medication Mentions ---
df_meds = df[df['ner'] == 'MEDICATION'].copy()
df_meds['medication_type'] = df_meds['lemma'].map(medications_dict)
df_meds = df_meds.dropna(subset=['medication_type'])
df_meds = df_meds.drop_duplicates(subset=['post_id', 'medication_type'])

print("Mapped medications to categories.")

# --- Sliding Time Window Setup ---
start_date = pd.Timestamp('2015-04-01') + pd.Timedelta(days=90)
end_date = pd.Timestamp('2023-08-15')
delta_days = (end_date - start_date).days

# --- Aggregate by Medication Type ---
print("Aggregating 6-month windows for medication types...")
results_type = []

for offset in tqdm(range(delta_days)):
    date = start_date + pd.Timedelta(days=offset)
    from_date = date - pd.Timedelta(days=180)
    to_date = date

    posts_window = df[(df['post_date'] >= from_date) & (df['post_date'] <= to_date)]
    meds_window = df_meds[(df_meds['post_date'] >= from_date) & (df_meds['post_date'] <= to_date)]
    num_users = posts_window['user'].nunique()

    for med_type in meds_window['medication_type'].unique():
        users_with_med = meds_window[meds_window['medication_type'] == med_type]['user'].nunique()
        percent = users_with_med / num_users * 100 if num_users > 0 else 0
        results_type.append({
            'post_date': date,
            'medication_type': med_type,
            'num_users_medication': users_with_med,
            'num_users': num_users,
            'percent': percent
        })

df_results_type = pd.DataFrame(results_type)

# --- Summary Stats ---
print("\nMedication Type Usage Summary:")
for med in df_results_type['medication_type'].unique():
    vals = df_results_type[df_results_type['medication_type'] == med]['percent']
    mean, ci_low, ci_high = mean_confidence_interval(vals)
    print(f"{med}: Mean = {mean:.2f}%, CI Width = {ci_high - ci_low:.2f}%")

# --- Plot Medication Type Usage ---
fig, ax = plt.subplots(figsize=(6, 4.5))
type_order = df_results_type.groupby('medication_type')['percent'].mean().sort_values(ascending=False).index
sns.barplot(data=df_results_type, y='medication_type', x='percent', order=type_order,
            ax=ax, color='darkred', alpha=0.8, capsize=0.2, errwidth=1.2, errcolor='black', orient='h')
ax.set_ylabel('')
ax.set_xlabel('(%) of r/PMDD users')
ax.spines[['top', 'right']].set_visible(False)
ax.grid(True, linestyle='--', alpha=0.7)
fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/fig4a_medication_categories.png", dpi=300)

# --- Select Individual Medications for Tracking ---
selected_lemmas = [
    'prozac', 'zoloft', 'yaz', 'lexapro', 'magnesium', 'progesterone',
    'wellbutrin', 'sertraline', 'fluoxetine', 'mirena', 'vitex', 'lupron'
]

rename_map = {
    'prozac': 'Prozac (Fluoxetine, SSRI)',
    'zoloft': 'Zoloft (Sertraline, SSRI)',
    'yaz': 'Yaz (Contraceptive)',
    'lexapro': 'Lexapro (Escitalopram, SSRI)',
    'magnesium': 'Magnesium',
    'progesterone': 'Progesterone',
    'wellbutrin': 'Wellbutrin (Bupropion, NDRI)',
    'vitex': 'Vitex (Chaste Tree)',
    'mirena': 'Mirena (IUD)',
    'lupron': 'Lupron (GnRH agonist)',
    'sertraline': 'Sertraline (SSRI)',
    'fluoxetine': 'Fluoxetine (SSRI)'
}

df_selected = df_meds[df_meds['lemma'].isin(selected_lemmas)].copy()
df_selected['lemma'] = df_selected['lemma'].map(rename_map)

# --- Aggregate by Selected Medications ---
print("\nAggregating selected medications over time...")
results_meds = []

for offset in tqdm(range(delta_days)):
    date = start_date + pd.Timedelta(days=offset)
    from_date = date - pd.Timedelta(days=180)
    to_date = date

    posts_window = df[(df['post_date'] >= from_date) & (df['post_date'] <= to_date)]
    meds_window = df_selected[(df_selected['post_date'] >= from_date) & (df_selected['post_date'] <= to_date)]
    num_users = posts_window['user'].nunique()

    for med in df_selected['lemma'].unique():
        users_with_med = meds_window[meds_window['lemma'] == med]['user'].nunique()
        percent = users_with_med / num_users * 100 if num_users > 0 else 0
        results_meds.append({
            'post_date': date,
            'lemma': med,
            'num_users_medication': users_with_med,
            'num_users': num_users,
            'percent': percent
        })

df_results_meds = pd.DataFrame(results_meds)

# --- Summary Stats for Medications ---
print("\nIndividual Medication Usage Summary:")
for med in df_results_meds['lemma'].unique():
    vals = df_results_meds[df_results_meds['lemma'] == med]['percent']
    mean, ci_low, ci_high = mean_confidence_interval(vals)
    print(f"{med}: Mean = {mean:.2f}%, CI Width = {ci_high - ci_low:.2f}%")

# --- Plot Selected Medications ---
fig, ax = plt.subplots(figsize=(6, 4.5))
med_order = df_results_meds.groupby('lemma')['percent'].mean().sort_values(ascending=False).index
sns.barplot(data=df_results_meds, y='lemma', x='percent', order=med_order,
            ax=ax, color='#e0e014', alpha=0.8, capsize=0.2, errwidth=1.2, errcolor='black', orient='h')
ax.set_ylabel('')
ax.set_xlabel('(%) of r/PMDD users')
ax.spines[['top', 'right']].set_visible(False)
ax.grid(True, linestyle='--', alpha=0.7)
fig.tight_layout()
fig.savefig("figures/fig4b_selected_medications.png", dpi=300)
