import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss, acf
from statsmodels.tools.sm_exceptions import InterpolationWarning
from datetime import timedelta
from tqdm import tqdm
from collections import defaultdict
from matplotlib.patches import Patch
import scipy
import warnings
import os

# --- Settings ---
warnings.simplefilter('ignore', InterpolationWarning)
plt.rcParams["font.family"] = "Garamond"
plt.rcParams.update({'font.size': 16})

def mean_confidence_interval(data, confidence=0.95):
    a = np.array(data)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., len(a)-1)
    return m, m - h, m + h

color_palette = {
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

# --- Load and prepare data ---
df_enriched = pd.read_csv("data/posts_pmdd_enriched.csv")
df_enriched = df_enriched[df_enriched['subreddit'] == 'PMDD']
df_enriched['post_date'] = pd.to_datetime(df_enriched['post_date'])
df_enriched['lemma'] = df_enriched['lemma'].str.lower()

# Load medication type dictionary
medications_dict = pd.read_csv("data/raw/medications_keywords.csv").set_index('keyword')['label'].dropna().to_dict()

# --- Filter for medication mentions ---
df_meds = df_enriched[df_enriched['ner'] == 'MEDICATION'].copy()
df_meds['medication_type'] = df_meds['lemma'].map(medications_dict)
df_meds = df_meds.dropna(subset=['medication_type'])
df_meds = df_meds.drop_duplicates(subset=['post_id'])

# Get users by medication class
df_ssris = df_meds[df_meds['medication_type'] == 'antidepressants (SSRI)']
df_contraceptives = df_meds[df_meds['medication_type'] == 'contraceptives']

ssri_users = df_ssris['user'].unique()
contraceptive_users = df_contraceptives['user'].unique()

# --- Load PMDD data for condition matching ---
df_pmdd = pd.read_csv("data/posts_by_pmdd.csv")
df_pmdd['date'] = pd.to_datetime(df_pmdd['date'])
df_pmdd['subcategory'] = df_pmdd['subcategory'].apply(lambda x: x + 's' if not x.endswith('s') else x)
df_pmdd = df_pmdd[df_pmdd['category'] != 'Other Conditions That May Be a Focus of Clinical Attention']
df_pmdd.set_index('date', inplace=True)

selected_conditions = [
    "Depressive Disorders",
    "Anxiety Disorders",
    "Attention-Deficit/Hyperactivity Disorders",
    "Substance-Related Disorders",
    "Post-Traumatic Stress Disorders",
    "Cluster B Personality Disorders",
    "Bipolar and Related Disorders",
    "Autism Spectrum Disorders",
    "Obsessive-Compulsive Disorders",
    "Social Anxiety Disorders"
]

# --- Aggregate % comorbidities per group across time ---
start_date = pd.Timestamp('2015-04-01') + timedelta(days=90)
end_date = pd.Timestamp('2023-08-15')
delta = (end_date - start_date).days

statistics = {
    'subcategory': [],
    'medication': [],
    'percent': [],
    'date': []
}

print("Computing time-varying disorder comorbidity for SSRI and contraceptive users...")
for day in tqdm(range(delta)):
    date = start_date + timedelta(days=day)
    from_date = date - timedelta(days=90)
    to_date = date + timedelta(days=90)

    df_window = df_pmdd.loc[(from_date <= df_pmdd.index) & (df_pmdd.index <= to_date)]

    df_ssri = df_window[df_window['user'].isin(ssri_users)]
    df_contra = df_window[df_window['user'].isin(contraceptive_users)]

    ssri_counts = df_ssri.groupby(['subcategory', 'user']).first().reset_index()['subcategory'].value_counts()
    contra_counts = df_contra.groupby(['subcategory', 'user']).first().reset_index()['subcategory'].value_counts()

    ssri_percents = ssri_counts[selected_conditions] / df_ssri['user'].nunique() * 100 if df_ssri['user'].nunique() else 0
    contra_percents = contra_counts[selected_conditions] / df_contra['user'].nunique() * 100 if df_contra['user'].nunique() else 0

    for cond in selected_conditions:
        statistics['subcategory'].append(cond)
        statistics['medication'].append('ssris')
        statistics['percent'].append(ssri_percents.get(cond, 0))
        statistics['date'].append(date)

        statistics['subcategory'].append(cond)
        statistics['medication'].append('contraceptives')
        statistics['percent'].append(contra_percents.get(cond, 0))
        statistics['date'].append(date)

df_stats = pd.DataFrame(statistics)
os.makedirs("data/results", exist_ok=True)
df_stats.to_csv("data/results/result4b_disorder_by_medication_group.csv", index=False)

# --- Ratio statistics ---
print("\nCondition-specific SSRIs / Contraceptives ratio CIs:")
for cond in selected_conditions:
    c_vals = df_stats[(df_stats['subcategory'] == cond) & (df_stats['medication'] == 'contraceptives')]['percent'].replace(0, 1)
    s_vals = df_stats[(df_stats['subcategory'] == cond) & (df_stats['medication'] == 'ssris')]['percent'].replace(0, 1)
    ratio = s_vals.values / c_vals.values
    print(f"{cond}: {mean_confidence_interval(ratio)}")

# --- Final Figure ---
fig, ax = plt.subplots(figsize=(9, 5))
hue_order = ['contraceptives', 'ssris']

sns.barplot(
    data=df_stats, x='percent', y='subcategory',
    hue='medication', order=selected_conditions, hue_order=hue_order,
    ax=ax, capsize=0.2, errwidth=1.2, errcolor='black', alpha=0.5, dodge=True
)

# Apply hatching manually
tick_positions = ax.get_yticks()
tick_labels = [label.get_text() for label in ax.get_yticklabels()]
groups = defaultdict(list)

for patch in ax.patches:
    center = patch.get_y() + patch.get_height() / 2
    distances = [abs(center - pos) for pos in tick_positions]
    idx = distances.index(min(distances))
    subcat = tick_labels[idx]
    groups[subcat].append(patch)

for subcat in groups:
    bars = sorted(groups[subcat], key=lambda p: p.get_x())
    for i, patch in enumerate(bars):
        patch.set_hatch('////' if i == 0 else '')
        patch.set_facecolor(color_palette[subcat])
        patch.set_edgecolor('dimgray')

# Custom legend
legend_handles = [
    Patch(facecolor='white', hatch='////', edgecolor='grey', label='contraceptives'),
    Patch(facecolor='white', hatch='', edgecolor='grey', label='antidepressants (SSRI)')
]
ax.legend(handles=legend_handles, title='User group', title_fontsize=14, fontsize=14, loc='lower right')

ax.set_ylabel('')
ax.set_xlabel('(%) r/PMDD users co-posting')
plt.tight_layout()
sns.despine()

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/fig4c_disorder_comorbidity_by_med_group.png", dpi=300, bbox_inches='tight')
