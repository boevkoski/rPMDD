import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

# --- Plot settings ---
plt.rcParams["font.family"] = "Garamond"
plt.rcParams.update({'font.size': 28})

# --- Load Data ---
df = pd.read_csv("data/posts_pmdd_enriched.csv")
df = df[df['subreddit'] == 'PMDD']
df['post_date'] = pd.to_datetime(df['post_date'])
df['lemma'] = df['lemma'].str.lower()

# --- Load mapping dictionaries (manual annotations) ---
symptoms_dict = pd.read_csv("data/raw/symptoms_keywords.csv").set_index('keyword')['label'].dropna().to_dict()
medications_dict = pd.read_csv("data/raw/medications_keywords.csv").set_index('keyword')['label'].dropna().to_dict()

# --- Assign category type to each entity ---
def get_category(x):
    if x['ner'] == 'SIGN_SYMPTOM':
        return symptoms_dict.get(x['lemma'], np.nan)
    elif x['ner'] == 'MEDICATION':
        return medications_dict.get(x['lemma'], np.nan)
df['type'] = df.apply(get_category, axis=1)

# --- Set of unique posts ---
posts = set(df['post_id'].unique())
print("Processed all entities with type mapping.")

# --- Medication-Medication Co-occurrence ---
df_meds = df[df['ner'] == 'MEDICATION'].dropna(subset=['type'])
med_types = df_meds['type'].unique()

odds_meds = np.ones((len(med_types), len(med_types)))
pvals_meds = np.ones_like(odds_meds)

for i, type1 in tqdm(enumerate(med_types), total=len(med_types), desc="Medication co-occurrence"):
    for j, type2 in enumerate(med_types):
        if i <= j:
            continue
        posts1 = set(df_meds[df_meds['type'] == type1]['post_id'])
        posts2 = set(df_meds[df_meds['type'] == type2]['post_id'])
        n11 = len(posts1 & posts2)
        n10 = len(posts1) - n11
        n01 = len(posts2) - n11
        n00 = len(df_meds['post_id'].unique()) - n11 - n10 - n01
        if min(n11, n10, n01, n00) > 5:
            table = [[n11, n10], [n01, n00]]
            odds, pval = scipy.stats.fisher_exact(table)
            odds_meds[i, j] = odds_meds[j, i] = odds
            pvals_meds[i, j] = pvals_meds[j, i] = pval

# --- Symptom-Symptom Co-occurrence ---
df_syms = df[df['ner'] == 'SIGN_SYMPTOM'].dropna(subset=['type'])
sym_types = df_syms['type'].unique()

odds_syms = np.ones((len(sym_types), len(sym_types)))
pvals_syms = np.ones_like(odds_syms)

for i, type1 in tqdm(enumerate(sym_types), total=len(sym_types), desc="Symptom co-occurrence"):
    for j, type2 in enumerate(sym_types):
        if i <= j:
            continue
        posts1 = set(df_syms[df_syms['type'] == type1]['post_id'])
        posts2 = set(df_syms[df_syms['type'] == type2]['post_id'])
        n11 = len(posts1 & posts2)
        n10 = len(posts1) - n11
        n01 = len(posts2) - n11
        n00 = len(df_syms['post_id'].unique()) - n11 - n10 - n01
        if min(n11, n10, n01, n00) > 5:
            table = [[n11, n10], [n01, n00]]
            odds, pval = scipy.stats.fisher_exact(table)
            odds_syms[i, j] = odds_syms[j, i] = odds
            pvals_syms[i, j] = pvals_syms[j, i] = pval

# --- Symptom-Medication Co-occurrence ---
df_both = df[df['ner'].isin(['MEDICATION', 'SIGN_SYMPTOM'])].dropna(subset=['type'])

odds_medsyms = np.ones((len(med_types), len(sym_types)))
pvals_medsyms = np.ones_like(odds_medsyms)

for i, med in tqdm(enumerate(med_types), total=len(med_types), desc="Symptom-Medication co-occurrence"):
    for j, sym in enumerate(sym_types):
        posts1 = set(df_both[df_both['type'] == med]['post_id'])
        posts2 = set(df_both[df_both['type'] == sym]['post_id'])
        n11 = len(posts1 & posts2)
        n10 = len(posts1) - n11
        n01 = len(posts2) - n11
        n00 = len(df_both['post_id'].unique()) - n11 - n10 - n01
        if min(n11, n10, n01, n00) > 5:
            table = [[n11, n10], [n01, n00]]
            odds, pval = scipy.stats.fisher_exact(table)
            odds_medsyms[i, j] = odds
            pvals_medsyms[i, j] = pval

# --- Store Results ---
os.makedirs("data/results/cooccurrences", exist_ok=True)

pd.DataFrame(odds_meds, index=med_types, columns=med_types).to_csv("data/results/cooccurrences/results_odds_medications.csv")
pd.DataFrame(pvals_meds, index=med_types, columns=med_types).to_csv("data/results/cooccurrences/results_p_values_medications.csv")

pd.DataFrame(odds_syms, index=sym_types, columns=sym_types).to_csv("data/results/cooccurrences/results_odds_symptoms.csv")
pd.DataFrame(pvals_syms, index=sym_types, columns=sym_types).to_csv("data/results/cooccurrences/results_p_values_symptoms.csv")

pd.DataFrame(odds_medsyms, index=med_types, columns=sym_types).to_csv("data/results/cooccurrences/results_odds_symptoms_medications.csv")
pd.DataFrame(pvals_medsyms, index=med_types, columns=sym_types).to_csv("data/results/cooccurrences/results_p_values_symptoms_medications.csv")

# --- Heatmap Plotting ---
fig, axes = plt.subplots(1, 3, figsize=(37, 13))

# Order categories
order_symptoms = ["B1", "B2", "B3", "B4", "C1", 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'Other']
order_medications = ["contraceptives", "antidepressants (SSRI)", "other medication", "vitamins/minerals",
                     "other hormonal therapy", "antidepressants (non-SSRI)", "benzodiazepines", "antipsychotics",
                     "antiepileptics", "stimulants", "beta blockers", "azapirones"]

# Load results
odds_syms_df = pd.read_csv("data/results/cooccurrences/results_odds_symptoms.csv", index_col=0).loc[order_symptoms, order_symptoms]
pvals_syms_df = pd.read_csv("data/results/cooccurrences/results_p_values_symptoms.csv", index_col=0).loc[order_symptoms, order_symptoms]

odds_meds_df = pd.read_csv("data/results/cooccurrences/results_odds_medications.csv", index_col=0).loc[order_medications, order_medications]
pvals_meds_df = pd.read_csv("data/results/cooccurrences/results_p_values_medications.csv", index_col=0).loc[order_medications, order_medications]

odds_medsyms_df = pd.read_csv("data/results/cooccurrences/results_odds_symptoms_medications.csv", index_col=0).loc[order_medications, order_symptoms]
pvals_medsyms_df = pd.read_csv("data/results/cooccurrences/results_p_values_symptoms_medications.csv", index_col=0).loc[order_medications, order_symptoms]

# Masking
mask_upper = np.triu(np.ones_like(odds_syms_df, dtype=bool), k=1)
mask_syms = (pvals_syms_df > 0.001 / (len(sym_types) * (len(sym_types) - 1) / 2)) | mask_upper

mask_upper = np.triu(np.ones_like(odds_meds_df, dtype=bool), k=1)
mask_meds = (pvals_meds_df > 0.001 / (len(med_types) * (len(med_types) - 1) / 2)) | mask_upper

mask_medsyms = (pvals_medsyms_df > 0.001 / (len(med_types) * len(sym_types) / 2))

# Color map
cmap = sns.diverging_palette(230, 15, as_cmap=True)
cmap.set_bad("white")

# Plot symptoms
sns.heatmap(odds_syms_df, ax=axes[0], mask=mask_syms, cmap=cmap,
            annot=True, fmt=".2f", cbar=False, vmin=0.25, vmax=4, center=1,
            xticklabels=True, yticklabels=True, square=True, annot_kws={"size": 28, "color": 'black'})
axes[0].set_title("Symptoms", fontsize=33)
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=90)

# Plot medications
sns.heatmap(odds_meds_df, ax=axes[1], mask=mask_meds, cmap=cmap,
            annot=True, fmt=".2f", cbar=False, vmin=0.25, vmax=4, center=1,
            xticklabels=True, yticklabels=True, square=True, annot_kws={"size": 28, "color": 'black'})
axes[1].set_title("Medications", fontsize=33)
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=90)

# Plot symptoms-medications
sns.heatmap(odds_medsyms_df, ax=axes[2], mask=mask_medsyms, cmap=cmap,
            annot=True, fmt=".2f", cbar=False, vmin=0.25, vmax=4, center=1,
            xticklabels=True, yticklabels=True, square=True, annot_kws={"size": 28, "color": 'black'})
axes[2].set_title("Symptoms and Medications", fontsize=33)
plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=90)

for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)

plt.tight_layout()
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/fig5.png", dpi=300, bbox_inches='tight')
