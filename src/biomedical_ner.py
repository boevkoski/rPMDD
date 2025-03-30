import pandas as pd
import pickle as pkl
from tqdm import tqdm
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import os

# Make sure NLTK resources are available
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# ---------- Helper Functions ----------

lemmatizer = WordNetLemmatizer()

def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatize_phrase(phrase):
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(phrase))
    return ' '.join([lemmatizer.lemmatize(word, pos=pos_tagger(tag)) for word, tag in pos_tagged])

# ---------- Load & Filter PMDD Posts ----------

df = pd.read_csv('data/raw/posts_by_pmdd.csv').drop_duplicates(subset=['text'])
df['text'] = df['text'].astype(str)
df = df[df['text'].str.len() > 10]
texts = df['text'].tolist()

print(f"Total unique posts to process: {len(texts)}")

# ---------- Load Biomedical NER Pipeline ----------

print("Loading NER model...")
tokenizer = AutoTokenizer.from_pretrained("d4data/biomedical-ner-all", model_max_length=512)
model = AutoModelForTokenClassification.from_pretrained("d4data/biomedical-ner-all")
pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple", device=0)  # GPU = 0

# ---------- Apply NER to Posts ----------

ner_results = {}
for text in tqdm(texts, desc="Applying biomedical NER"):
    entities = pipe(text)
    ner_results[text] = [(e['word'], e['entity_group'], e['score']) for e in entities]

# Save raw NER output (optional, useful for backup or debugging)
with open('data/results/posts_by_pmdd_ner.pkl', 'wb') as f:
    pkl.dump(ner_results, f)

# ---------- Enrich and Structure Data ----------

ner_data = {
    'post_id': [], 'user': [], 'post_date': [], 'phrase': [],
    'lemma': [], 'ner': [], 'prob': [], 'subreddit': [],
    'condition': [], 'subcategory': [], 'category': [],
}

print("Structuring NER data...")
for idx, row in tqdm(df.iterrows(), total=len(df)):
    entities = ner_results.get(row['text'], None)
    if not entities:
        continue
    for phrase, ner, prob in entities:
        ner_data['post_id'].append(idx)
        ner_data['user'].append(row['user'])
        ner_data['post_date'].append(row['date'])
        ner_data['phrase'].append(phrase)
        ner_data['lemma'].append(lemmatize_phrase(phrase))
        ner_data['ner'].append(ner)
        ner_data['prob'].append(prob)
        ner_data['subreddit'].append(row['subreddit'])
        ner_data['condition'].append(row.get('condition', ''))
        ner_data['subcategory'].append(row['subcategory'])
        ner_data['category'].append(row['category'])

df_ner = pd.DataFrame(ner_data)
df_ner.to_csv("data/posts_pmdd_enriched.csv", index=False)

print("Biomedical NER complete. Output saved to: data/posts_pmdd_enriched.csvv")
