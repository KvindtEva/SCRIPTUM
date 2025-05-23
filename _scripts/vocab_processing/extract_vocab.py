"""
Slighlty opinionated vocab extraction from our dataset.
Used to generate a list for further vocab pruning.

Steps
=====
a) czech stopwords removal, except salient exceptions
b) tokens has to appear at least 10 times
c) lowercasing
"""

import pandas as pd
import stopwordsiso
from sklearn.feature_extraction.text import CountVectorizer

# parameters
PATH_TEXTS = "../../data/data_cleaning_result_250522.csv"

df = pd.read_csv(PATH_TEXTS)
documents = df["cleaned_text"]

czech_stopwords = stopwordsiso.stopwords("cs")
exceptions = set(["strana", "clanek", "tisíc", "zprávy", "prosím"])
czech_stopwords = czech_stopwords - exceptions
czech_stopwords = list(czech_stopwords)

vectorizer = CountVectorizer(
    min_df=100,
    max_df=0.99,
    stop_words=czech_stopwords,
    lowercase=True,
    )

vocab = vectorizer.fit(documents).get_feature_names_out()
print(f"Number of tokens: {len(vocab)}")
vocab_df = pd.DataFrame({"term": vocab})
vocab_df.to_csv("../../data/vocab.csv", index=False)
