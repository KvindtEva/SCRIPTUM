"""
Fit an S3 model
"""
# %%
from datetime import datetime
import pandas as pd
import numpy as np
from turftopic import SemanticSignalSeparation
from sentence_transformers import SentenceTransformer

# parameters
MODEL = "BAAI/bge-m3"
PATH_TEXTS = "../../data/data_cleaning_result_250522.csv"
PATH_EMBED = "../../data/embeddings_bge_250523.npy"

# %%
# data & embs
df = pd.read_csv(PATH_TEXTS)
emb = np.load(PATH_EMBED)
documents = df["cleaned_text"]

assert len(df) == len(emb)


# %%
# encoder
encoder = SentenceTransformer(MODEL, model_kwargs={"torch_dtype": "float16"})

# %%
# vectorizer

from sklearn.feature_extraction.text import CountVectorizer
import stopwordsiso

czech_stopwords = stopwordsiso.stopwords("cs")
exceptions = set(["strana", "clanek", "tisíc", "zprávy", "prosím"])
czech_stopwords = czech_stopwords - exceptions
czech_stopwords = list(czech_stopwords)

vectorizer = CountVectorizer(
    min_df=100,
    stop_words=czech_stopwords,
    lowercase=True,
    )


# %%
# static
s3 = SemanticSignalSeparation(
    n_components=20,
    encoder=encoder,
    vectorizer=vectorizer,
    feature_importance="combined"
)

s3.fit(
    raw_documents=documents,
    embeddings=emb
    )

s3.print_topics()

# %%
# dynamic
ts = [datetime(year=int(year), month=1, day=1) for year in df["year"].tolist()]

s3_dynamic = SemanticSignalSeparation(
    n_components=10,
    encoder=encoder,
    vectorizer=vectorizer,
    feature_importance="combined"
)

s3_dynamic.fit_dynamic(
    documents,
    timestamps=ts,
    bins=20
)

s3_dynamic.plot_topics_over_time()
# %% 
 
# Exporting as CSV
with open("keywordProbs.csv", "w") as f:
    f.write(s3.export_topics())

#%%
theta = s3.transform(documents, embeddings=emb)
np.save("../../data/DTM_s3_v1.npy", theta)

# %%
