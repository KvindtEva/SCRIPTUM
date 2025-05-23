"""
Get document embeddings
"""

import numpy as np
import pandas as pd
import torch
from torch.nn import DataParallel
from sentence_transformers import SentenceTransformer

# paths
MODEL = "BAAI/bge-m3"
INPUT_PATH = "../../data/data_cleaning_result_250522.csv"
OUTPUT_PATH_SAMPLE = "../../data/embeddings_bge_250523_first1k.npy"
OUTPUT_PATH_PART = "../../data/embeddings_bge_250523_last5k.npy"
OUTPUT_PATH_WHOLE = "../../data/embeddings_bge_250523.npy"


# data
df = pd.read_csv(INPUT_PATH)
sentences = df["cleaned_text"].tolist()
del df


# model on multiple gpus
model = SentenceTransformer(MODEL, model_kwargs={"torch_dtype": "float16"})
torch.cuda.empty_cache()

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = DataParallel(model)

model = model.to("cuda")
encoder = model.module if hasattr(model, 'module') else model


# run the first 1000 documents (sample)
embeddings_a = encoder.encode(
    sentences[0:1000],
    batch_size=4,
    show_progress_bar=True,
    )

np.save(OUTPUT_PATH_SAMPLE, embeddings_a)

# the the remaining documents
embeddings_b = encoder.encode(
    sentences[1000::],
    batch_size=4,
    show_progress_bar=True,
    )

np.save(OUTPUT_PATH_PART, embeddings_b)

# merge
embeddings_all = np.concatenate((embeddings_a, embeddings_b))
np.save(OUTPUT_PATH_WHOLE, embeddings_all)
