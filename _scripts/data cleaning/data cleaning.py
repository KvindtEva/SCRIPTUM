# https://huggingface.co/datasets/janko/250521-scriptum
# https://github.com/stopwords-iso/stopwords-cs/blob/master/stopwords-cs.txt

#%% IMPORTS
from datasets import load_dataset
from datasets import Dataset
import pandas as pd

#%% LOAD DATA FROM HUGGINGFACE
dataset = load_dataset('janko/250521-scriptum')

#%% FILTERING THE DATA BY YEAR

# we have created a list of filenames
# of textdata between the years 1971-1999
# we are extracting these of the huggingface dataset

with open("filenames_SCRIPTUM_1971-1999.txt", mode='r') as file:
    filename_list = file.read().splitlines()

subset = dataset['train'].filter(lambda row: row["file"] in filename_list)
len(subset)

# --------------------------------------
# Testing
# dataset['train']['file'] # -> list
# full_filename_set = set(dataset['train']['file'])
# subset_filename_set = set(filename_list)
# --------------------------------------

#%% convert data to pandas
scriptum_text_df = subset.to_pandas() # too large, returns error

# ... merge huggingface data with metadata-set ... #

#%%

# extract nonsense data,
# and wrongly detected ocr,
# or files which have an insufficient token length.

#%% [markdown]

# **Todos for actual Data Cleaning**
# - Exlude pages where the ocr is just too bad
#     -> maybe thouugh lang-detect or something
#      -> research lang-detect
# - exclude Sonderzeichen
# - exclude [pagebreak\d\d]
# - exclude where number of tokens is too small
# - exclude the journal name
