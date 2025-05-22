# https://huggingface.co/datasets/janko/250521-scriptum
# https://github.com/stopwords-iso/stopwords-cs/blob/master/stopwords-cs.txt

#%% IMPORTS
from datasets import load_dataset
from datasets import Dataset
import pandas as pd
from langdetect import detect
import re

#%% LOAD DATA FROM HUGGINGFACE
dataset_text = load_dataset('janko/250521-scriptum')

#%% FILTERING THE DATA BY YEAR

# we have created a list of filenames
# of textdata between the years 1971-1999
# we are extracting these of the huggingface dataset

with open("filenames_SCRIPTUM_1971-1999.txt", mode='r') as file:
    filename_list = file.read().splitlines()

#%%

subset = dataset_text['train'].filter(lambda row: row["file"] in filename_list)
len(subset)

# --------------------------------------
# Testing
# dataset_text['train']['file'] # -> list
# full_filename_set = set(dataset_text['train']['file'])
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
# X exclude where number of tokens is too small
# - exclude the journal name
#%% MERGE THE DATASETS

scriptum_metadata_df = pd.read_json("https://raw.githubusercontent.com/CCS-ZCU/scriptum/refs/heads/master/data/files_df.json")
scriptum_text_df = subset.to_pandas()

scriptum_metadata_df.rename(columns={'filename': 'file'}, inplace=True)

scriptum_df = scriptum_metadata_df.merge(scriptum_text_df, on='file', how='inner')

scriptum_df.head()

# %% CLEANING THE TEXT

# removing text with insufficent token length
scriptum_df = scriptum_df.loc[scriptum_df['tokens_N']>0]

# FUNCTION FOR cleaning the text

def clean_text(text):
    if text:
        # CLEANING PAGEENDS AND SUPERFLUOUS BLANK SPACES
        text = re.sub(r'\[pageend\d+\]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[■•>ů♦©®►▲]', '', text)
    return text

def check_language(text_string):
    detect(text_string)
    pass

scriptum_df['cleaned_text'] = scriptum_df['text'].apply(clean_text)

scriptum_df['detected_language'] = scriptum_df['text'].apply(check_language)    

scriptum_df.head()

# %%
