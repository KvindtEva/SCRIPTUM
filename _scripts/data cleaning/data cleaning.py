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

with open("filenames_SCRIPTUM_1968-1989.txt", mode='r') as file:
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

#FUNCTION FOR CLEANING 
def clean_text(text):
    if text:
        text = re.sub(r'\[pageend\d+\]', '', text)
        text = re.sub(r'[■•>ů♦©®►▲°\[\]\\\(\)\"\'<>+-=_\^„]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
    return text


#%%

scriptum_df['cleaned_text'] = scriptum_df['text'].apply(clean_text)

scriptum_df = scriptum_df.loc[scriptum_df["tokens_N"]>0]
scriptum_df = scriptum_df.loc[scriptum_df['text'].str.len() > 100]
#%%
scriptum_df_clean = scriptum_df.loc[~scriptum_df.file.str.contains('obsah_ocr')]

#%%

def remove_periodicals(text, periodicals):
    for string in periodicals:
        text = text.replace(string, '')
    return text.strip()

periodical_titles = set(scriptum_df_clean['periodical_title'].to_list())

scriptum_df_clean['cleaned_text'] = scriptum_df_clean['cleaned_text'].apply(lambda x: remove_periodicals(x, periodical_titles))

print(periodical_titles)
scriptum_df_clean.head()

# %% INCLUDE manual annotations

checkup = pd.read_csv(r'data_screening_reviewed.csv', index_col='INDEX')
checkup = checkup['ANNOTATION']

merged_df = scriptum_df_clean.join(checkup)

export_df = merged_df.loc[merged_df['ANNOTATION']!=0]

# %%

export_df.to_csv('scriptum_cleaned_data.csv', index=False)
# %%
