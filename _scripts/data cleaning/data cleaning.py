# https://huggingface.co/datasets/janko/250521-scriptum
# https://github.com/stopwords-iso/stopwords-cs/blob/master/stopwords-cs.txt

# from datasets import load_dataset
# dataset = load_dataset("text", data_files={"train": ["my_text_1.txt", "my_text_2.txt"], "test": "my_test_file.txt"})

# dataset = load_dataset("text", data_dir="path/to/text/dataset")

#%% load data from huggingface
from datasets import load_dataset
from datasets import Dataset
import pandas as pd

#%%
dataset_text = load_dataset('janko/250521-scriptum')

#%% convert data to pandas
# scriptum_text_df = Dataset.to_pandas('dataset') # too large, returns error

#%%

with open("filenames_SCRIPTUM_1971-1999.txt", mode='r') as file:
    filename_list = file.read().splitlines()

#%%

subset = dataset_text['train'].filter(lambda row: row["file"] in filename_list)
len(subset)

#%% --------------------------------------

dataset_text['train']['file'] # -> list

#%% MERGE THE DATASETS

scriptum_metadata_df = pd.read_json("https://raw.githubusercontent.com/CCS-ZCU/scriptum/refs/heads/master/data/files_df.json")
scriptum_text_df = subset.to_pandas()

scriptum_metadata_df.rename(columns={'filename': 'file'}, inplace=True)

scriptum_df = scriptum_metadata_df.merge(scriptum_text_df, on='file', how='inner')

scriptum_df.head()

# %%
