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
dataset = load_dataset('janko/250521-scriptum')

#%% convert data to pandas
# scriptum_text_df = Dataset.to_pandas('dataset') # too large, returns error

#%%

with open("filenames_SCRIPTUM_1971-1999.txt", mode='r') as file:
    filename_list = file.read().splitlines()

#%%

subset = dataset['train'].filter(lambda row: row["file"] in filename_list)
len(subset)

#%% --------------------------------------

dataset['train']['file'] # -> list

