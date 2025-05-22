#%% IMPORTS
import pandas as pd


#%% LOAD DATASET WITH METADATA
scriptum_df = pd.read_json("https://raw.githubusercontent.com/CCS-ZCU/scriptum/refs/heads/master/data/files_df.json")

# CREATE NEW COLUMN
sum(scriptum_df["year"].apply(lambda x: 1948 <= x <= 1989))
scriptum_df["communism"] = scriptum_df["year"].apply(lambda x: 1948 <= x <= 1989)

#%% Looking at the data
list(scriptum_df) # list all dataset columns
scriptum_df.head(5) # print start of dataset

#%% CREATE SUBSET of years of interest

scriptum_subset = scriptum_df.loc[scriptum_df["year"]>=1971]
scriptum_subset = scriptum_subset.loc[scriptum_subset["year"]<=1999]

# %% CREATE TXT FILE 
# so that we can filter out the the huggingface dataset, which only contains two columns,
# 1) the fulltext of the journals and 2) the filename of the ocr'ed text

filenames = scriptum_subset['filename'].to_list()

with open("filenames_SCRIPTUM_1971-1999.txt", mode='w') as file:
    file.write('\n'.join(filenames))
# %%
