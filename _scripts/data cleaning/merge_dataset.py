#%% IMPORTS
import pandas as pd


#%% LOAD DATASET WITH METADATA
scriptum_df = pd.read_json("https://raw.githubusercontent.com/CCS-ZCU/scriptum/refs/heads/master/data/files_df.json")

#%% ENRICH DATA WITH YEARS FROM FILENAMES
mask = scriptum_df['year'].isna()
for idx in scriptum_df[mask].index:
    filename = scriptum_df.at[idx, 'filename']
    if '88' in filename:
        scriptum_df.at[idx, 'year'] = 1988.0
    elif '89' in filename:
        scriptum_df.at[idx, 'year'] = 1989.0
    elif '87' in filename:
        scriptum_df.at[idx, 'year'] = 1987.0
    elif '51' in filename and scriptum_df.at[idx, 'filename'] == 'novy-brak_05_chybi-str-51_ocr.txt':
        # see https://scriptum.cz/cs/periodika/novy-brak
        scriptum_df.at[idx, 'year'] = 1983.0
    elif '71' in filename:
        # see https://scriptum.cz/cs/periodika/poradni-svitek
       continue
    else:
        continue

# CREATE NEW COLUMN
sum(scriptum_df["year"].apply(lambda x: 1948 <= x <= 1989))
scriptum_df["communism"] = scriptum_df["year"].apply(lambda x: 1948 <= x <= 1989)

#%% Looking at the data
list(scriptum_df) # list all dataset columns
scriptum_df.head(5) # print start of dataset

#%% CREATE SUBSET of years of interest

scriptum_subset = scriptum_df.loc[scriptum_df["year"]>=1968]
scriptum_subset = scriptum_subset.loc[scriptum_subset["year"]<=1989]

# %% CREATE TXT FILE 
# so that we can filter out the the huggingface dataset, which only contains two columns,
# 1) the fulltext of the journals and 2) the filename of the ocr'ed text

filenames = scriptum_subset['filename'].to_list()

with open("filenames_SCRIPTUM_1968-1989.txt", mode='w') as file:
    file.write('\n'.join(filenames))


# %%
