#%% IMPORTS
import numpy as np
import pandas as pd
from turftopic import load_model
from datasets import load_dataset

#%% LOAD DATA

# topic model
model = load_model('janko/s3_scriptum')
model.print_topics()

# cleaned data set that was used for topic modelling
scriptum_dataset = load_dataset('janko/scriptum')
scriptum_df = scriptum_dataset['train'].to_pandas()
scriptum_df.head()

#%%
dtm = model.transform(scriptum_df['cleaned_text'], embeddings=model.embeddings)

# %% merge
column_names = [f'Topic_{i+1}' for i in range(dtm.shape[1])]
df = pd.DataFrame(dtm, columns=column_names)
df['Highest_Topic'] = df.idxmax(axis=1)

merged_df = pd.merge(scriptum_df, df, left_index=True, right_index=True, how='inner')
merged_df.head()

# %%



# %%
