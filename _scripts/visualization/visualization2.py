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
column_names = [f'Topic_{i}' for i in range(0, dtm.shape[1])]
df = pd.DataFrame(dtm, columns=column_names)
df['Highest_Topic'] = df.idxmax(axis=1)

merged_df = pd.merge(scriptum_df, df, left_index=True, right_index=True, how='inner')
merged_df.head()

# %% [markdown]

# Questions for visualization
# - how big the different topics are?
# - take top topic per document and make a bar plot; document topic matrix
# - topics over time; document topic matrix with metadata

# %%

topics_exil = merged_df.loc[merged_df['periodical_category']=="exil", 'Highest_Topic'].value_counts()
topics_exil['periodical_category'] = 'exil'
topics_saizdat = merged_df.loc[merged_df['periodical_category']=="samizdat", 'Highest_Topic'].value_counts()
topics_saizdat['periodical_category'] = 'saizdat'



# %%



#%% EXPORT

# %%
merged_df.to_csv("small_dataset_for_viz.csv", columns=['periodical_category',
 'periodical_href',
 'periodical_title',
 'file_url',
 'file',
 'year',
 'pages_N',
 'tokens_N',
 'Topic_0',
 'Topic_1',
 'Topic_2',
 'Topic_3',
 'Topic_4',
 'Topic_5',
 'Topic_6',
 'Topic_7',
 'Topic_8',
 'Topic_9',
 'Highest_Topic'])
# %%
