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
column_names = [f'Topic_{i}' for i in range(dtm.shape[1])]
df = pd.DataFrame(dtm, columns=column_names)
df['Highest_Topic'] = df.idxmax(axis=1)

merged_df = pd.merge(scriptum_df, df, left_index=True, right_index=True, how='inner')
merged_df.head()

# %%

import pandas as pd 
import matplotlib.pyplot as plt

viz_df = pd.read_csv('/root/SCRIPTUM/_scripts/visualization/small_dataset_for_viz.csv')

haeufigkeiten = viz_df['Highest_Topic'].value_counts()

# Zugriff auf die einzigartigen Datenpunkte (Index)
datapoints = haeufigkeiten.index.tolist()

# Zugriff auf die Häufigkeiten (Werte)
anzahlen = haeufigkeiten.values.tolist()

plt.bar(datapoints, anzahlen, color='blue')

plt.xlabel('Topics')
plt.ylabel('Counts')
plt.title('Counts of the topics')
plt.xticks(rotation=45)  

plt.tight_layout()  
plt.show()
# %%

# Schritt 3: Diagramm anpassen
plt.xlabel('Jahr')
plt.ylabel('Anzahl der Veröffentlichungen')
plt.title('Anzahl der Veröffentlichungen pro Thema über die Jahre')
plt.xticks(df['Jahr'])  # Alle Jahre auf der x-Achse anzeigen
plt.legend()  # Legende hinzufügen
plt.grid()

# Schritt 4: Diagramm anzeigen
plt.tight_layout()  # Optional: Layout anpassen
plt.show()

# %%

topics_over_time = pd.DataFrame()


for thema in viz_df.columns[9:19]:  # Alle Spalten außer der ersten (Jahr)
    topics_over_time[thema] = viz_df[thema]
topics_over_time['year'] = viz_df['year']

topics_over_time.head()
# %%

print(len(topics_over_time))

topics_over_time = topics_over_time.dropna()
print(len(topics_over_time))

# Schritt 2: Mittelwerte pro Jahr berechnen
df_mean = topics_over_time.groupby('year').mean().reset_index()

relevant_topics = {'Topic_1': 'Illegality and Legal Accusation',
                   'Topic_2': '',
                   'Topic_3': '',
                   'Topic'
                   'Topic_8': 'Religion, Resurrection, and Divinity',
                   'Topic_6': 'Democratization and Political Ideologies'}
# Schritt 3: Diagramm erstellen
for topic in df_mean.columns[1:]:  # Alle Spalten außer der ersten (year)
    plt.plot(df_mean['year'], df_mean[topic], marker='o', label=topic)

# Schritt 4: Diagramm anpassen
plt.xlabel('Years')
plt.ylabel('Mean values')
plt.title('Mean Values per topic over the years')
plt.xticks(df_mean['year'].unique(), rotation=45, fontsize=10)  # Alle Jahre auf der x-Achse anzeigen
plt.legend()  # Legende hinzufügen
plt.grid()

# Schritt 5: Diagramm anzeigen
plt.tight_layout()  # Optional: Layout anpassen
plt.show()

# %%

import numpy as np
df_std = topics_over_time.groupby('year').std().reset_index()

# Step 3: Calculate confidence intervals (e.g., 95% CI)
confidence_level = 1.96  # For 95% confidence
ci_upper = df_mean.iloc[:, 1:] + (confidence_level * (df_std.iloc[:, 1:] / np.sqrt(len(topics_over_time))))
ci_lower = df_mean.iloc[:, 1:] - (confidence_level * (df_std.iloc[:, 1:] / np.sqrt(len(topics_over_time))))

relevant_topics = {'Topic_1': 'Illegality and Legal Accusation',
                   'Topic_8': 'Religion, Resurrection, and Divinity',
                   'Topic_6': 'Democratization and Political Ideologies'}
for topic, label in relevant_topics.items():  # Alle Spalten außer der ersten (year)
    plt.plot(df_mean['year'], df_mean[topic], marker='o', label=label)
    plt.fill_between(df_mean['year'], ci_lower[topic], ci_upper[topic], alpha=0.2)

# Schritt 4: Diagramm anpassen
plt.xlabel('Years')
plt.ylabel('Mean values')
plt.title('Mean Values per topic over the years (including CIs)')
plt.xticks(df_mean['year'].unique(), rotation=45, fontsize=10)  # Alle Jahre auf der x-Achse anzeigen
plt.legend()  # Legende hinzufügen
plt.grid()

# Schritt 5: Diagramm anzeigen
plt.tight_layout()  # Optional: Layout anpassen
plt.show()

# %%


