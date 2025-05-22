# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CCS-ZCU/EuPaC_shared/blob/master/SCRIPTUM_getting-started.ipynb)
# 
# This Jupyter notebook has been prepared for the EuPaC Hackathon and provides an easy way to start working with the SCRIPTUM dataset — no need to clone the entire repository or download additional data. It is fully compatible with cloud platforms like Google Colaboratory (click the badge above) and runs without requiring any specialized library installations.
# 
# As such, it is intended as a starting point for EuPaC participants, including those with minimal coding experience.

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests
import nltk
import os
from wordcloud import WordCloud
import json
import io

#%% [markdown]
# This jupyter notebook consists of three sections. In the first part, you are guided through loading a metadata spreadsheet file containing all crucial metadata about the textual files covered by the SCRIPTUM project.
# 
# In the second part, you get a basic demonstration on how to load raw texts of specific documents from SCRIPTUM, based on the metadata. For instance, you can load all texts from a certain journal or a certain year.
# 
# In the final part, the same process is repeated, but this time with textual data automatically preprocessed and filtered - each document is represented as a list of lines, with each line representing one (automatically tokenized) sentence and each sentence contains only filtered lemmata of nouns, verbs, adjectives and proper names. Based on a fully automatized pipeline, these proprocessed are far from perfect and perhaps not available for all documents within the corpus.

# %% 

# load the actual dataset
scriptum_df = pd.read_json("https://raw.githubusercontent.com/CCS-ZCU/scriptum/refs/heads/master/data/files_df.json")

sum(scriptum_df["year"].apply(lambda x: 1948 <= x <= 1989))
# create new column
scriptum_df["communism"] = scriptum_df["year"].apply(lambda x: 1948 <= x <= 1989)

list(scriptum_df) # list all dataset columns
scriptum_df.head(5) # print start of dataset

scriptum_df.periodical_title.value_counts()

# %% [markdown]
# ### Temporal overview

# %%
decade_bins = np.arange(1940, 2000, 10)
decades_labels = ["40th", "50th", "60th", "70th", "80th"]
decade_bins

# %%
fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(6,6), dpi=300, tight_layout=True)
decade_bins = np.arange(1940, 2000, 10)
periodical_decade_counts = scriptum_df.groupby(['periodical_category', pd.cut(scriptum_df['year'], bins=decade_bins)], observed=True).size()
periodical_decade_counts.unstack(level=0).plot(kind='bar', stacked=True, color=["green", "red"], ax=axs[0])
axs[0].set_title("Number of files")
axs[0].set_xticklabels([])
axs[0].set_xlabel("")

periodical_decade_counts = scriptum_df.groupby(['periodical_category', pd.cut(scriptum_df['year'], bins=decade_bins)], observed=True)["pages_N"].sum()
periodical_decade_counts.unstack(level=0).plot(kind='bar', stacked=True, color=["green", "red"], ax=axs[1])
axs[1].set_title("Number of pages")
axs[1].set_xticklabels([])
axs[1].set_xlabel("")


periodical_decade_counts = scriptum_df.groupby(['periodical_category', pd.cut(scriptum_df['year'], bins=decade_bins)], observed=True)["tokens_N"].sum()
periodical_decade_counts.unstack(level=0).plot(kind='bar', stacked=True, color=["green", "red"], ax=axs[2])
axs[2].set_title("Number of tokens")
axs[2].set_xticklabels(decades_labels)
axs[2].set_xlabel("decade")
axs[2].set_yticks(range(0,100000000, 20000000))
axs[2].set_yticklabels([str(n)+ "M" for n in range(0,100, 20)])


# %% [markdown]
# ### Most prominent periodicals

# %%
scriptum_df = scriptum_df[scriptum_df["periodical_title"]!="Obsah"]
len(scriptum_df)

# %%
scriptum_df = scriptum_df[scriptum_df["communism"]]
len(scriptum_df)

# %%
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(12,6), dpi=300, tight_layout=True)
scriptum_df[scriptum_df["periodical_category"]=="exil"].groupby("periodical_title").size().sort_values(ascending=True).tail(5).plot(kind='barh', color="lightgreen", ax=axs[0, 0])
axs[0, 0].set_title("Number of files")
axs[0, 0].set_ylabel("")

scriptum_df[scriptum_df["periodical_category"]=="exil"].groupby("periodical_title")["pages_N"].sum().sort_values(ascending=True).tail(5).plot(kind='barh', color="green", ax=axs[1,0])
axs[1, 0].set_title("Number of pages")
axs[1, 0].set_ylabel("")

scriptum_df[scriptum_df["periodical_category"]=="exil"].groupby("periodical_title")["tokens_N"].sum().sort_values(ascending=True).tail(5).plot(kind='barh', color="darkgreen", ax=axs[2,0])
axs[2, 0].set_title("Number of tokens")
axs[2, 0].set_ylabel("")
axs[2, 0].set_xticks(range(0,20000000, 5000000))
axs[2, 0].set_xticklabels([str(n)+ "M" for n in range(0,20, 5)])

scriptum_df[scriptum_df["periodical_category"]=="samizdat"].groupby("periodical_title").size().sort_values(ascending=True).tail(5).plot(kind='barh', color="pink", ax=axs[0, 1])
axs[0, 1].set_title("Number of files")
axs[0, 1].set_ylabel("")

scriptum_df[scriptum_df["periodical_category"]=="samizdat"].groupby("periodical_title")["pages_N"].sum().sort_values(ascending=True).tail(5).plot(kind='barh', color="red", ax=axs[1, 1])
axs[1, 1].set_title("Number of pages")
axs[1, 1].set_ylabel("")

scriptum_df[scriptum_df["periodical_category"]=="samizdat"].groupby("periodical_title")["tokens_N"].sum().sort_values(ascending=True).tail(5).plot(kind='barh', color="darkred", ax=axs[2, 1])
axs[2, 1].set_title("Number of tokens")
axs[2, 1].set_ylabel("")
axs[2, 1].set_xticks(range(0,4000000, 1000000))
axs[2, 1].set_xticklabels([str(n)+ "M" for n in range(0,4)])

# %% [markdown]
# # Access the raw texts
# 
# This section shows you how to load content of either individual text or a subset of texts based on some metadata.

# %%
# if you in mind a specific document you want to explore
dir = "data/rawtexts"
filename = "150000-slov_1983_06_ocr.txt"
url = "https://raw.githubusercontent.com/CCS-ZCU/scriptum/5bdc0262f061eb98029de0c1b77e5969c5407073/{0}/{1}".format(dir,filename)
resp = requests.get(url)
doc_text = resp.text
doc_text[:1000]

# %%
# you can also load textual data based on specific metadata criteria
metadata_subset = scriptum_df[(scriptum_df["year"].between(1970,1975)) & (scriptum_df["periodical_category"]=="samizdat")]

# %%
subset_rawtexts = []
dir = "data/rawtexts"
for filename in metadata_subset["filename"]:
    url = "https://raw.githubusercontent.com/CCS-ZCU/scriptum/5bdc0262f061eb98029de0c1b77e5969c5407073/{0}/{1}".format(dir,filename)
    try:
        doc_text = requests.get(url).text
    except:
        doc_text = ""
    subset_rawtexts.append(doc_text)

# %%
# now you have a list of raw texts from your selected documents within a list
# these texts can be easily mapped back on your subset of filtered metadata, as they keep the same shape:
# but some of the texts might not be available..
# for instance, here is the beginning of your third document:
subset_rawtexts[2][:2000]

# %% [markdown]
# # Access lemmatized texts
# 
# The sents_data is a list of sentences from the given document accompanied by morphological annotations and lemmata for individual tokens.
# 
# For each sentence, you see the following elements:
# * (1) name of the source text fild
# * (2) raw text of the sentence
# * (2) token data for the sentence
# The token data for each token contain:
#    * (a) The token as it is in the sentence
#    * (b) The automatically assigned lemma corresponding to the token
#    * (c) Its Part-of-Speech
#    * (d) Its starting positional index within the sentence
#    * (e) Its ending positional index within the sentence
# 

# %%
# load an example sentence data file:
filename = "americke-listy_1977_11-11_45_ocr.json"
base_url = "https://ccs-lab.zcu.cz/scriptum_sents_data/"
f_sents_data = json.load(io.BytesIO(requests.get(base_url + filename).content))

# %%
# look at a slice of the data:
f_sents_data[10:15]

# %%
#
scriptum_df["filename"].tolist()[:100]

# %%
subset_lemmatized_sentences = []
for id in ids: # for each work ID from our subset of IDs
    f_sents_data = json.load(io.BytesIO(requests.get(base_url.format(str(id))).content))
    sents_n = len(f_sents_data)
    for sent_data in f_sents_data:
        sent_lemmata = [t[1] for t in sent_data[3] if t[2] in ["NOUN", "VERB", "ADJ", "PROPN"]] # filter for specific POS-tags
        sent_lemmata = [re.sub(r"\W*|\d*", "", t) for t in sent_lemmata] # remove all non-alphanumeric characters
        sent_lemmata = [l for l in sent_lemmata if len(l) > 1] # remove all one-letter words
        sent_lemmata = [l.lower() for l in sent_lemmata] # lowercase all words
        subset_lemmatized_sentences.append(sent_lemmata) # add the lemmatized words from the current sentence to the overall list of lemmatized words

# %%
len(file_list)

# %%
def load_lemmatized_sentences(filename, base_url="https://raw.githubusercontent.com/CCS-ZCU/scriptum/5bdc0262f061eb98029de0c1b77e5969c5407073/{0}/{1}", dir="data/lemsents"):
    url = base_url.format(dir,filename)
    resp = requests.get(url)
    if resp.ok:
        doc_text = requests.get(url).text
    else:
        doc_text = ""
    lemmatized_sents = [[lemma for lemma in sent.split()] for sent in doc_text.split("\n")]
    return lemmatized_sents

# %%
# again, you can load sentence data based on specific metadata criteria
metadata_subset = scriptum_df[(scriptum_df["year"].between(1970,1985)) & (scriptum_df["periodical_category"]=="samizdat")]

# %%
sents_data = []
base_url = "https://ccs-lab.zcu.cz/scriptum_sents_data/"
for filename in metadata_subset["filename"]:
    # the lemmatized data are currently available for approx. 80% of texts
    try:
        f_sents_data = json.load(io.BytesIO(requests.get(base_url + filename.replace(".txt", ".json")).content))
        sents_data.extend(f_sents_data)
    except:
        pass

# %%
len(sents_data)

# %%
lemmatized_sentences = []
for sent_data in sents_data:
    lemmatized_sent = [token[1] for token in sent_data[2] if token[2] in ["NOUN", "VERB", "ADJ", "PROPN"]]
    lemmatized_sentences.append(lemmatized_sent)

# %%
lemmatized_sentences[10:15]

# %%
# to get a flat list of lemmata from all documents in a subset:
lemmata_list = [lemma for sent in lemmatized_sentences for lemma in sent]
# calculate the frequency of each lemma with nltk:
lemmata_freqs = nltk.FreqDist(lemmata_list).most_common()
lemmata_freqs[:10]

# %%
# you can then proceed to a comparison across periods or other subsets of texts
periods_freqs = {}
periods = [(1960,1964), (1965,1969), (1970,1974)]
periods_labels = ["Scriptum {0}-{1}".format(str(period[0]), str(period[1])) for period in periods]
for period, period_label in zip(periods, periods_labels):
    subset_df = scriptum_df[scriptum_df["year"].between(period[0], period[1])]
    sents_data = []
    base_url = "https://ccs-lab.zcu.cz/scriptum_sents_data/"
    for filename in metadata_subset["filename"]:
        # the lemmatized data are currently available for approx. 80% of texts
        try:
            f_sents_data = json.load(io.BytesIO(requests.get(base_url + filename.replace(".txt", ".json")).content))
            sents_data.extend(f_sents_data)
        except:
            pass
    lemmatized_sentences = []
    for sent_data in sents_data:
        lemmatized_sent = [token[1] for token in sent_data[2] if token[2] in ["NOUN", "VERB", "ADJ", "PROPN"]]
        lemmatized_sentences.append(lemmatized_sent)
    lemmata_list = [lemma for sent in lemmatized_sentences for lemma in sent]
    lemmata_list = [lemma for lemma in lemmata_list if len(lemma) > 1]
    lemmata_freqs = nltk.FreqDist(lemmata_list).most_common()
    periods_freqs[period_label] = lemmata_freqs

# %%
# check frequencies for a specific period:
periods_freqs["Scriptum 1960-1964"][:10]

# %%
# plot the freuquencies for a specific period:
wc = WordCloud(width=800, height=400).generate_from_frequencies(dict(periods_freqs[periods_labels[0]][:50]))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()

# %%
n = 100
fig, axs = plt.subplots(3,1, figsize=(4.5, 5) , dpi=300, tight_layout=True)
for item, ax in zip(periods_freqs.items(), axs.ravel()):
    wc = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(dict(item[1][:n]))
    ax.imshow(wc, interpolation='bilinear')
    ax.set_title(item[0])
    ax.axis("off")

# %%



