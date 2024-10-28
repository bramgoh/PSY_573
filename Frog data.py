#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 14:44:17 2024

@author: bramgoh
"""

import pandas as pd
import numpy as np
import os
import re
import string
import math
os.chdir("/Users/bramgoh/Documents/U Southern California/PSY 573/Project")

frog_raw = pd.read_csv("scene_descriptions.tsv", sep = "\t", index_col = 0)

#%%
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer

all_ppts = pd.DataFrame(columns = ["qualtrics_id", "scene_no", "sentences"])
for i, row in frog_raw.iterrows():
    df_ppt = pd.DataFrame(columns = ["qualtrics_id", "scene_no", "sentences"])
    for j, col in enumerate(row): 
        trial_sentences = sent_tokenize(col)
        df_trial = pd.DataFrame({"qualtrics_id" : i, "scene_no": j+1, "sentences": trial_sentences})
        df_ppt = pd.concat([df_ppt, df_trial], axis = 0)
    all_ppts = pd.concat([all_ppts, df_ppt], axis=0)

#%%

# DESCRIPTIVE STATS
# Sentence level
no_of_sentences = all_ppts.groupby(["qualtrics_id", "scene_no"]).count()
no_of_sentences = pd.pivot_table(no_of_sentences, values = "sentences", index = "qualtrics_id", columns = "scene_no")
scene_means = no_of_sentences.mean(axis = 0)


ppt_means = no_of_sentences.mean(axis = 1)
import matplotlib.pyplot as plt
plt.hist(ppt_means)
plt.show()


#%%

subtlex_us = pd.read_excel("SUBTLEXusfrequencyabove1.xls")
subtlex_us["Word"] = subtlex_us["Word"].apply(lambda x: str(x).lower())
subtlex_wf = subtlex_us[["Word", "SUBTLWF"]].set_index("Word").T.to_dict("list")

punctuation = list(string.punctuation)


# Token level

all_sent_lengths = []

for sentence in all_ppts["sentences"]:
    trial_tokens = word_tokenize(sentence)
    trial_tokens = [tok.lower() for tok in trial_tokens if tok not in punctuation]
    all_sent_lengths.append(len(trial_tokens))

all_ppts["word_count"] = all_sent_lengths
all_ppts = all_ppts[all_ppts["word_count"] != 0]

#%%

all_tokens = []
all_tagged_tokens = []
all_tags = []
all_bigrams = []
all_trigrams = []
content_word_count = []
median_word_freqs = []
log_min_word_freq = []

content_tags_penn = ["JJ", "JJR", "JJS", "NN", "NNS", "NNP", "NNPS", "RB", "RBR", "RBS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
content_tags_universal = ["NOUN", "VERB", "ADJ", "ADV"]

for sentence in all_ppts["sentences"]: 
    trial_tokens = word_tokenize(sentence)
    trial_tokens = [tok.lower() for tok in trial_tokens if tok not in punctuation]
    all_tokens.append(trial_tokens)
    tagged_tokens = nltk.pos_tag(trial_tokens, tagset = "universal")
    all_tagged_tokens.append(tagged_tokens)
    
    sent_tags = []
    wfs = []
    content_count = 0
    for tup in tagged_tokens:
        word, tag = tup
        sent_tags.append(tag)
        if tag in content_tags_universal:
            content_count += 1
        if word in subtlex_wf.keys():
            wf = subtlex_wf[word]
        else: 
            wf == 0
        wfs.append(wf)
    all_tags.append(sent_tags)
    content_word_count.append(content_count)
    median_word_freqs.append(np.median(np.array(wfs)))
    log_min_word_freq.append(np.log(np.array(wfs).min()))
    
    all_bigrams.append([(trial_tokens[idx] + " " + trial_tokens[idx+1]) for idx in range(len(trial_tokens)-1)])
    all_trigrams.append([(trial_tokens[idx] + " " + trial_tokens[idx+1] + " " + trial_tokens[idx+2]) for idx in range(len(trial_tokens)-2)])
    
all_ppts["tokens"] = all_tokens
all_ppts["tagged_tokens"] = all_tagged_tokens
all_ppts["token_tags"] = all_tags
all_ppts["bigrams"] = all_bigrams
all_ppts["trigrams"] = all_trigrams
all_ppts["bigram_count"] = all_ppts["word_count"] + 1 - 2
all_ppts["trigram_count"] = all_ppts["word_count"] + 1 - 3
all_ppts["combination_ratio"] = all_ppts["trigram_count"]/all_ppts["word_count"]
all_ppts["content_word_count"] = content_word_count
all_ppts["content_word_ratio"] = all_ppts["content_word_count"]/all_ppts["word_count"]
all_ppts["median_SUBTLEX_freq"] = median_word_freqs
all_ppts["log_min_SUBTLEX_freq"] = log_min_word_freq
#%%
bigram_corpus = []
for sentence in all_bigrams:
    for bigram in sentence:
        bigram_corpus.append(bigram)

trigram_corpus = []
for sentence in all_trigrams:
    for trigram in sentence:
        trigram_corpus.append(trigram)


from collections import Counter
bigram_freqs = Counter(bigram_corpus)
trigram_freqs = Counter(trigram_corpus)

#%%

def median_ngram_freq(ngrams, counter):
    freqs = []
    for n in ngrams:
      f = counter[n]
      freqs.append(f)
    median_freq = np.median(np.array(freqs))
    return(median_freq)


def min_ngram_freq(ngrams, counter):
    freqs = []
    for n in ngrams:
      f = counter[n]
      freqs.append(f)
    min_freq = np.min(np.array(freqs))
    return(min_freq)

all_ppts["median_bigram_freq"] = all_ppts["bigrams"].apply(lambda x: median_ngram_freq(x, bigram_freqs))
all_ppts["median_trigram_freq"] = all_ppts["trigrams"].apply(lambda x: median_ngram_freq(x, trigram_freqs))


def find_ngram_minmax(counter):
    keys = list(counter.keys())
    vals = list(counter.values())
    min_ngram = keys[vals.index(min(vals))]
    max_ngram = keys[vals.index(max(vals))]
    print("Least frequent: {}, Most frequent: {}".format(min_ngram, max_ngram))
    
find_ngram_minmax(bigram_freqs)
find_ngram_minmax(trigram_freqs)    

#%%

data_mean = all_ppts[["qualtrics_id", "word_count", "bigram_count", "trigram_count", "combination_ratio", "content_word_count", "content_word_ratio", "median_SUBTLEX_freq", "log_min_SUBTLEX_freq"]].groupby("qualtrics_id").mean()

#%%
just_tokens = all_ppts[["qualtrics_id", "sentences"]]
just_tokens = just_tokens.groupby("qualtrics_id")["sentences"].apply(lambda x: " ".join(x)).reset_index()

no_of_types = []
no_of_tokens = [] 
for sentence in just_tokens["sentences"]: 
    trial_tokens = word_tokenize(sentence)
    trial_tokens = [tok.lower() for tok in trial_tokens if tok not in punctuation]
    no_of_types.append(len(set(trial_tokens)))
    no_of_tokens.append(len(trial_tokens))

just_tokens["type_count"] = no_of_types
just_tokens["token_count"] = no_of_tokens
reduced_data = data_mean.merge(just_tokens, on = "qualtrics_id", how = "left")
reduced_data["ttr"] = reduced_data["type_count"]/reduced_data["token_count"]
#%%
demospinspan = pd.read_csv("demospinspan.csv", index_col= 0)
frog_full = reduced_data.merge(demospinspan, how = "left", on = "qualtrics_id")
frog_reduced = frog_full[["qualtrics_id", "word_count", "bigram_count", "trigram_count", "combination_ratio", "content_word_count", "content_word_ratio", "median_SUBTLEX_freq", "log_min_SUBTLEX_freq", "spin_high", "spin_low", "span", "SES.ycoord", "political_scale", "political_interest", "schooling_level", "age", "ttr"]]
frog_reduced = frog_reduced[frog_reduced["spin_high"].notnull()].set_index("qualtrics_id")

#%%
import seaborn as sns

sns.pairplot(data = frog_reduced[["word_count", "combination_ratio", "content_word_ratio", "median_SUBTLEX_freq", "ttr", "SES.ycoord", "political_scale", "political_interest", "schooling_level", "age", "spin_high", "spin_low", "span"]])
plt.show()

corr_matrix = frog_reduced.corr()
# frog_reduced.to_csv("processed_frog_data.csv")
