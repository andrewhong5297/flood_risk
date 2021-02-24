# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 10:38:20 2020

@author: Andrew
"""
import pandas as pd
import numpy as np
import datetime
import re
from tqdm import tqdm

corpus = pd.read_csv(r'C:\Users\Andre\OneDrive - nyu.edu\Documents\Python Script Backup\flood files\bills_flood_pruned.csv',index_col=0)

print('cleaning up bills')
corpus = corpus[corpus['Bill_Text'].notna()]
corpus["Summary"]=corpus["Bill_Text"] #setting for rest of nlp
corpus.drop(columns="Bill_Text",inplace=True)

#no duplicate bills
corpus["Agg_Name"] = corpus["State"] + corpus["Legislature"] + corpus["Bill_Number"].astype(str)
corpus = corpus.drop_duplicates(subset=['Agg_Name'])
corpus["Date"] = pd.to_datetime(corpus["Date"])
corpus["Year"] = corpus["Date"].apply(lambda x: x.year)
'''date and count chart'''
import seaborn as sns
import matplotlib.pyplot as plt

bill_state_counts = corpus["State"].value_counts()
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1,figsize=(8,15),gridspec_kw=dict(height_ratios=[1,3]))
                               
ax2 = sns.barplot(x=bill_state_counts, y=bill_state_counts.index)
# ax1.title("Count of Flood Mitigation Bill by State 2000-2020")

date_pivot = corpus[corpus["Year"]>2000].pivot_table(index="Date",values="Bill_Number",aggfunc="count")
date_pivot.cumsum().plot(kind="line",ax=ax1)
'''NLP analysis'''
# from nltk.corpus import stopwords
# from nltk.stem.wordnet import WordNetLemmatizer
# from nltk.tokenize import word_tokenize, sent_tokenize
# from sklearn.feature_extraction.text import CountVectorizer #acts like a model pretty much

# stop_words = stopwords.words("english")
# lemmatizer = WordNetLemmatizer()

# def tokenize(text):
#     # normalize case and remove punctuation
#     text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
#     # tokenize text
#     tokens = word_tokenize(text)
    
#     # lemmatize andremove stop words
#     tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

#     return tokens
# # initialize count vectorizer object
# vect = CountVectorizer(tokenizer=tokenize)
# X = vect.fit_transform(corpus["Summary"]) # [] needs to be wrapped around the string to make it a "document" if you select single element
# vect.vocabulary_

# from sklearn.feature_extraction.text import TfidfTransformer

# # initialize tf-idf transformer object
# transformer = TfidfTransformer(smooth_idf=False)
# tfidf_X = transformer.fit_transform(X)

'''gensim cosine'''
# import time
# import gensim
# import gensim.downloader as api
# from gensim.utils import simple_preprocess
# # https://stackabuse.com/python-for-nlp-working-with-the-gensim-library-part-1/ good primer on tokenizing with gensim

# fasttext_model300 = api.load('fasttext-wiki-news-subwords-300')

# from gensim.test.utils import common_texts
# from gensim.corpora import Dictionary
# from gensim.models import Word2Vec, WordEmbeddingSimilarityIndex
# from gensim.similarities import SoftCosineSimilarity, SparseTermSimilarityMatrix
# #termsim

# # model = Word2Vec(common_texts, size=20, min_count=1)  # train word-vectors on your own text
# termsim_index = WordEmbeddingSimilarityIndex(fasttext_model300.wv) #use FastText.wv here if you want instead of model.uw

# dictionary = Dictionary([simple_preprocess(doc) for doc in corpus["Summary"]]) 
# bow_corpus = [dictionary.doc2bow(simple_preprocess(document)) for document in corpus["Summary"]]

# # similarity_matrix = fasttext_model300.similarity_matrix(dictionary, tfidf=None, threshold=0.0, exponent=2.0, nonzero_limit=100)
# similarity_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary)  # construct similarity  with L2 norm on inner product
# docsim_index = SoftCosineSimilarity(bow_corpus, similarity_matrix)#, num_best=10)

# sims = docsim_index[bow_corpus]
# cosine = pd.DataFrame(data=sims)
# cosine = cosine.applymap(lambda x: round(x,4))
# cosine.to_pickle(r'C:\Users\Andrew\Documents\PythonScripts\NLP\webscraping\flood_research\cosine_bill.pkl')
'''plots for nltk and gensim'''
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, KernelPCA
# clust_df = pd.DataFrame(data=tfidf_X.toarray()) 
# clust_df.columns = vect.vocabulary_

# principalComponents = pca.fit_transform(clust_df) #replace with cosine or clust_df

pca = PCA(n_components=20)
cosine = pd.read_pickle(r'C:\Users\Andre\OneDrive - nyu.edu\Documents\Python Script Backup\flood files\cosine_bill.pkl')
principalComponents = pca.fit_transform(cosine) #replace with cosine or clust_df

PCA_components = pd.DataFrame(principalComponents)
PCA_components["State"] = corpus["State"]
PCA_components["Legislature"] = corpus["Legislature"]
PCA_components["Status"] = corpus["Status"]
PCA_components["Summary"] = corpus["Bill_Summary"]
PCA_components["Agg_Name"] = corpus["Agg_Name"]
PCA_components["Size"] = 1

import seaborn as sns
import matplotlib.pyplot as plt

fig,ax = plt.subplots(figsize=(10,10))

ax = sns.jointplot(PCA_components[0], PCA_components[1], kind="hex", color="#4CB391")
ax.fig.suptitle("First Two Components of t-SNE on Cosine Similarity Matrix of Bill Text (736 Bills, 47 States)")
ax.fig.tight_layout()
ax.fig.subplots_adjust(top=0.95) # Reduce plot to make room 

import plotly.express as px
from plotly.offline import plot
PCA_components = PCA_components.fillna("filler")

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=300)
tsne_results = tsne.fit_transform(PCA_components.iloc[:,:2])

PCA_components["tsne-2d-one"]=tsne_results[:,0]
PCA_components["tsne-2d-two"]=tsne_results[:,1]

fig,ax = plt.subplots(figsize=(10,10))

ax = sns.jointplot(PCA_components["tsne-2d-one"], PCA_components["tsne-2d-two"], kind="hex", color="#4CB391")
ax.fig.suptitle("First Two Components of t-SNE on Cosine Similarity Matrix of Bill Text (736 Bills, 47 States)")
ax.fig.tight_layout()
ax.fig.subplots_adjust(top=0.95) # Reduce plot to make room 

#join regions here instead of states
PCA_components = PCA_components[PCA_components['State']!="filler"]
regions = pd.read_excel(r'C:\Users\Andre\OneDrive - nyu.edu\Documents\Python Script Backup\flood files\States_Abrev_Regions.xlsx', engine='openpyxl')
state_to_region = dict(zip(regions["State Code"],regions["Region"]))
PCA_components["Regions"]=PCA_components["State"].apply(lambda x: state_to_region[x])

#filter by status
PCA_components["Bill_Status"] = PCA_components["Status"].apply(lambda x: x.split('-')[0])

# fig2 = px.scatter(PCA_components, x=0, y=1, hover_data=['Legislature','Bill_Status',"Summary"],color="State",opacity=0.7)
# plot(fig2,filename='2dPCA_cosine_bill.html')

fig_px = px.scatter(PCA_components, x="tsne-2d-one", y="tsne-2d-two", hover_data=['Legislature','Regions',"Agg_Name"],color="Bill_Status",opacity=0.7)
# plot(fig_px,filename='2dtSNE_cosine_bill.html')

# fig3 = px.scatter_3d(PCA_components, x=0, y=1,z=2, size= 'Size', size_max=8,hover_data=['Legislature','Status'],color="State",opacity=0.7)
# plot(fig3,filename='3dPCA_cosine.html')