import os
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from embedding4bert import Embedding4BERT
from numpy import matlib
from EOO import EOO
from GWO import GWO
from Glob_Vars import Glob_Vars
from LOA import LOA
from Model_Biclustering import Model_Biclustering
from Model_CNN import Model_CNN
from Model_Fuzzy import Model_Fuzzy
from Model_GPT_3 import Model_GPT_3
from Model_KNN import Model_KNN
from Model_Proposed import Model_Proposed
from Objfun import Objfun
from PROPOSED import *
from Plot_Results import *
from SEO import SEO


# Removing Puctuations
def rem_punct(my_str):
    # define punctuation
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

    # remove punctuation from the string
    no_punct = ""
    for char in my_str:
        if char not in punctuations:
            no_punct = no_punct + char + " "

    # display the unpunctuated string
    return no_punct


# Read Dataset
an = 0
if an == 1:
    Path = './Datasets/patent_claims_fulltext.csv'
    data = pd.read_csv(Path)
    nrows = 1000000
    Data = data['claim_txt'].to_numpy()[:nrows]
    Target = data['ind_flg'].to_numpy()[:nrows]

    np.save('Data.npy', Data)
    np.save('Target.npy', Target)

# Text Pre-processing
an = 0
if an == 1:
    ps = PorterStemmer()
    Data = np.load('Data.npy', allow_pickle=True)
    Prep_Data = []
    Data = Data[:1000]
    for j in range(len(Data)):
        print(j, len(Data))
        text_tokens = word_tokenize(Data[j])  # convert in to tokens
        stem = []
        for w in text_tokens:  # Stemming
            stem_tokens = ps.stem(w)
            stem.append(stem_tokens)
        words = [word for word in stem if
                 not word in stopwords.words()]  # tokens without stop words
        prep = rem_punct(words)  # Special character removal (Punctuation Removal)
        Prep_Data.append(prep)
    np.save('Preprocess.npy', np.asarray(Prep_Data))

# Bidirectional Encoder Representations from Transformers (BERT)
an = 0
if an == 1:
    prep = np.load('Preprocess.npy', allow_pickle=True)  # Load the Preprocessing Data
    BERT = []
    for i in range(prep.shape[0]):
        print(i, prep.shape[0])
        emb4bert = Embedding4BERT("bert-base-cased")  # bert-base-uncased
        tokens, embeddings = emb4bert.extract_word_embeddings(prep[i])
        BERT.append(embeddings[0])
    np.save('Bert.npy', BERT)  # Save the BERT data

#  Model_GPT_3 feature
an = 0
if an == 1:
    prep = np.load('Preprocess.npy', allow_pickle=True)  # Load the Preprocessing Data
    Minlen = []
    Feature = []
    for i in range(prep.shape[0]):
        print(i, prep.shape[0])
        Feat = Model_GPT_3(prep[i])
        Minlen.append(len(Feat))
        Feature.append(Feat)

    Feat_min = np.min(Minlen)
    GPT_Feature = []
    for j in range(len(Feature)):
        print(j, len(Feature))
        Feat = Feature[j][:Feat_min]
        GPT_Feature.append(Feat)
    np.save('GPT_3.npy', GPT_Feature)  # Save the BERT data

# Feature concatenation
an = 0
if an == 1:
    Bert = np.load('Bert.npy', allow_pickle=True)
    GPT_3 = np.load('GPT_3.npy', allow_pickle=True)
    Concatenated_Feature = np.concatenate((Bert, GPT_3), axis=1)
    np.save('Feature.npy', Concatenated_Feature)

# Optimization for Retrieval process
an = 0
if an == 1:
    Feat = np.load('Feature.npy', allow_pickle=True)
    Targets = np.load('Target.npy', allow_pickle=True)
    Targets = np.reshape(Targets, (-1, 1))
    Glob_Vars.Feat = Feat
    Glob_Vars.Target = Targets[:len(Feat)]
    Npop = 10
    Chlen = 3
    xmin = np.matlib.repmat([50, 0, 5], Npop, 1)
    xmax = np.matlib.repmat([100, 1, 15], Npop, 1)
    fname = Objfun
    initsol = np.zeros((Npop, Chlen))
    for p1 in range(initsol.shape[0]):
        for p2 in range(initsol.shape[1]):
            initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
    Max_iter = 50

    print("GWO...")
    [bestfit1, fitness1, bestsol1, time1] = GWO(initsol, fname, xmin, xmax, Max_iter)

    print("EOO...")
    [bestfit2, fitness2, bestsol2, time2] = EOO(initsol, fname, xmin, xmax, Max_iter)

    print("LOA...")
    [bestfit4, fitness4, bestsol4, time3] = LOA(initsol, fname, xmin, xmax, Max_iter)

    print("SEO...")
    [bestfit3, fitness3, bestsol3, time4] = SEO(initsol, fname, xmin, xmax, Max_iter)

    print("PROPOSED...")
    [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)
    Bestsol_Feat = ([bestsol1.ravel(), bestsol2.ravel(), bestsol3.ravel(), bestsol4.ravel(), bestsol5.ravel()])
    Fitness = ([fitness1.ravel(), fitness2.ravel(), fitness3.ravel(), fitness4.ravel(), fitness5.ravel()])

    np.save('BestSol.npy', Bestsol_Feat)
    np.save('Fitness.npy', Fitness)


an = 0
if an == 1:
    BestSol = np.load('BestSol.npy', allow_pickle=True)
    Feat = np.load('Feature.npy', allow_pickle=True)
    Targets = np.load('Target.npy', allow_pickle=True)
    Targets = np.reshape(Targets, [-1, 1])
    Targets = Targets[: len(Feat)]
    Query = Feat[0]
    Q_T = Targets[0]
    Precision = np.zeros((10, 10))
    Recall = np.zeros((10, 10))
    F1Score = np.zeros((10, 10))
    for i in range(len(BestSol)):  # For all Algorithms
        [Precision[i, :], Recall[i, :], F1Score[i, :]] = Model_Proposed(Feat, BestSol[i], Targets)
    [Precision[5, :], Recall[5, :], F1Score[5, :]] = Model_KNN(Feat, Targets)
    [Precision[6, :], Recall[6, :], F1Score[6, :]] = Model_CNN(Feat, Targets)
    [Precision[7, :], Recall[7, :], F1Score[7, :]] = Model_Fuzzy(Feat, Targets)
    [Precision[8, :], Recall[8, :], F1Score[8, :]] = Model_Biclustering(Feat, Targets)
    np.save('Precision.npy', Precision)
    np.save('Recall.npy', Recall)
    np.save('F1Score.npy', F1Score)

plot_Con_results()
plot_results()
plot_stas_alg()
plot_stas_mtd()
Plot_Features()
