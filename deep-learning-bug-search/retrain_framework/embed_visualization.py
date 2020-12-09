# -*- coding: utf-8 -*-

print(__doc__)
from time import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import random
from random import randint
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=5, help='select N points to plot')
parser.add_argument('--top_sim', type=bool, default=False, help='choose top similar points')
parser.add_argument('--last_sim', type=bool, default=False, help='choose last similar points')

args = parser.parse_args()
N = args.n
GET_SIM_POINTS = args.top_sim
GET_NOT_SIM_POINTS = args.last_sim

random.seed(4)

# number of application vectors == number of source vectors
df_appl = pd.read_csv("retrain/ourresults/application.csv",header=None)
df_src = pd.read_csv("retrain/ourresults/source.csv",header=None)
size = pd.read_csv("retrain/ourresults/size.csv",sep= " ", header=None)


df = pd.concat([df_appl, df_src], ignore_index=True)

X = df.as_matrix()
n_samples, n_features = X.shape

half_y = list(range(df_appl.shape[0]))
y = np.asarray( half_y +half_y)

# compute cosine similarity
cosine_similarity_id_lst = []
for idx in range(len(df_appl)):
    tmpa = df_appl.loc[idx].as_matrix().reshape(1,-1)
    tmpb = df_src.loc[idx].as_matrix().reshape(1,-1)
    similarity = cosine_similarity(tmpa,tmpb)
    cosine_similarity_id_lst.append((idx,similarity.tolist()[0][0]))


sorted_by_similarity = sorted(cosine_similarity_id_lst, key=lambda tup: tup[1],reverse=True)
topN_similar = sorted_by_similarity[:N]
lastN_similar = sorted_by_similarity[-N:]

topN_ids = []
lastN_ids = []
for i in range(len(topN_similar)):
    topN_ids.append(topN_similar[i][0])
    lastN_ids.append(lastN_similar[i][0])

if GET_SIM_POINTS:
    # take top N most silimar points
    X = pd.concat([df_appl.iloc[topN_ids],df_src.iloc[topN_ids]] , ignore_index=True)
    half_y = list(range(df_appl.iloc[topN_ids].shape[0]))
    size = size.iloc[topN_ids]

elif GET_NOT_SIM_POINTS:
    # take last N most silimar points
    X = df = pd.concat([df_appl.iloc[lastN_ids],df_src.iloc[lastN_ids]] , ignore_index=True)
    half_y = list(range(df_appl.iloc[lastN_ids].shape[0]))
    size = size.iloc[lastN_ids]

else:
    random_index = []
    for i in range(N):
        random.seed(4)# fix random result
        tmp = randint(0,len(df_appl))
        random_index.append(tmp)
    X = df = pd.concat([df_appl.iloc[random_index],df_src.iloc[random_index]] , ignore_index=True)
    half_y = list(range(df_appl.iloc[random_index].shape[0]))
    size = size.iloc[random_index]

y = np.asarray( half_y +half_y)

def plot_our_embedding(X,title=None,s=None):
    #x_min, x_max = np.min(X, 0), np.max(X, 0)
    #X = (X - x_min) / (x_max - x_min)
    plt.figure()
    ax = plt.subplot(111)
    #cmap = plt.cm.jet
    c = ['red', '#f0c2e0', '#e085c2','#800080','#bf8040','red', '#f0c2e0', '#e085c2','#800080','#bf8040','yellow','#999900','#ff9933', 'green','#9fdf9f','yellow','#999900','#ff9933', 'green','#9fdf9f','#99d6ff','#33adff','blue','#8c8c8c','#cccccc','#99d6ff','#33adff','blue','#8c8c8c','#cccccc']*3

    for i in range(X.shape[0]):
        #print i
        #print (X[i, 0], X[i, 1])
        #ax.scatter(X[i, 0], X[i, 1],
        #         color=c[i], s = s[i])

        ax.scatter(X[i, 0], X[i, 1],
            s=s[i])
        ax.annotate(y[i], xy = (X[i, 0], X[i, 1]), xytext=(-0.3*i,0.3*i),
            textcoords='offset points', ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.2))
        #ax.annotate(y[i], xy = (X[i, 0], X[i, 1]), xytext = (-5*y[i], 5*y[i]),
        #            textcoords = 'offset points', ha = 'right', va = 'bottom',
        #            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        #            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    plt.xticks([]), plt.yticks([])
    plt.show()

def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)

    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.show()

def using_random(X,s=None):

    print "using random"
    print("Computing random projection")
    rp = random_projection.SparseRandomProjection(n_components=2, random_state=42)
    X_projected = rp.fit_transform(X)
    #plot_embedding(X_projected, "Random Projection of the results")
    plot_our_embedding(X_projected,
              "Random Projection of the results" ,s)

def using_pca(X,s=None):
    t0 = time()
    X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(X)
    #plot_embedding(X_pca,"Principal Components projection of the results (time %.2fs)" %(time() - t0))
    plot_our_embedding(X_pca,
              "Principal Components projection of the results (time %.2fs)" %
               (time() - t0),s)
def using_lda(X,s=None):
    X = X.as_matrix()
    print("Computing Linear Discriminant Analysis projection")
    X2 = X.copy()
    X2.flat[::X.shape[1] + 1] += 0.01  # Make X invertible
    t0 = time()
    X_lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=2).fit_transform(X2, y)
    #plot_embedding(X_lda,"Linear Discriminant projection of the results (time %.2fs)" %(time() - t0))
    plot_our_embedding(X_lda,
              "Linear Discriminant projection of the results (time %.2fs)" %
               (time() - t0),s)
def using_tsne(X,s):
    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    X_tsne = tsne.fit_transform(X)
    plot_our_embedding(X_tsne,
              "t-SNE embedding of the digits (time %.2fs)" %
               (time() - t0),s)
    #plot_embedding(X_tsne,
    #           "t-SNE embedding of the results (time %.2fs)" %
    #           (time() - t0))

def using_pca_tsne(X,dims_pca=40,s=None):

    print("Computing PCA reduction with %s dims"% str(dims_pca))
    t0 = time()
    X_pca = decomposition.TruncatedSVD(n_components=dims_pca).fit_transform(X)
    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=42)
    X_tsne = tsne.fit_transform(X_pca)
    plot_our_embedding(X_tsne,
              "t-SNE embedding of the digits (time %.2fs)" %
               (time() - t0),s)
    #plot_embedding(X_tsne,
    #           "t-SNE embedding of the results (time %.2fs)" %
    #           (time() - t0))

def using_mds(X,s=None):
    t0 = time()
    print("Computing mds embedding")
    mds = manifold.MDS(n_components=2,random_state=42)
    X_mds = mds.fit_transform(X)
    plot_our_embedding(X_mds,
              "t-SNE embedding of the digits (time %.2fs)" %
               (time() - t0),s)


# s is data size to define plot point size
s = np.concatenate((size[0].as_matrix(), size[1].as_matrix()), axis=0)
#using_random(X,s=s)
#using_pca(X,s=s)
#using_lda(X,s=s)
#using_tsne(X,s=s)
#using_pca_tsne(X,dims_pca=30,s=s)
using_mds(X,s=s)




