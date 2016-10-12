#!/usr/bin/python
from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
import numpy as np
import urllib

CLUSTERS=2

cols=39
begin,end = 16,24
#begin,end = 1,30
categorycol=38
raw_data = urllib.urlopen("train")
dataset = np.loadtxt(raw_data, delimiter=",", usecols=range(cols))
x = dataset[:,begin:end]
y = dataset[:,categorycol]

varify_data1 = urllib.urlopen("varification")
#varify_dataset = np.loadtxt(varify_data1, delimiter=",", usecols=range(cols-1))
varify_dataset = np.loadtxt(varify_data1, delimiter=",", usecols=range(cols))
z = varify_dataset[:,begin:end]
a = varify_dataset[:,categorycol]

color_data = urllib.urlopen("colors")
#varify_dataset = np.loadtxt(varify_data1, delimiter=",", usecols=range(cols-1))
color_dataset = np.loadtxt(color_data, delimiter=",", usecols=range(3))
color_x = color_dataset[:,0:2]
color_y = color_dataset[:,2]


from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
# normalize the data attributes
normalized_X = preprocessing.normalize(x)
# standardize the data attributes
standardized_X = preprocessing.scale(x)

plt.figure(figsize=(12, 12))

import matplotlib.pyplot as pl
from sklearn import decomposition
pca = decomposition.PCA(n_components=2)
pca.fit(x)
X = pca.transform(x)
pl.subplot(331)
plt.title("source", size=18)
pl.scatter(X[:,0], X[:,1], c=y)

from sklearn import cluster
model = cluster.MiniBatchKMeans(n_clusters=CLUSTERS)
x_pred = model.fit_predict(x)
print x_pred
pl.subplot(332)
plt.title("MiniBatchKMeans", size=18)
pl.scatter(X[:,0], X[:,1], c=x_pred)

# connectivity matrix for structured Ward
connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)
# make connectivity symmetric
connectivity = 0.5 * (connectivity + connectivity.T)
ward = cluster.AgglomerativeClustering(n_clusters=CLUSTERS, linkage='ward',
                                       connectivity=connectivity)
w_pred = ward.fit_predict(x)
print w_pred
pl.subplot(333)
plt.title("ward", size=18)
pl.scatter(X[:,0], X[:,1], c=w_pred)

bandwidth = cluster.estimate_bandwidth(X, quantile=0.3)
ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms_pred = ms.fit_predict(x)
pl.subplot(334)
plt.title("MeanShift", size=18)
pl.scatter(X[:,0], X[:,1], c=ms_pred)

spectral = cluster.SpectralClustering(n_clusters=CLUSTERS,
                                      eigen_solver='arpack',
                                      affinity="nearest_neighbors")
spectral_pred = spectral.fit_predict(x)
pl.subplot(335)
plt.title("SpectralClustering", size=18)
pl.scatter(X[:,0], X[:,1], c=spectral_pred)

dbscan = cluster.DBSCAN(eps=.2)
dbscan_pred = dbscan.fit_predict(x)
pl.subplot(335)
plt.title("DBSCAN", size=18)
pl.scatter(X[:,0], X[:,1], c=dbscan_pred)

affinity_propagation = cluster.AffinityPropagation(damping=.9,
                                                   preference=-200)
affinity_propagation_pred = affinity_propagation.fit_predict(x)
pl.subplot(336)
plt.title("AffinityPropagation", size=18)
pl.scatter(X[:,0], X[:,1], c=affinity_propagation_pred)

average_linkage = cluster.AgglomerativeClustering(
    linkage="average", affinity="cityblock", n_clusters=CLUSTERS,
    connectivity=connectivity)
average_linkage_pred = average_linkage.fit_predict(x)
pl.subplot(337)
plt.title("AgglomerativeClustering", size=18)
pl.scatter(X[:,0], X[:,1], c=average_linkage_pred)

birch = cluster.Birch(n_clusters=CLUSTERS)
birch_pred = birch.fit_predict(x)
pl.subplot(338)
plt.title("Birch", size=18)
pl.scatter(X[:,0], X[:,1], c=birch_pred)

pl.subplot(339)
plt.title("Color", size=18)
pl.scatter(color_x[:,0], color_x[:,1], c=color_y)