#!/usr/bin/python

import numpy as np
import urllib
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


from sklearn import preprocessing
# normalize the data attributes
normalized_X = preprocessing.normalize(x)
# standardize the data attributes
standardized_X = preprocessing.scale(x)

print "---------------------------------------"
from sklearn import cluster
model = cluster.KMeans(n_clusters=2)
model.fit(x)

print model.labels_[::10]
