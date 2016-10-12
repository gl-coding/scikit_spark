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
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x, y)
#print(model)
#print model.predict(z)
print "+++++++++++++++++++++++++++++++++++++++"
print "DecisionTreeClassifier"
z_pred = model.predict(z)
print z_pred
print metrics.accuracy_score(a, z_pred)

#exit()

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x,y)
#print model.predict(z[0])
print "+++++++++++++++++++++++++++++++++++++++"
print "LogisticRegression"
z_pred = model.predict(z)
print z_pred
print metrics.accuracy_score(a, z_pred)

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(x,y)
#print model.predict(z[0])
print "+++++++++++++++++++++++++++++++++++++++"
print "GaussianNB"
z_pred = model.predict(z)
print z_pred
print metrics.accuracy_score(a, z_pred)

from sklearn.svm import SVC
model = SVC()
model.fit(x, y)
#print model.predict(z[0])
print "+++++++++++++++++++++++++++++++++++++++"
print "svc"
z_pred = model.predict(z)
print z_pred
print metrics.accuracy_score(a, z_pred)

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(x, y)
#print model.predict(z[0]);
print "+++++++++++++++++++++++++++++++++++++++"
print "KNeighborsClassifier"
z_pred = model.predict(z)
print z_pred
print metrics.accuracy_score(a, z_pred)

from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(x, y)
print model.feature_importances_
