#!/usr/bin/python

import numpy as np
import scipy as sp
import urllib
def error(f, x, y):
    return sp.sum((f(x)-y)**2)
cols=39
begin,end = 16,24
#begin,end = 1,30
categorycol=38
raw_data = urllib.urlopen("train")
dataset = np.loadtxt(raw_data, delimiter=",", usecols=range(cols))
x = dataset[:,0:end]
y = dataset[:,categorycol]

#a = x[:,1]
a = x[:,0]
print len(a)
b = x[:,2]

inflection=140

a1 = a[:inflection]
print len(a1)
print a1
b1 = b[:inflection]
print b1
a2 = a[inflection:]
print a2
b2 = b[inflection:]
print b2

fStraight1p = sp.polyfit(a1,b1,1)
fStraight1 = sp.poly1d(fStraight1p)
fStraight2p = sp.polyfit(a2,b2,1)
fStraight2 = sp.poly1d(fStraight2p)

import matplotlib.pyplot as plt
plt.scatter(a, b)
plt.autoscale(tight=True)
plt.grid()

fStraight1p = sp.polyfit(a,b,1)
fStraight1 = sp.poly1d(fStraight1p)

fp1= sp.polyfit(a, b, 4)
print fp1
aa = polyval(fp1, 5)
print aa
fStraight = sp.poly1d(fp1)
print "Error of Curve3 line:",error(fStraight,a,b)
#draw fitting straight line
fx = sp.linspace(0,a[-1], 10) # generate X-values for plotting
plt.plot(fx, fStraight(fx), linewidth=4)
plt.legend(["d=%i" % fStraight.order], loc="upper left")