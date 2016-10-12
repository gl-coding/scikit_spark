#!/usr/bin/python
import random
import os
import sys

category = int(sys.argv[1])
filename = "colors"

if os.path.exists(filename):
    os.remove(filename)

start = -100
step = 100
rangelen = 20
pointNum = 100

for i in range(0, category):
    x = start + i * step
    y = 0
    for j in range(0, pointNum):
        tmpx = x + random.randint(-rangelen, rangelen)
        tmpy = y + random.randint(-rangelen, rangelen)
        line = str(tmpx) + "," + str(tmpy) + "," + str(i)
        command = "echo " + line + " >> " + filename
        os.system(command)
