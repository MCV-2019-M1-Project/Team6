#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 11:29:17 2019
LET'S TRY STUFF
@author: jordi
"""
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

def read_im(name):
    im = cv2.imread(name)
    return im

db = []
images = glob.glob('./db/*.jpg')

acc=np.zeros((3,256))

#Number of elements in database
print(len(images))

for p in images:
    img=cv2.imread(p,cv2.IMREAD_COLOR)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    
    #Compute histogram for each channel of database images
    hist_1=cv2.calcHist([img],[0],None,[256],[0,256])
    hist_2=cv2.calcHist([img],[1],None,[256],[0,256])
    hist_3=cv2.calcHist([img],[2],None,[256],[0,256])

    #Concatenate all channels
    hist_db=np.concatenate((hist_1,hist_2,hist_3))
    db.append(hist_db) 

#Query image
q_im = cv2.imread('starred.jpg',cv2.IMREAD_COLOR)
# cv2.imshow('paint',q_im)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#Histogram for each channel from query image
img=cv2.cvtColor(q_im,cv2.COLOR_BGR2HSV)
q_hist_1=cv2.calcHist([img],[0],None,[256],[0,256])
q_hist_2=cv2.calcHist([img],[1],None,[256],[0,256])
q_hist_3=cv2.calcHist([img],[2],None,[256],[0,256])

#Concatenate all channels
q_hist=np.concatenate((q_hist_1,q_hist_2,q_hist_3))
print(q_hist.shape)

plt.hist(img.ravel(),256,[0,256])
plt.show()

#Find k most similar images to the query one
dist=[]

for h in db:
    #Euclidian distance to measure distance between histograms
    distance=math.sqrt(sum(pow(h-q_hist,2)))
    print(distance)

