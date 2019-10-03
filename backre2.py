#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
BACKGROUND REMOVAL
@author: jordi
"""
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os

#Read QSD2
images = glob.glob('./qsd2_w1/*.jpg')
print("Query set 2 has " + str(len(images))+" images")

img=cv.imread('qsd2_w1/00006.jpg',cv.IMREAD_COLOR)
gray_img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
hist = cv.calcHist([gray_img],[0],None,[256],[0,256])
hist_norm = hist/(gray_img.shape[0]*gray_img.shape[1])

height,width=gray_img.shape[:2]
maxval=np.argmax(hist_norm)

mask=np.zeros((height,width))
    
for i in xrange(gray_img.shape[0]):
    for j in xrange(gray_img.shape[1]):
        if gray_img[i,j]>=maxval-30 and gray_img[i,j]<=maxval+30:
            mask[i,j]=255
        else:
            mask[i,j]=0
                
cv.imwrite('prova.png',mask)

# =============================================================================
# plt.hist(gray_img.ravel(),256,[0,256])
# plt.show()
# 
# ind = np.unravel_index(np.argmax(hist_norm, axis=None), hist_norm.shape)
# 
# print(ind)
# =============================================================================

# =============================================================================
# hist_norm_1=np.array(hist_norm)
# print(np.argmax(hist_norm_1))
# 
# k =hist_norm_1.flatten()
# print(k.sort())
# 
# =============================================================================

# =============================================================================
# print(hist_norm)
# print("new one")
# print(hist_norm_1)
# 
# #print(hist_norm_1)
# 
# print ("Input unsorted array : ", hist_norm_1) 
# out_arr = np.argsort(hist_norm_1) 
# print ("Output sorted array indices : ", out_arr) 
# print("Output sorted array : ", hist_norm_1[out_arr]) 
# =============================================================================



# =============================================================================
# print(np.argmax(hist_norm))
# print(np.argsort(hist_norm))
# idx = (-hist).argsort()[:4]
# print(idx)
# =============================================================================

# =============================================================================
# idx = (-hist).argsort()[:4]
# print(idx)
# 
# plt.hist(gray_img.ravel(),256,[0,256])
# plt.show()
# =============================================================================

# =============================================================================
# print(np.argpartition(hist, -4)[-4:])
# print(np.argmax(hist))
# print(np.sort(hist))
# =============================================================================

maxval = int(np.argmax(hist))
mask = gray_img==maxval
plt.imshow(mask)
plt.show()

# =============================================================================
# print hist_norm
# plt.hist(gray_img.ravel(),256,[0,256])
# plt.show()
# =============================================================================

# =============================================================================
# plt.hist(img.ravel(),64,[0,256])
# plt.show()
# 
# cv.imshow('paint',gray_img)
# cv.waitKey(0)
# cv.destroyAllWindows()
# =============================================================================

