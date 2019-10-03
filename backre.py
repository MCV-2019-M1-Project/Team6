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

for f in sorted(images):
    #Compute channels on the chosen color space
    img=cv.imread(f,cv.IMREAD_COLOR)
    img=cv.cvtColor(img,cv.COLOR_BGR2HSV)
    sat=img[:,:,0]
    hue=img[:,:,1]
    val=img[:,:,2]
    
# =============================================================================
#     plt.imshow(img[:,:,1])
#     plt.show()
# =============================================================================

    sample = hue[0,0]
    print(sample)
    height,width=hue.shape[:2]
    
    percent=0.4

    #Crop a portion of the background
    portion=hue[0:60, 0:width]
    portion2=hue[0:height,0:60]
    portion3=hue[0:height,0:width-20]

# =============================================================================
# plt.imshow(portion3)
# plt.show()
# =============================================================================

    min_p1=int(np.amin(portion))
    max_p1=int(np.amax(portion))
    
    min_p2=int(np.amin(portion2))
    max_p2=int(np.amax(portion2))

# =============================================================================
# min_p3=int(np.amin(portion3))
# max_p3=int(np.amax(portion3))
# =============================================================================

    minval=min(min_p1,min_p2)
    maxval=min(max_p1,max_p2)
    
    
    mask=np.zeros((height,width))
    
    for i in xrange(hue.shape[0]):
        for j in xrange(hue.shape[1]):
            if hue[i,j]>=min_p1 and hue[i,j]<=max_p1:
                mask[i,j]=0
            else:
                mask[i,j]=255
    
    name=os.path.splitext(os.path.split(f)[1])[0]
    print(mask[0,0])
    cv.imwrite('masks/'+name+'.png',mask)
# =============================================================================
#     plt.imshow(mask)
#     plt.show()
# =============================================================================


# =============================================================================
# print("height and width")
# # =============================================================================
# # print(height)
# # print(width)
# # =============================================================================
# 
# mask_2=img[:,:,0]
# 
# mask_2 = hue>(sample+20) 
# print("types")
# print(type(mask_2))
# print(type(hue))
# 
# #cv.imwrite('mask1.png',cv.Umat(mask_2))
# 
# print(type(mask_2))
# cv.imwrite('mask.jpg',hue)
# 
# print("hue im type")
# # =============================================================================
# # print(type(hue))
# # =============================================================================
# 
# 
# # =============================================================================
# # print("min value")
# # print(np.amin(hue))
# # 
# # print("max value")
# # print(np.amax(hue))
# # =============================================================================
# 
# plt.imshow(mask_2)
# plt.show()
# =============================================================================







    