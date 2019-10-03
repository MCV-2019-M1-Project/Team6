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
    hue=img[:,:,0]
    sat=img[:,:,1]
    val=img[:,:,2]
    
# =============================================================================
#     plt.imshow(img[:,:,1])
#     plt.show()
# =============================================================================

    sample = hue[0,0]
    print(sample)
    height,width=hue.shape[:2]
    
    #Amount of pixels per each side of the image
    percent=0.03
    aop_h=int(round(percent*height))
    aop_w=int(round(percent*width))
    
    print(aop_h)
    print(aop_w)

    #Crop different portions of the image containing the background
    portion=hue[0:aop_h, 0:width]
    portion2=hue[0:height,0:aop_w]
    portion3=hue[0:height,width-aop_w:width]
    portion4=hue[height-aop_h:height,0:width]
    

# =============================================================================
# plt.imshow(portion3)
# plt.show()
# =============================================================================
    
    #Compute thresholds from each portion
    min_p1=int(np.amin(portion))
    max_p1=int(np.amax(portion))
    
    min_p2=int(np.amin(portion2))
    max_p2=int(np.amax(portion2))

    min_p3=int(np.amin(portion3))
    max_p3=int(np.amax(portion3))
    
    min_p4=int(np.amin(portion4))
    max_p4=int(np.amax(portion4))

    #Compute absolute thresholds from all portions
    minval=min(min_p1,min_p2)
    maxval=min(max_p1,max_p2)
    
    #Matrix where the mask will be stored
    mask=np.zeros((height,width))
    
    #Loop over the channel and create the mask
    for i in xrange(hue.shape[0]):
        for j in xrange(hue.shape[1]):
            if hue[i,j]>=minval and hue[i,j]<=maxval:
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







    