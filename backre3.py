#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 21:22:01 2019
BACKGROUND REMOVAL 
@author: jordi
"""

import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os

################FUNCTIONS COPIED FROM PROJECT GITHUB#################################
def performance_accumulation_pixel(pixel_candidates, pixel_annotation):
    """ 
    performance_accumulation_pixel()

    Function to compute different performance indicators 
    (True Positive, False Positive, False Negative, True Negative) 
    at the pixel level
       
    [pixelTP, pixelFP, pixelFN, pixelTN] = performance_accumulation_pixel(pixel_candidates, pixel_annotation)
       
    Parameter name      Value
    --------------      -----
    'pixel_candidates'   Binary image marking the foreground areas
    'pixel_annotation'   Binary image containing ground truth
       
    The function returns the number of True Positive (pixelTP), False Positive (pixelFP), 
    False Negative (pixelFN) and True Negative (pixelTN) pixels in the image pixel_candidates
    """
    
    pixel_candidates = np.uint64(pixel_candidates>0)
    pixel_annotation = np.uint64(pixel_annotation>0)
    
    pixelTP = np.sum(pixel_candidates & pixel_annotation)
    pixelFP = np.sum(pixel_candidates & (pixel_annotation==0))
    pixelFN = np.sum((pixel_candidates==0) & pixel_annotation)
    pixelTN = np.sum((pixel_candidates==0) & (pixel_annotation==0))


    return [pixelTP, pixelFP, pixelFN, pixelTN]

def performance_evaluation_pixel(pixelTP, pixelFP, pixelFN, pixelTN):
    """
    performance_evaluation_pixel()

    Function to compute different performance indicators (Precision, accuracy, 
    specificity, sensitivity) at the pixel level
    
    [pixelPrecision, pixelAccuracy, pixelSpecificity, pixelSensitivity] = PerformanceEvaluationPixel(pixelTP, pixelFP, pixelFN, pixelTN)
    
       Parameter name      Value
       --------------      -----
       'pixelTP'           Number of True  Positive pixels
       'pixelFP'           Number of False Positive pixels
       'pixelFN'           Number of False Negative pixels
       'pixelTN'           Number of True  Negative pixels
    
    The function returns the precision, accuracy, specificity and sensitivity
    """

    pixel_precision   = 0
    pixel_accuracy    = 0
    pixel_specificity = 0
    pixel_sensitivity = 0
    if (pixelTP+pixelFP) != 0:
        pixel_precision   = float(pixelTP) / float(pixelTP+pixelFP)
    if (pixelTP+pixelFP+pixelFN+pixelTN) != 0:
        pixel_accuracy    = float(pixelTP+pixelTN) / float(pixelTP+pixelFP+pixelFN+pixelTN)
    if (pixelTN+pixelFP):
        pixel_specificity = float(pixelTN) / float(pixelTN+pixelFP)
    if (pixelTP+pixelFN) != 0:
        pixel_sensitivity = float(pixelTP) / float(pixelTP+pixelFN)

    return [pixel_precision, pixel_accuracy, pixel_specificity, pixel_sensitivity]
################FUNCTIONS COPIED FROM PROJECT GITHUB#################################
    

COLORSPACE = cv.COLOR_BGR2HSV

images = glob.glob('./qsd2_w1/*.jpg')
print("Query set 2 has " + str(len(images))+" images")
num=0
num2=0
num3=0
for f in sorted(images):
    

    name=os.path.splitext(os.path.split(f)[1])[0]
    g_t=cv.imread('qsd2_w1/'+name+'.png',cv.IMREAD_COLOR)
    g_t=cv.cvtColor(g_t,cv.COLOR_BGR2GRAY)
    #print("Image name:"+str(name))
    
    im_c=cv.imread(f,cv.IMREAD_COLOR)
    im=cv.cvtColor(im_c,COLORSPACE)
    
    c0=im[:,:,0]
    c1=im[:,:,1]
    c2=im[:,:,2]
    
    #Height and width of channel (=image dims)
    height,width=c0.shape[:2]
        
    #Amount of pixels per each side of the image
    percent_c0=0.02
    percent_c1=0.02
    percent_c2=0.02
    
    aop_h_c0=int(round(percent_c0*height))
    aop_w_c0=int(round(percent_c0*width))
    
    aop_h_c1=int(round(percent_c1*height))
    aop_w_c1=int(round(percent_c1*width))
    
    aop_h_c2=int(round(percent_c2*height))
    aop_w_c2=int(round(percent_c2*width))
    
    
    portionc0_1=c0[0:aop_h_c0, 0:width]
    portionc1_1=c1[0:aop_h_c1, 0:width]
    portionc2_1=c2[0:aop_h_c2, 0:width]
    
    portionc0_2=c0[height-aop_h_c0:height,0:width]
    portionc1_2=c1[height-aop_h_c1:height,0:width]
    portionc2_2=c2[height-aop_h_c2:height,0:width]
    
    min_c0_1=int(np.amin(portionc0_1))
    min_c1_1=int(np.amin(portionc1_1))
    min_c2_1=int(np.amin(portionc2_1))
    
    min_c0_2=int(np.amin(portionc0_2))
    min_c1_2=int(np.amin(portionc1_2))
    min_c2_2=int(np.amin(portionc2_2))
    
    max_c0_1=int(np.amax(portionc0_1))
    max_c1_1=int(np.amax(portionc1_1))
    max_c2_1=int(np.amax(portionc2_1))
    
    max_c0_2=int(np.amax(portionc0_2))
    max_c1_2=int(np.amax(portionc1_2))
    max_c2_2=int(np.amax(portionc2_2))
    
    min_c0=min(min_c0_1,min_c0_2)
    min_c1=min(min_c1_1,min_c1_2)
    min_c2=min(min_c2_1,min_c2_2)
    
    max_c0=max(max_c0_1,max_c0_2)
    max_c1=max(max_c1_1,max_c1_2)
    max_c2=max(max_c2_1,max_c2_2)


    mask = 255-(cv.inRange(im,(min_c0,min_c1,min_c2),(max_c0,max_c1,max_c2)))
    #final_mask=cv.bitwise_and(cv.cvtColor(im,cv.COLOR_BGR2GRAY),mask)
    
    cv.imwrite('masks/'+name+'.png',mask)
    
    
    pixelTP,pixelFP,pixelFN,pixelTN = performance_accumulation_pixel(mask,g_t)
    pixel_precision, pixel_accuracy, pixel_specificity, pixel_sensitivity = performance_evaluation_pixel(pixelTP, pixelFP, pixelFN, pixelTN)
    
    #Multiplies mask and original image
    mask=cv.imread('masks/'+name+'.png',cv.IMREAD_COLOR)
    comp=cv.bitwise_and(im_c,mask)
    cv.imwrite('masks/'+name+'comp.jpg',comp)
    
    #print("Precision: "+str(pixel_precision))
    #print("Accuracy: "+str(pixel_accuracy))
    #print("Specificity: "+str(pixel_specificity))
    #print("Recall (sensitivity): "+str(pixel_sensitivity))
    
    
    if (pixel_precision+pixel_sensitivity !=0):
        fscore=(2*pixel_precision*pixel_sensitivity)/(pixel_precision+pixel_sensitivity)
        #print("F1-score: "+str(fscore))
        if fscore>=0.9:
            num=num+1
        elif fscore>=0.7 and fscore<0.9:
            num2=num2+1
        elif fscore<0.7:
            num3=num3+1
    
print("Total number of paintings with Fscore > 0.9: " + str(num))
print("Total number of paintings with Fscore between 0.7 and 0.9: " + str(num2))
print("Total number of paintings with Fscore below 0.7: " + str(num3))
# =============================================================================
# cv.imshow("cuadrito2",mask)
# cv.waitKey(0)
# cv.destroyAllWindows
# =============================================================================
