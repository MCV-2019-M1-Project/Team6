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
    

#Parameters
COLORSPACE = cv.COLOR_BGR2HSV

#Read QSD2
images = glob.glob('./qsd2_w1/*.jpg')
print("Query set 2 has " + str(len(images))+" images")

for f in sorted(images):
    
    #Numeric identifier for every image
    name=os.path.splitext(os.path.split(f)[1])[0]
    print("Image name:"+str(name))
    
    #Read ground truth mask
    g_t=cv.imread('qsd2_w1/'+name+'.png',cv.IMREAD_COLOR)
    g_t=img=cv.cvtColor(g_t,cv.COLOR_BGR2GRAY)

    #Read image and conver it to the chosen color space
    img=cv.imread(f,cv.IMREAD_COLOR)
    img=cv.cvtColor(img,COLORSPACE)
    
    #Compute channel on the chosen color space
    num_ch=1
    ch=img[:,:,num_ch]
  
    #Height and width of channel (=image dims)
    height,width=ch.shape[:2]
    
    #Amount of pixels per each side of the image
    percent=0.03
    aop_h=int(round(percent*height))
    aop_w=int(round(percent*width))
    
    #Crop different portions of the image containing the background
    portion=ch[0:aop_h, 0:width]
    portion2=ch[0:height,0:aop_w]
    portion3=ch[0:height,width-aop_w:width]
    portion4=ch[height-aop_h:height,0:width]
     
    #Compute thresholds from chosen image sides
    min_p1=int(np.amin(portion))
    max_p1=int(np.amax(portion))
    
    min_p2=int(np.amin(portion2))
    max_p2=int(np.amax(portion2))

    min_p3=int(np.amin(portion3))
    max_p3=int(np.amax(portion3))
    
    min_p4=int(np.amin(portion4))
    max_p4=int(np.amax(portion4))

    #Compute absolute thresholds from chosen portions
    minval=min(min_p1,min_p2)
    maxval=min(max_p1,max_p2)
    
    mask=np.zeros((height,width))
    
    cv.imwrite('newmasks/'+name+'.png',mask)
    
    #Loop over the channel and create the mask
    for i in xrange(ch.shape[0]):
        for j in xrange(ch.shape[1]):
            if ch[i,j]>=minval and ch[i,j]<=maxval:
                #Pixels belong to the background
                mask[i,j]=0
            else:
                #Pixels belong to the painting
                mask[i,j]=255
    
    #Save binary mask
    cv.imwrite('masks/'+name+'.png',mask)
    

    #Compute evaluation metrics    
    pixelTP,pixelFP,pixelFN,pixelTN = performance_accumulation_pixel(mask,g_t)
    pixel_precision, pixel_accuracy, pixel_specificity, pixel_sensitivity = performance_evaluation_pixel(pixelTP, pixelFP, pixelFN, pixelTN)

    print("Precision: "+str(pixel_precision))
    print("Accuracy: "+str(pixel_accuracy))
    print("Specificity: "+str(pixel_specificity))
    print("Recall (sensitivity): "+str(pixel_sensitivity))