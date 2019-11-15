import cv2 as cv
import numpy as np
import glob
import pickle
import ml_metrics
import math
import pandas as pd
import os
import yaml
from matplotlib import pyplot as plt 
from evaluation_funcs import performance_accumulation_pixel
from evaluation_funcs import performance_evaluation_pixel
from bbox_iou import bbox_iou
import imutils


def compute_mask(img, im_or, name, qs, alpha, x, y, w, h):
    qs_l = '../qs/' + qs + '/*.jpg'
    # Compute height and width of original image
    or_height, or_width = im_or.shape[:2]
    # Apply another median filter
    img_median = cv.medianBlur(img,9)

    # Compute masks in two channels of the image
    img1 = img_median[:,:,1]
    img2 = img_median[:,:,2]

    height, width = np.shape(img1)

    # Compute morphological gradient by dilation (keep inner edges to remove wall posters)
    kernel = np.ones((20,20),np.uint8)
    img_dilation1 = cv.dilate(img1,kernel,iterations = 1)
    img_gradient1 = img_dilation1 - img1
    
    kernel = np.ones((40,40),np.uint8)
    img_dilation2 = cv.dilate(img2,kernel,iterations = 1)
    img_gradient2 = img_dilation2 - img2

    # Thresholding
    _,img_th1 = cv.threshold(img_gradient1, 20, 255, cv.THRESH_BINARY)
    _,img_th2 = cv.threshold(img_gradient2, 20, 255, cv.THRESH_BINARY)
    
    # Computing external contours
    _ ,contours1, _ = cv.findContours(img_th1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    img_contour1 = np.zeros_like(img_th1)
    cv.drawContours(img_contour1, contours1, -1, 255, -1)
    
    _ ,contours2, _ = cv.findContours(img_th2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    img_contour2 = np.zeros_like(img_th2)
    cv.drawContours(img_contour2, contours2, -1, 255, -1)

    # Opening to remove wall posters & others according to the image size
    size_h = round((1/6)*height)
    size_w = round((1/6)*width)
    kernel = np.ones((size_h,size_w),np.uint8)
    
    mask1 = cv.morphologyEx(img_contour1, cv.MORPH_OPEN, kernel)
    mask2 = cv.morphologyEx(img_contour2, cv.MORPH_OPEN, kernel)
    
    # Avoid all zeros image
    if(mask1.any()==0 and mask2.any()==0):
        mask = img_contour1
    else:
        mask = mask1 + mask2
    
    # Rotate mask back to its original position and crop
    rotated_mask = imutils.rotate(mask, alpha)
    mask = rotated_mask[y:y+h,x:x+w]
    mask = cv.resize(mask, (or_width, or_height))

    # Save mask
    cv.imwrite('masks/' + name + '.png', mask)

    # Compute evaluation metrics only if development set
    if (qs == 'qsd2_w1' or qs == 'qsd2_w2' or qs == 'qsd2_w3' or qs == 'qsd3_w3' or qs == 'qsd1_w4'):
        # Read ground truth
        g_t = cv.imread('../qs/' + qs + '/' + name + '.png', cv.IMREAD_COLOR)
        g_t = cv.cvtColor(g_t, cv.COLOR_BGR2GRAY)

        # Compute evaluation metrics
        pixelTP, pixelFP, pixelFN, pixelTN = performance_accumulation_pixel(mask,g_t)
        pixel_precision, pixel_accuracy, pixel_specificity, pixel_sensitivity = performance_evaluation_pixel(pixelTP, pixelFP, pixelFN, pixelTN)
        F1 = 2*pixel_precision*pixel_sensitivity/(pixel_precision+pixel_sensitivity)

        eval_metrics = [pixel_precision, pixel_accuracy, pixel_specificity, pixel_sensitivity, F1]
        '''
        print("Precision: "+str(pixel_precision))
        print("Accuracy: "+str(pixel_accuracy))
        print("Specificity: "+str(pixel_specificity))
        print("Recall (sensitivity): "+str(pixel_sensitivity))
        print("F1: "+str(F1))
        '''
    else:
        eval_metrics = [0,0,0,0,0]

    # DETECT IF THERE ARE TWO IMAGES
    # First method: check if the central mask is black
    '''
    central_column = round(width/2)
    central_column_mean = np.mean(mask[:,central_column:central_column+1])

    # If central column is not zero, analyze some extra columns
    # From 0.25 to 0.75 of the image, with a step of 100px
    if central_column_mean != 0:
        for i in range(round(0.5*(width/2)), round(1.5*(width/2)), 100):
            central_column_mean = np.mean(mask[:,i:i+1])
            if (central_column_mean == 0): # If found, exit for and keep central_column
                central_column = i
                break

    # If after the second attempt two masks are detected 
    if central_column_mean == 0:
        # Generate white masks
        mask_left = np.ones((height,width),np.uint8)
        mask_right = np.ones((height,width),np.uint8)

        # Compute
        mask_left[:,central_column:width] = 0
        mask_right[:,0:central_column] = 0
        mask_left = mask_left*mask
        mask_right = mask_right*mask
        bg_mask = [mask_left, mask_right]
        
        # If one mask is black, keep only the correct one
        if mask_left.any()==0:
            bg_mask = [mask_right]
        if mask_right.any()==0:
            bg_mask = [mask_left]

    else:
        bg_mask= [mask]
    '''

    # Second method: detect contours
    _, contours, hier = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # If two paintings, return 2 masks
    if hier is not None:
        if(np.shape(hier)[1] == 2):
            mask_1 = np.zeros_like(mask)
            mask_0 = np.zeros_like(mask)
            cv.fillPoly(mask_1, pts =[contours[0]], color=(255,255,255))
            cv.fillPoly(mask_0, pts =[contours[1]], color=(255,255,255))

            # Left and right mask in the correct order
            indices_0 = np.where(mask_0 != [0])
            indices_1 = np.where(mask_1 != [0])

            if(indices_0[0].size != 0 and indices_0[1].size !=0 and indices_1[0].size != 0 and indices_1[1].size !=0 ):
                if( min(indices_0[1]) < min(indices_1[1]) ):
                    bg_mask = [mask_0, mask_1]
                else:
                    bg_mask = [mask_1, mask_0]
            
        # If three paintings, return 3 masks
        elif(np.shape(hier)[1] == 3):
            mask_0 = np.zeros_like(mask)
            mask_1 = np.zeros_like(mask)
            mask_2 = np.zeros_like(mask)
            
            cv.fillPoly(mask_0, pts =[contours[0]], color=(255,255,255))
            cv.fillPoly(mask_1, pts =[contours[1]], color=(255,255,255))
            cv.fillPoly(mask_2, pts =[contours[2]], color=(255,255,255))

            masks = [mask_0, mask_1, mask_2]
            
            # Left and right mask in the correct order
            indices_0 = np.where(mask_0 != [0])
            indices_1 = np.where(mask_1 != [0])
            indices_2 = np.where(mask_2 != [0])

            if(indices_0[0].size != 0 and indices_0[1].size !=0 and indices_1[0].size != 0 and indices_1[1].size !=0 and indices_2[0].size != 0 and indices_2[1].size !=0 ):
                # Check which painting is on the left (lower Y index)
                minimum = [min(indices_0[1]), min(indices_1[1]), min(indices_2[1])]
                indices = np.argsort(minimum)
                
                # Return masks in order
                bg_mask = [masks[indices[0]], masks[indices[1]], masks[indices[2]]]
        
        else:
            bg_mask = [mask]
    else:
        bg_mask = [mask]

    return bg_mask, eval_metrics, contours