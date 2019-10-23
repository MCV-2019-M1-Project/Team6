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


def compute_mask(img, name, qs):
    height, width = np.shape(img)

    # Compute morphological gradient by dilation (keep inner edges to remove wall posters)
    kernel = np.ones((40,40),np.uint8)
    img_dilation = cv.dilate(img,kernel,iterations = 1)
    img_gradient = img_dilation - img
    
    # Thresholding
    _,img_th = cv.threshold(img_gradient, 30, 255, cv.THRESH_BINARY)
    #retval,mask_img = cv.threshold(final_img, 30, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

    # Computing external contours
    contours, _ = cv.findContours(img_th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    img_contour = np.zeros_like(img_th)
    cv.drawContours(img_contour, contours, -1, 255, -1)
    
    # Opening to remove wall posters
    kernel = np.ones((100,200),np.uint8)
    mask = cv.morphologyEx(img_contour, cv.MORPH_OPEN, kernel)

    # Avoid all zeros image
    if(mask.any()==0):
        mask = img_contour

    # Save mask
    cv.imwrite('masks/' + name + '.png', mask)
    
    if qs == 'qsd1_w3' or qs == 'qsd2_w3':
        # Read ground truth
        g_t = cv.imread('../qs/' + qs + '/' + name + '.png', cv.IMREAD_COLOR)
        g_t = cv.cvtColor(g_t, cv.COLOR_BGR2GRAY)
        
        # Compute evaluation metrics
        pixelTP, pixelFP, pixelFN, pixelTN = performance_accumulation_pixel(mask,g_t)
        pixel_precision, pixel_accuracy, pixel_specificity, pixel_sensitivity = performance_evaluation_pixel(pixelTP, pixelFP, pixelFN, pixelTN)
        F1 = 2*pixel_precision*pixel_sensitivity/(pixel_precision+pixel_sensitivity)
        
        eval_metrics = [pixel_precision, pixel_accuracy, pixel_specificity, pixel_sensitivity, F1]

    else:
        eval_metrics = None

    # DETECT IF THERE ARE TWO IMAGES
    # First method: check if the central mask is black
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
   
    else:
         bg_mask= [mask]

    return bg_mask, eval_metrics