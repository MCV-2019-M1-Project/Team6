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

def text_removal_mask(img_gray, name, strel, strel_pd, num_cols, coords, background_mask, qs):
            
    # Obtain image dimensions
    height,width = img_gray.shape[:2]
    
    c=[]

    length = np.shape(background_mask)[0]

    # Create variable where final mask will be stored
    #f_mask = np.ones(shape=(height,width))
    f_mask = []

    for picture in range(length):
        
        # Boundaries of the analyzed area
        columns = np.array(background_mask[picture]).sum(axis=0)
        rows = np.array(background_mask[picture]).sum(axis=1)
        non_zero_col = np.nonzero(columns)
        non_zero_col = np.array(non_zero_col[0])

        if non_zero_col != []:
            min_a = round((non_zero_col[0] + non_zero_col[len(non_zero_col)-1])/2)
            max_a = round(min_a + num_cols)

            # Store pixel values in the analyzed area
            values_t = np.zeros(shape=(int(max_a-min_a), int(max_a-min_a)))
        else:
            min_a = width/2
            max_a = min_a + num_cols

        non_zero_row = np.nonzero(rows)
        non_zero_row = np.array(non_zero_row[0])

        if non_zero_row != []:
            min_h = non_zero_row[0]
            max_h = non_zero_row[len(non_zero_row)-1]
        else:
            min_h = 0
            max_h = height - 1
                
        i = 0

        
        for p in range(int(min_a), int(max_a)):
            # Per each column, compute number of ocurrences of every pixel value
            col = img_gray[min_h:max_h,p]
            
            # Pixel values and number of ocurrences for the whole column
            values = pd.Series(col).value_counts().keys().tolist()
            
            # Get highest pixel values (most frequent ones)
            values_t[0:4,i] = values[0:4]

            i += 1

        j = 0
        w = 0
        h = 0

        while((w*h < 20000 or w*h > 1000000 or w<h) and j < num_cols):

            level = round(np.mean(values_t[j,:]))
            
            if level <= 128:
                final_img = cv.morphologyEx(img_gray, cv.MORPH_OPEN, strel)
                #mask = final_img == level #Method 1
                mask = (final_img >= max(0,level-1)) * (final_img <= level+1) #Method 2
                mask = mask.astype(np.uint8)
                mask = mask*background_mask[picture]
                #mask *= 255
            elif level > 128:
                final_img = cv.morphologyEx(img_gray, cv.MORPH_CLOSE, strel)
                #mask = final_img == level #Method 1
                mask = (final_img >= max(0,level-1)) * (final_img <= level+1) #Method 2
                mask = mask.astype(np.uint8)
                mask = mask*background_mask[picture]
                #mask *= 255

            if np.sum(mask) != 0:

                # Find contours of created mask
                contours,_ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

                # Find largest contour (it will contain the text bounding box)
                contour_sizes = [(cv.contourArea(contour), contour) for contour in contours]
                largest_contour = max(contour_sizes, key=lambda x: x[0])[1]
                
                # Find bounding box belonging to detected contour
                (x,y,w,h) = cv.boundingRect(largest_contour)

                # Draw bounding boxes coordinates on original image to visualize it
                cv.rectangle(img_gray, (x,y), (x+w,y+h), (0,255,0), 2)
                
                # Bboxes coordinates of the text positions
                tlx = x
                tly = y
                brx = x + w
                bry = y + h
            j+=1

        background_mask[picture][y:y + h, x:x + w] = 0
        cv.imwrite('results/'+ name + str(picture) + 'txtrmask.png', background_mask[picture])

        # Add bboxes coordinates to a list of lists
        if qs != 'qsd2_w3' and  qs != 'qst2_w3':
            f_mask.append(background_mask[picture])
        elif picture < length-1:
            c.append([(tlx,tly,brx,bry)])
            f_mask.append(background_mask[picture])
        else:
            f_mask.append(background_mask[picture])
            c.append([(tlx,tly,brx,bry)])
            coords.append([c])

        # Create new mask with bboxes coordinates
        # Bboxes pixels are black (text), white otherwise (painting)

        #f_mask[y:y + h, x:x + w] = 0
        #f_mask = f_mask + background_mask[picture]

    return f_mask, coords