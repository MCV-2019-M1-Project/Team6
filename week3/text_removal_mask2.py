# IMPORTS #
import cv2 as cv
import glob
import os
import numpy as np
from matplotlib import pyplot as plt

def find_text(img, background_mask, name):

    #Image pre-processing: the difference betwen the image opening and closing is computed
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2*6-1, 2*6-1))
    img_opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    img_closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    img_enhanced = img_closing - img_opening
    img_lowpass	= cv.GaussianBlur(img_enhanced, (5,5), 20)

    #Binarization: only the 4% brightest pixels are set to positive
    height = img.shape[0]
    width = img.shape[1]
    npx =  height*width
    hist = cv.calcHist([img_lowpass], [0], None, [256], [0, 255])/npx
    prob=0
    i=0
    while(prob<0.004):
        prob += hist[255-i][0]
        i+=1
    th=255-i
    ret,img_binary = cv.threshold(img_lowpass,th,255,cv.THRESH_BINARY)

    #Candidate text region extraction
    kernel = np.ones((15,30), np.uint8)
    img_dilated = cv.dilate(img_binary,kernel,iterations = 3)
    kernel = np.ones((10,10), np.uint8)
    img_eroded = cv.erode(img_dilated,kernel,iterations = 2)
    img_opening = cv.morphologyEx(img_eroded, cv.MORPH_OPEN, np.ones((15,15), np.uint8))

    num, labels = cv.connectedComponents(img_opening)
    sorted_labels = labels.ravel()
    sorted_labels = np.sort(sorted_labels)

    labels = np.uint8(labels)
    """
    cv.imshow("img_binary", labels*255)
    cv.waitKey(0)
    cv.destroyAllWindows()
    """
    #Discarting non desired regions
    if np.shape(background_mask)[0]>1:
        min_size = npx/(20*20)
    else:
        min_size = npx/(15*15)
    max_aspect_ratio = 0.7
    min_aspect_ratio = 1/15
    min_occupancy_ratio = 0.5
    min_compactness_ratio = 0.002
    coords = []

    for i in range(1, num+1):

        region = labels == i
        region = np.uint8(region)
        positions =  np.where(labels == i)
        size = len(positions[0])

        if size < min_size:

            labels[positions] = 0

        else:

            contours,_ = cv.findContours(region, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            contour_sizes = [(cv.contourArea(contour), contour) for contour in contours]
            largest_contour = max(contour_sizes, key=lambda x: x[0])[1]
            (x,y,w,h) = cv.boundingRect(largest_contour)
            perimeter = cv.arcLength(largest_contour,True)
            if h/w>max_aspect_ratio or h/w<min_aspect_ratio or len(positions[0])/(w*h)<min_occupancy_ratio or (w*h)/(perimeter*perimeter)<min_compactness_ratio:
                labels[positions] = 0
            else:
                coords.append((x,y,w,h,size))
    
    """
    cv.imshow("img_binary", labels*255)
    cv.waitKey(0)
    cv.destroyAllWindows()
    """
    #Creating the expected number of masks by keeping the biggest bounding boxes
    coords.sort(key=lambda x:x[4], reverse=True)
    masks = []
    for i in range(np.shape(background_mask)[0]):
        if len(coords)>0:
            (x,y,w,h,size) = coords.pop(0)
            m = np.zeros((height,width),np.uint8)
            m[y:y + h, x:x + w] = 255
            masks.append(np.uint8(m))
        else:
            m = np.zeros((height,width),np.uint8)
            masks.append(np.uint8(m))
    
    #Matching text mask to background mask (only if the image contains 2 or more paintings)
    #NOT FINISHED!!!
    # if np.shape(background_mask)[0]>1:
    #     for i in range(np.shape(background_mask)[0]):
    #         moments = cv.moments(background_mask[i])
    #         cx = int(moments["m10"] / moments["m00"])
	#         cy = int(moments["m01"] / moments["m00"])
    #         for j 
    #         distance = math.sqrt(((cx-(x+w/2))**2)+((cy-(y+h/2))**2))
    """
    cv.imshow("mask", masks[0])
    cv.waitKey(0)
    cv.destroyAllWindows()
    """
    return masks
