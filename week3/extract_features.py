import cv2 as cv
import numpy as np
import math

def extract_features(img,mask):

# Extracts feature vector from image. Returns a flat list consisting of the obtained features
    
    # Mask preprocessing
    if mask is not None:
        indices = np.where(mask != [0])
        if(indices[0].size != 0 and indices[1].size !=0):
            img = img[min(indices[0]):max(indices[0]),min(indices[1]):max(indices[1])]
            mask = mask[min(indices[0]):max(indices[0]),min(indices[1]):max(indices[1])]
    
    # Level 0 histograms:
    hist_img = []
    npx = img.shape[0]*img.shape[1]
    hist_1 = cv.calcHist([img],[0],mask,[NBINS],[0,256])/npx 
    hist_2 = cv.calcHist([img],[1],mask,[NBINS],[0,256])/npx
    hist_3 = cv.calcHist([img],[2],mask,[NBINS],[0,256])/npx
    hists = np.concatenate((hist_1,hist_2,hist_3))
    hist_img.append(hists)

    
    # Multilevel histograms
    for i in range(0,DIVISIONS):
        for j in range(0,DIVISIONS):
            # Compute the normalized histograms
            subimg = img[i*round(img.shape[0]/DIVISIONS):(i+1)*round(img.shape[0]/DIVISIONS)-1, 
                         j*round(img.shape[1]/DIVISIONS):(j+1)*round(img.shape[1]/DIVISIONS)-1]
            if mask is not None :
                submask = mask[i*round(img.shape[0]/DIVISIONS):(i+1)*round(img.shape[0]/DIVISIONS)-1, 
                            j*round(img.shape[1]/DIVISIONS):(j+1)*round(img.shape[1]/DIVISIONS)-1]
            else :
                submask = None
            npx = subimg.shape[0]*subimg.shape[1]
            hist_1 = cv.calcHist([subimg],[0],submask,[NBINS],[0,256])/npx 
            hist_2 = cv.calcHist([subimg],[1],submask,[NBINS],[0,256])/npx
            hist_3 = cv.calcHist([subimg],[2],submask,[NBINS],[0,256])/npx
            hists = np.concatenate((hist_1,hist_2,hist_3))
            hist_img.append(hists)

    # Flatten the obtained lists
    flat_list = []
    for sublist in hist_img:
        for item in sublist:
            flat_list.append(item)
    return flat_list