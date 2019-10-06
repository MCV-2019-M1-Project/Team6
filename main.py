# Imports
import cv2 as cv
import numpy as np
import glob
import pickle
import ml_metrics
import math
import os
from bitarray import bitarray

## PARAMETERS ##
NBINS = 64
COLORSPACE = cv.COLOR_BGR2Lab
#COLORSPACE = cv.COLOR_RGB2YUV

## FUNCTIONS ##

def compute_mask(img,name):
    
    # Computes channels in chosen color space
    c0 = img[:,:,0]
    c1 = img[:,:,1]
    c2 = img[:,:,2]
    
    # Height and width of channel (=image dims)
    height,width = c0.shape[:2]
        
    # Percentage defining number of pixels per every portion of the image
    percent_c0 = 0.02
    percent_c1 = 0.03
    percent_c2 = 0.02
    
    # Computes the amount of pixels per every channel
    aop_h_c0 = int(round(percent_c0 * height))
    aop_w_c0 = int(round(percent_c0 * width))
    
    aop_h_c1 = int(round(percent_c1 * height))
    aop_w_c1 = int(round(percent_c1 * width))
    
    aop_h_c2 = int(round(percent_c2 * height))
    aop_w_c2 = int(round(percent_c2 * width))
    
    # Defines image portions to get background pixels from
    portionc0_1 = c0[0:aop_h_c0, 0:width]
    portionc1_1 = c1[0:aop_h_c1, 0:width]
    portionc2_1 = c2[0:aop_h_c2, 0:width]
    
    portionc0_2 = c0[height - aop_h_c0:height, 0:width]
    portionc1_2 = c1[height - aop_h_c1:height, 0:width]
    portionc2_2 = c2[height - aop_h_c2:height, 0:width]
       
    # Computes minimum and max values per every portion and channel
    min_c0_1 = int(np.amin(portionc0_1))
    min_c1_1 = int(np.amin(portionc1_1))
    min_c2_1 = int(np.amin(portionc2_1))
    
    min_c0_2 = int(np.amin(portionc0_2))
    min_c1_2 = int(np.amin(portionc1_2))
    min_c2_2 = int(np.amin(portionc2_2))
    
    max_c0_1 = int(np.amax(portionc0_1))
    max_c1_1 = int(np.amax(portionc1_1))
    max_c2_1 = int(np.amax(portionc2_1))
    
    max_c0_2 = int(np.amax(portionc0_2))
    max_c1_2 = int(np.amax(portionc1_2))
    max_c2_2 = int(np.amax(portionc2_2))
    
    min_c0 = min(min_c0_1, min_c0_2)
    min_c1 = min(min_c1_1, min_c1_2)
    min_c2 = min(min_c2_1, min_c2_2)
    
    max_c0 = max(max_c0_1, max_c0_2)
    max_c1 = max(max_c1_1, max_c1_2)
    max_c2 = max(max_c2_1, max_c2_2)
    
    # Computes and saves the mask by thresholding every channel in the chosen color space
    mask = 255 - (cv.inRange(img,(min_c0, min_c1, min_c2),(max_c0, max_c1, max_c2)))
    #mask = cv.imread('qsd2_w1/'+ name + '.png', cv.IMREAD_COLOR)
    #mask = bitarray(mask.ravel())
    return mask

def extract_features(img,mask):

#Extracts feature vector from image. The returned vecor consists of the 1D histograms of
# each of the image channels concatenated.

    hists = []
    npx = img.shape[0] * img.shape[1]
    hist_1 = cv.calcHist([img], [0], mask, [NBINS], [0, 255]) / npx
    hist_2 = cv.calcHist([img], [1], mask, [NBINS], [0, 255]) / npx
    hist_3 = cv.calcHist([img], [2], mask, [NBINS], [0, 255]) / npx
    hists = np.concatenate((hist_1, hist_2, hist_3))
    return hists

def search(queries, database, distance):

# For each of the queries, searches for the 10  most similar images in the database. The
#decision is based on the feature vectors and a distance or similarity measure (Euclidean
# distance and Hellinger Kernel similarity. Returns a 2D array containing the results of
#the search for each of the queries.

    final_ranking = np.zeros((len(queries), 10), dtype=float)
    if(distance == "euclidean"):
        for i in range(0, len(queries)):
            ranking = np.ones((10, 2), dtype=float) * 3
            for j in range(0, len(database)):
                # Compute the distance metric
                dist = sum(pow(abs(database[j] - queries[i]), 2))
                # Check the ranking and update it
                if (dist < max(ranking[:, 1])):
                    # Add the distance and the id to the db
                    idx = np.argmax(ranking[:, 1])
                    ranking[idx, 0] = j
                    ranking[idx, 1] = dist
            # Store the closest K images
            for j in range(0, 10):
                idx = np.argmin(ranking[:, 1])
                final_ranking[i, j] = ranking[idx, 0]
                ranking[idx, :] = [3, 3]
    if(distance == "hellinger"):
        for i in range(0, len(queries)):
            ranking = np.zeros((10, 2), dtype=float)
            for j in range(0, len(database)):
                # Compute the distance metric
                dist = np.sum(np.sqrt(np.multiply(np.array(database[j]),np.array(queries[i]))))
                # Check the ranking and update it
                if (dist > min(ranking[:, 1])):
                    # Add the distance and the id to the db
                    idx = np.argmin(ranking[:, 1])
                    ranking[idx, 0] = j
                    ranking[idx, 1] = dist
            # Store the closest K images
            for j in range(0, 10):
                idx = np.argmax(ranking[:, 1])
                final_ranking[i, j] = ranking[idx, 0]
                ranking[idx, :] = [0, 0]
    return final_ranking

## READ THE DB AND STORE THE FEATURES ##


def main():
    database = []
    for f in sorted(glob.glob('./database/*.jpg')):
        img = cv.imread(f, cv.IMREAD_COLOR)
        img = cv.cvtColor(img, COLORSPACE)
        database.append(extract_features(img,None))
    print('Database has ' + str(len(database)) + ' images')

    queries = []
    
    # Change to switch datasets
    qs = 'qsd1_w1'
    qs_l = './' + qs + '/*.jpg'
    for f in sorted(glob.glob(qs_l)):
        name = os.path.splitext(os.path.split(f)[1])[0]
        img = cv.imread(f, cv.IMREAD_COLOR)
        img = cv.cvtColor(img, COLORSPACE)
        if qs == 'qsd1_w1':
            mask = None
        elif qs == 'qsd2_w1':
            mask = compute_mask(img,name)
            
        queries.append(extract_features(img,mask))

    print('Query set has ' + str(len(queries)) + ' images')

    gt = pickle.load(open('./' + qs + '/gt_corresps.pkl','rb'))

    ## SEARCH FOR THE QUERIES IN THE DB ##
    final_ranking = search(queries, database, "euclidean")

    ## EVALUATION USING MAP@K ##
        
    mapk_ = ml_metrics.mapk(gt,final_ranking.tolist(),10)
    print('MAP@K = '+ str(mapk_))

    ## WRITE OUTPUT FILES ##
    pickle.dump(final_ranking.tolist(), open('./' + qs + '/actual_corresps.pkl','wb'))

if __name__== "__main__":
  main()