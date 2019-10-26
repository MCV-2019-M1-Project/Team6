#!/usr/bin/python3.5
## PYTHON LIBS ##
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

## CUSTOM LIBS ## 
from extract_features import extract_features
from compute_mask import compute_mask
from text_removal_mask import text_removal_mask
from search_queries import search
from compute_lbp import compute_lbp


## PARAMETERS ##
with open("config.yml", 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)
if cfg['colorspace'] == 'HSV' :
    COLORSPACE = cv.COLOR_BGR2HSV
elif cfg['colorspace'] == 'YUV' :
    COLORSPACE = cv.COLOR_BGR2YUV
elif cfg['colorspace'] == 'LAB' :
    COLORSPACE = cv.COLOR_BGR2Lab
elif cfg['colorspace'] == 'RGB' : 
    COLORSPACE = cv.COLOR_BGR2RGB 

NBINS = cfg['nbins']        # Number of bins (from 0 to 255)
DIVISIONS = cfg['divs']     # Number of divisions per dimension [2,4,8,...]
DIST_METRIC= cfg['dist']    #'euclidean' 'chisq' or 'hellinger'
BG_REMOVAL = cfg['bgrm']    # 1, 2 or 3 bg removal method
QUERY_SET= cfg['queryset']  # Input query set
K = 10                      # find K closest images

def main():

    # Read the image database
    database = []
    i = 0
    for f in sorted(glob.glob('../database/*.jpg')):
        # Read image
        img = cv.imread(f, cv.IMREAD_COLOR)

        # Apply median blur
        img = cv.medianBlur(img,3)

        # Colorspace changes
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = cv.cvtColor(img, COLORSPACE)

        # Compute descriptors
        descriptor = compute_lbp(img_gray, 8, 16, 8, 2, 'uniform')
        #descriptor = extract_features(img, None, NBINS, DIVISIONS)

        # Store the descriptor
        database.append(descriptor)
        print(str(i))
        i+=1
    print('Image database read!')

    # Read the text database - TODO: Correct encoding
    database_txt = []
    for f in sorted(glob.glob('../database_text/*.txt')):
        with open(f, encoding = "ISO-8859-1") as fp:
            line = fp.readline()
            database_txt.append(str(line))
    print('Text database read!')
    print('Database has ' + str(len(database)) + ' images')

    # Evaluation metrics storing arrays
    qs_l = '../qs/' + QUERY_SET + '/*.jpg'
    nqueries = 30
    precision = np.zeros(30)
    recall = np.zeros(30)
    fscore = np.zeros(30)
    iou = np.zeros(30)

    # Read and process the queries
    final_ranking = []
    coords = []
    for f in sorted(glob.glob(qs_l)):
        print('pew')
        # Read image 
        name = os.path.splitext(os.path.split(f)[1])[0]
        im = cv.imread(f, cv.IMREAD_COLOR)

        # Remove salt and pepper noise
        im = cv.medianBlur(im,3)
        
        # Color conversions
        img = cv.cvtColor(im, COLORSPACE)
        img_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

        bg_mask = None
        # NO BACKGROUND
        if QUERY_SET == 'qsd1_w3' or QUERY_SET == 'qst1_w3':
            bg_mask = None

        # BACKGROUND REMOVAL
        elif QUERY_SET == 'qsd2_w3' or QUERY_SET == 'qst2_w3':
            bg_mask, eval_metrics = compute_mask(img_gray,name,QUERY_SET)

            if eval_metrics is not None:
                precision[i] = eval_metrics[0]
                recall[i] = eval_metrics[3]
                fscore[i] = eval_metrics[4]

        # TEXT REMOVAL
        # Use the mask created (image without background) to indicate search text
        mask, pred_coords = text_removal_mask(img_gray, name, np.ones((15,15), np.uint8), np.ones((20,20),np.uint8), 6, coords, bg_mask, QUERY_SET)
        
        # Iterate the masks (1 or 2 according to the images)
        length = np.shape(mask)[0]
        if length > 2:
            length = 1
            mask = [mask]
        
        pre_list = []
        for m in range(length):
            # Compute the descriptor using the mask
            descriptor = compute_lbp(img_gray, 8, 16, 8, 2, 'uniform')
            #descriptor = extract_features(img,mask[m].astype(np.uint8), NBINS, DIVISIONS)

            # Search for the query in the DB
            rank = search([descriptor], database, DIST_METRIC, K)
            print(rank)
            pre_list.append(rank)

        final_ranking.append(pre_list)

    # Print the final ranking
    print('FINAL RANKING:')
    print(len(final_ranking))
    print(final_ranking)

    # Print the evaluation metrics
    if QUERY_SET == 'qsd1_w3' or QUERY_SET == 'qsd2_w3':

        print('Query set has ' + str(nqueries) + ' images')
        print('Precision: ' + str(np.mean(precision)))
        print('Recall: ' + str(np.mean(recall)))
        print('F-measure: ' + str(np.mean(fscore)))

        gt = pickle.load(open('../qs/' + QUERY_SET + '/gt_corresps.pkl','rb'))

        mapk_ = np.mean([ml_metrics.mapk([a],p,K) for a,p in zip(gt, final_ranking)])
        #mapk_ = ml_metrics.mapk(gt, final_ranking, K)
        print('MAP@K = '+ str(mapk_))

    ## WRITE OUTPUT FILES ##
    pickle.dump(final_ranking, open('../qs/' + QUERY_SET + '/actual_corresps.pkl','wb'))

if __name__== "__main__":
    main()