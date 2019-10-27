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
from compute_mask_old import compute_mask_old
from text_removal_mask import text_removal_mask
from text_removal_mask2 import find_text
from search_queries import search
from compute_lbp import compute_lbp
from compute_hog import compute_hog
from get_text import get_text

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
        #descriptor = compute_lbp(img_gray, None, 8, 16, 8, 2, 'uniform')
        #descriptor = extract_features(img, None, NBINS, DIVISIONS)
        descriptor = compute_hog(img, 2)
        print(np.shape(descriptor))
        
        # Store the descriptor
        database.append(descriptor)
        print(str(i))
        i+=1
    print('Image database read!')

    # Read the text database 
    database_txt = []
    for f in sorted(glob.glob('../database_text/*.txt')):
        with open(f, encoding="ISO-8859-1") as textfile:
            line = str(textfile.readline())
            line = line.strip("'()")
            line = line.split(', ')
            author = line[0].strip("'")
            if author == '':
                author = '#################'
            database_txt.append( author.lower() )

    print('Text database read!')
    print('Database has ' + str(len(database)) + ' images')
    print(database_txt)

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
    i = 0
    qst_txt = []
    for f in sorted(glob.glob(qs_l)):
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
        if QUERY_SET == 'qsd1_w2' or QUERY_SET == 'qsd1_w3' or QUERY_SET == 'qst1_w3':
            bg_mask = None

        # BACKGROUND REMOVAL
        elif QUERY_SET == 'qsd2_w2' or QUERY_SET == 'qsd2_w3' or QUERY_SET == 'qst2_w3':
            bg_mask, eval_metrics = compute_mask(img,name,QUERY_SET)

            if eval_metrics is not None:
                precision[i] = eval_metrics[0]
                recall[i] = eval_metrics[3]
                fscore[i] = eval_metrics[4]
                i += 1

        # TEXT REMOVAL
        # Use the mask created (image without background) to indicate search text
        #mask, pred_coords = text_removal_mask(img_gray, name, kernel, post_kernel, num_cols, coords, bg_mask, QUERY_SET)
        
        if bg_mask is not None:
            mask = find_text(img_gray, bg_mask, name)
        else:
            bg_mask = [np.ones((img_gray.shape[0],img_gray.shape[1]))]
            mask = find_text(img_gray, bg_mask, name)
        
        #mask = bg_mask # No text removal mask

        #TEXT DETECTION
        qst_txt.append(get_text(img_gray, mask, database_txt))


        # Iterate the masks (1 or 2 according to the images)
        length = np.shape(mask)[0]
        print(length)
        if length > 2:
            length = 1
            mask = [mask]
        
        pre_list = []
        for m in range(length):
            # Use either one or the other mask
            prod = cv.bitwise_not(mask[m]) * bg_mask[m]
            prod = prod.astype(np.uint8)

            # Extract the features
            #descriptor = extract_features(img, prod, NBINS, DIVISIONS)
            #descriptor = compute_lbp(img_gray, prod, 8, 16, 8, 2, 'uniform')
            descriptor = compute_hog(img, 2)
            print(np.shape(descriptor))

            # Search the query in the DB
            rank = search([descriptor], database, DIST_METRIC, K)
            print(rank)
            pre_list.append(rank)

        final_ranking.append(pre_list)

    # Print the final ranking
    print('FINAL RANKING:')
    print(len(final_ranking))
    print(final_ranking)

    print('AUTHORS:')
    print(qst_txt)
            

    # Print the evaluation metrics
    if QUERY_SET == 'qsd1_w2' or QUERY_SET == 'qsd2_w2' or QUERY_SET == 'qsd1_w3' or QUERY_SET == 'qsd2_w3':

        print('Query set has ' + str(nqueries) + ' images')
        print('Precision: ' + str(np.mean(precision)))
        print('Recall: ' + str(np.mean(recall)))
        print('F-measure: ' + str(np.mean(fscore)))

        gt = pickle.load(open('../qs/' + QUERY_SET + '/gt_corresps.pkl','rb'))

        mapk_ = np.mean([ml_metrics.mapk([a],p,K) for a,p in zip(gt, final_ranking)])
        #mapk_ = ml_metrics.mapk(gt, final_ranking, K)
        print('MAP@K = '+ str(mapk_))

        if qst_txt != []:
            mapk_ = np.mean([ml_metrics.mapk([a],p,K) for a,p in zip(gt, qst_txt)])
            #mapk_ = ml_metrics.mapk(gt, final_ranking, K)
            print('MAP@K text = '+ str(mapk_))


    ## WRITE OUTPUT FILES ##
    pickle.dump(final_ranking, open('../qs/' + QUERY_SET + '/actual_corresps.pkl','wb'))

if __name__== "__main__":
    main()