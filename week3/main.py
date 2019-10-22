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
from search import search


## PARAMETERS ##
with open("config.yml", 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)
if cfg['colorspace'] == 'HSV' :
    COLORSPACE = cv.COLOR_BGR2HSV
elif cfg['colorspace'] == 'YUV' :
    COLORSPACE = cv.COLOR_BGR2YUV
elif cfg['colorspace'] == 'LAB' :
    COLORSPACE = cv.COLOR_BGR2Lab

NBINS = cfg['nbins']        # Number of bins (from 0 to 255)
DIVISIONS = cfg['divs']     # Number of divisions per dimension [2,4,8,...]
DIST_METRIC= cfg['dist']    #'euclidean' 'chisq' or 'hellinger'
BG_REMOVAL = cfg['bgrm']    # 1, 2 or 3 bg removal method
QUERY_SET= cfg['queryset']  # Input query set
K = 10                      # find K closest images

def main():

    # Read the image database
    database = []
    for f in sorted(glob.glob('../database/*.jpg')):
        img = cv.imread(f, cv.IMREAD_COLOR)
        img = cv.cvtColor(img, COLORSPACE)
        database.append(extract_features(img, None, NBINS, DIVISIONS))
    print('Image database read!')

    # Read the text database
    database_txt = []
    for f in sorted(glob.glob('../database_text/*.txt')):
        with open(f, encoding = "ISO-8859-1") as fp:
            line = fp.readline()
            database_txt.append(str(line))
    print('Text database read!')
    print('Database has ' + str(len(database)) + ' images')
    print(database_txt)

    # Text removal variables
    # Structuring element
    kernel = np.ones((15,15), np.uint8)
    
    # Structuring element used after text removal
    post_kernel = np.ones((20,20),np.uint8)
    
    # Number of columns considered from the center of the image towards the right
    num_cols = 6

    # Evaluation metrics storing arrays
    nqueries = len(glob.glob('../qs/' + QUERY_SET + '/*.jpg'))
    precision = np.zeros(nqueries)
    recall = np.zeros(nqueries)
    fscore = np.zeros(nqueries)
    iou = np.zeros(nqueries)

    # Read and process the queries
    i = 0
    final_ranking = []
    for f in sorted(glob.glob('../qs/' + QUERY_SET + '/*.jpg')):
        # Read image and color conversions
        name = os.path.splitext(os.path.split(f)[1])[0]
        im = cv.imread(f, cv.IMREAD_COLOR)
        img = cv.cvtColor(im, COLORSPACE)
        img_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

        # NO BACKGROUND
        if QUERY_SET == 'qsd1_w3' or QUERY_SET == 'qst1_w3':
            bg_mask = None

        # BACKGROUND REMOVAL
        elif QUERY_SET == 'qsd2_w3' or QUERY_SET == 'qst2_w3':
            bg_mask, eval_metrics = compute_mask(img_gray,name)

            if eval_metrics is not None:
                precision[i] = eval_metrics[0]
                recall[i] = eval_metrics[3]
                fscore[i] = eval_metrics[4]

        # TEXT REMOVAL
        # Use the mask created (image without background) to indicate search text
        mask, pred_coords = text_removal_mask(img_gray, name, kernel, post_kernel, num_cols, bg_mask)
        
        # Iterate the masks (1 or 2 according to the images)
        length = np.shape(mask)[0]
        if length > 2:
            length = 1
            mask = [mask]

        pre_list = []
        for m in range(length):
            listilla = search([extract_features(img,mask[m].astype(np.uint8), NBINS, DIVISIONS)], database, DIST_METRIC, K)
            pre_list.append(listilla.tolist())
        final_ranking.append(pre_list)
        i += 1

    # Print the final ranking
    print('FINAL RANKING:')
    print(final_ranking)

    # Print the evaluation metrics
    if QUERY_SET == 'qsd1_w3' or QUERY_SET == 'qsd2_w3':

        print('Query set has ' + str(len(nqueries)) + ' images')
        print('Precision: ' + str(np.mean(precision)))
        print('Recall: ' + str(np.mean(recall)))
        print('F-measure: ' + str(np.mean(fscore)))

        gt = pickle.load(open('../qs/' + QUERY_SET + '/gt_corresps.pkl','rb'))
        mapk_ = ml_metrics.mapk(gt, final_ranking, K)
        print('MAP@K = '+ str(mapk_))

    ## WRITE OUTPUT FILES ##
    pickle.dump(final_ranking, open('../qs/' + QUERY_SET + '/actual_corresps.pkl','wb'))

if __name__== "__main__":
    main()