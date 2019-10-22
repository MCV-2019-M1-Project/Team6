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
#from matplotlib import pyplot as plt 
from evaluation_funcs import performance_accumulation_pixel
from evaluation_funcs import performance_evaluation_pixel
from bbox_iou import bbox_iou

## CUSTOM LIBS ## 
from extract_features import extract_features


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

    # Read and process the queries
    queries = []
    for f in sorted(glob.glob('../qs/' + QUERY_SET + '/*.jpg')):
        name = os.path.splitext(os.path.split(f)[1])[0]
        im = cv.imread(f, cv.IMREAD_COLOR)
        img = cv.cvtColor(im, COLORSPACE)
        
        # NO BACKGROUND
        if QUERY_SET == 'qsd1_w1' or QUERY_SET == 'qst1_w1' or QUERY_SET == 'qsd1_w2' or QUERY_SET == 'qst1_w2':
            bg_mask = None
        # BACKGROUND REMOVAL
        elif QUERY_SET == 'qsd2_w1' or QUERY_SET == 'qst2_w1' or QUERY_SET == 'qsd2_w2':
            if BG_REMOVAL==3:
                img_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
                bg_mask, eval_metrics = compute_mask(img_gray,name)
            else:
                bg_mask, eval_metrics = compute_mask(img,name)
            precision[i] = eval_metrics[0]
            recall[i] = eval_metrics[3]
            fscore[i] = eval_metrics[4]
        elif QUERY_SET == 'qst2_w2':
            if BG_REMOVAL==3:
                img_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
                bg_mask, eval_metrics = compute_mask(img_gray,name)
            else:
                bg_mask,_eval_metrics = compute_mask(img,name)


"""
    queries = []    
    qs_l = '../qs/' + QUERY_SET + '/*.jpg'
    
    # Evaluation metrics storing arrays
    precision = np.zeros(len(glob.glob(qs_l)))
    recall = np.zeros(len(glob.glob(qs_l)))
    fscore = np.zeros(len(glob.glob(qs_l)))
    iou = np.zeros(len(glob.glob(qs_l)))
    
    i=0
    
    # Text removal variables
    # Structuring element
    strel = np.ones((15,15), np.uint8)
    
    # Structuring element used after text removal
    strel_pd = np.ones((20,20),np.uint8)
    
    # Number of columns considered from the center of the image towards the right
    num_cols = 6
    
    # List to store detected bounding boxes coordinates
    coords = []
    final_ranking = []
    for f in sorted(glob.glob(qs_l)):
        name = os.path.splitext(os.path.split(f)[1])[0]
        im = cv.imread(f, cv.IMREAD_COLOR)
        img = cv.cvtColor(im, COLORSPACE)
        
        # NO BACKGROUND
        if QUERY_SET == 'qsd1_w1' or QUERY_SET == 'qst1_w1' or QUERY_SET == 'qsd1_w2' or QUERY_SET == 'qst1_w2':
            bg_mask = None
        # BACKGROUND REMOVAL
        elif QUERY_SET == 'qsd2_w1' or QUERY_SET == 'qst2_w1' or QUERY_SET == 'qsd2_w2':
            if BG_REMOVAL==3:
                img_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
                bg_mask, eval_metrics = compute_mask(img_gray,name)
            else:
                bg_mask, eval_metrics = compute_mask(img,name)
            precision[i] = eval_metrics[0]
            recall[i] = eval_metrics[3]
            fscore[i] = eval_metrics[4]
        elif QUERY_SET == 'qst2_w2':
            if BG_REMOVAL==3:
                img_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
                bg_mask, eval_metrics = compute_mask(img_gray,name)
            else:
                bg_mask,_eval_metrics = compute_mask(img,name)
        
        # TEXT REMOVAL
        if QUERY_SET == 'qsd1_w2' or QUERY_SET == 'qst1_w2' or QUERY_SET == 'qsd2_w2' or QUERY_SET == 'qst2_w2':
            img_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
            # Use the mask created (image without background) to indicate search text
            mask, pred_coords = text_removal_mask(img_gray, name, strel, strel_pd, num_cols, coords, bg_mask)
            #for m in range(np.shape(mask)[0]):
                #mask = mask[m].astype(np.uint8)
        else:
            mask = [bg_mask]

        i+=1

  
        # Iterate the masks (1 or 2 according to the images)
        query_data = []
        
        length = np.shape(mask)[0]
        if length > 2:
            length = 1
            mask = [mask]  

        pre_list = []
        for m in range(length):
            listilla = search([extract_features(img,mask[m].astype(np.uint8))], database, DIST_METRIC, K)
            pre_list.append(listilla.tolist())
        final_ranking.append(pre_list)

    print('FINAL RANKING:')
    print(final_ranking)

        #queries.append(extract_features(img,mask))

    if QUERY_SET == 'qsd2_w1':
        print('Query set has ' + str(len(queries)) + ' images')
        print('Precision: ' + str(np.mean(precision)))
        print('Recall: ' + str(np.mean(recall)))
        print('F-measure: ' + str(np.mean(fscore)))

    if (QUERY_SET == 'qsd1_w2' or QUERY_SET == 'qst1_w2'):
        realcoords = pickle.load(open(QUERY_SET + '/text_boxes.pkl','rb'))
        i = 0
        for i in range(0, len(realcoords)):
            predicted = coords[i][0]
            real = realcoords[i][0]
            iou[i] = bbox_iou(real, predicted)
            i += 1
        print('Mean IOU: ' + str(np.mean(iou)))
        print(predicted)

        ## WRITE PREDICTED BOUNDING BOXES ##
        pickle.dump(pred_coords, open('../qs/' + QUERY_SET + '/pred_bboxes.pkl','wb'))
    
    ## ADD QSD2_W2 AND QST2_W2

    ## SEARCH FOR THE QUERIES IN THE DB ##
    final_ranking = search(queries, database, DIST_METRIC, K)
    print('FINAL RANKING:')
    print(final_ranking)

    ## EVALUATION USING MAP@K ##
    if QUERY_SET == 'qsd1_w1' or QUERY_SET == 'qsd2_w1'  or QUERY_SET == 'qsd1_w2' or QUERY_SET == 'qsd2_w2':
        gt = pickle.load(open('../qs/' + QUERY_SET + '/gt_corresps.pkl','rb'))
        mapk_ = ml_metrics.mapk(gt,final_ranking.tolist(),K)
        print('MAP@K = '+ str(mapk_))

    ## WRITE OUTPUT FILES ##
    pickle.dump(final_ranking, open('../qs/' + QUERY_SET + '/actual_corresps.pkl','wb'))
"""
if __name__== "__main__":
    main()