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
from compute_dct import compute_dct
from get_text import get_text
from compute_SIFT import compute_SIFT
from search_matches import search_matches_FLANN
from search_matches import search_matches_ORBS

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
SIZE = 512
SEARCH_METHOD = 2           # 1 for normal descriptors (distance) and 2 for keypoints matches (FLANN)
TEXT_DESCRIPTOR = False

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
        #descriptor_1 = compute_lbp(img_gray, None, 8, 16, 8, 2, 'uniform')
        #descriptor_2 = extract_features(img, None, NBINS, DIVISIONS)
        #descriptor_3 = compute_hog(img, None, 2)
        #descriptor_4 = compute_dct(img_gray, 8, 64, 128)
        descriptor_5 = compute_SIFT(img_gray, None, SIZE)

        descriptor = descriptor_5

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
    nqueries = 50
    precision = np.zeros(nqueries)
    recall = np.zeros(nqueries)
    fscore = np.zeros(nqueries)
    iou = np.zeros(nqueries)

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
        im = cv.medianBlur(im, 3)
        
        # Color conversions
        img = cv.cvtColor(im, COLORSPACE)
        img_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

        bg_mask = None
        # NO BACKGROUND
        if QUERY_SET == 'qsd1_w2' or QUERY_SET == 'qsd1_w3' or QUERY_SET == 'qst1_w3':
            bg_mask = None

        # BACKGROUND REMOVAL
        elif QUERY_SET == 'qsd2_w2' or QUERY_SET == 'qsd2_w3' or QUERY_SET == 'qst2_w3' or QUERY_SET == 'qsd1_w4' or QUERY_SET == 'qst1_w4' or QUERY_SET == 'qs_guasa':
            bg_mask, eval_metrics = compute_mask(img,name,QUERY_SET)

            if eval_metrics is not None:
                precision[i] = eval_metrics[0]
                recall[i] = eval_metrics[3]
                fscore[i] = eval_metrics[4]
        

        # TEXT REMOVAL
        # Use the mask created (image without background) to indicate search text
        #mask, pred_coords = text_removal_mask(img_gray, name, kernel, post_kernel, num_cols, coords, bg_mask, QUERY_SET)
        
        if bg_mask is not None:
            mask = find_text(img_gray, bg_mask, name, 'remove')
            mask_find = find_text(img_gray, bg_mask, name, 'find')
        else:
            bg_mask = [np.ones((img_gray.shape[0],img_gray.shape[1]))]
            mask = find_text(img_gray, bg_mask, name, 'remove')
            mask_find = find_text(img_gray, bg_mask, name, 'find')
        
        #mask = bg_mask # No text removal mask

        #TEXT DETECTION
        prob_paintings, found_text = get_text(img_gray, mask_find, database_txt)
        #print('Prob paintings:')
        #print(prob_paintings)
        #qst_txt.append( final_authors )
        # print(found_text[0])
        # file_auth =  open('../qs/' + QUERY_SET + '/author_' + str(i) + '.txt','w') 
        # print(len(found_text))
        # if len(found_text) == 1:
        #     print(found_text[0], file = file_auth)
        # else:
        #     print(found_text[1])
        #     print(found_text[1], file = file_auth)
        #     print(found_text[0], file = file_auth)
        
        # file_auth.close()

        # Iterate the masks (1 or 2 according to the images)
        length = np.shape(mask)[0]
        if length > 2:
            length = 1
            mask = [mask]
        
        picture_rank = []
        for m in range(length):
            # Use either one or the other mask
            prod = cv.bitwise_not(mask[m]) * bg_mask[m]
            prod = prod.astype(np.uint8)

            _, prod = cv.threshold(prod,0,255,cv.THRESH_BINARY)
            cv.imwrite('masks/' + name + '_' + str(m) + '_text.png', cv.bitwise_not(mask[m]))
            cv.imwrite('masks/' + name + '_' + str(m) + '_bg.png', bg_mask[m])
            cv.imwrite('masks/' + name + '_' + str(m) + '_merged.png', prod)
            
            """
            img_show = cv.resize(cv.bitwise_and(img_gray, prod), (512,512))
            cv.imshow('window',  img_show)
            cv.waitKey()
            cv.destroyAllWindows()
            """

            # Extract the features
            #descriptor_1 = compute_lbp(img_gray, prod, 8, 16, 8, 2, 'uniform')
            #descriptor_2 = extract_features(img, prod, NBINS, DIVISIONS)
            #descriptor_3 = compute_hog(img, prod, 2)
            #descriptor_4 = compute_dct(img_gray, 8, 64, 128)
            descriptor_5 = compute_SIFT(img_gray, prod, SIZE)

            descriptor = descriptor_5
            
            # Reduce database if text descriptor is used
            if len(prob_paintings[m]) > 0 and TEXT_DESCRIPTOR == True:
                reduced_db = []
                for ind in range(len(prob_paintings[m])):
                    reduced_db.append(database[prob_paintings[m][ind]])
            else:
                reduced_db = database

            # Search the query in the DB according to the descriptor
            if SEARCH_METHOD == 1:
                painting_rank = search([descriptor], reduced_db, DIST_METRIC, K)
            elif SEARCH_METHOD == 2:
                painting_rank = search_matches_FLANN(descriptor, reduced_db, K)
            
            if len(prob_paintings[m])>0 and TEXT_DESCRIPTOR == True:
                rank = []
                for i in painting_rank:
                    rank.append(prob_paintings[m][i])
                painting_rank = rank
            print(painting_rank)
            picture_rank.append(painting_rank)

            """
            print("picture_rank:")
            print(picture_rank)
            picture_rank.append(painting_rank)
            """

        final_ranking.append(picture_rank)
        i += 1

    # Print the final ranking
    print('FINAL RANKING:')
    print(len(final_ranking))
    print(final_ranking)

    # print('AUTHORS:')
    # print(qst_txt)
            

    # Print the evaluation metrics
    if QUERY_SET == 'qsd1_w2' or QUERY_SET == 'qsd2_w2' or QUERY_SET == 'qsd1_w3' or QUERY_SET == 'qsd2_w3' or QUERY_SET == 'qsd1_w4':

        print('Query set has ' + str(nqueries) + ' images')
        print('Precision: ' + str(np.mean(precision)))
        print('Recall: ' + str(np.mean(recall)))
        print('F-measure: ' + str(np.mean(fscore)))

        gt = pickle.load(open('../qs/' + QUERY_SET + '/gt_corresps.pkl','rb'))

        mapk_ = np.mean([ml_metrics.mapk([a],p,K) for a,p in zip(gt, final_ranking)])
        #mapk_ = ml_metrics.mapk(gt, final_ranking, K)
        print('MAP@K = '+ str(mapk_))

        # if qst_txt != []:
        #     mapk_ = np.mean([ml_metrics.mapk([a],p,K) for a,p in zip(gt, qst_txt)])
        #     #mapk_ = ml_metrics.mapk(gt, final_ranking, K)
        #     print('MAP@K text = '+ str(mapk_))

        if QUERY_SET == 'qsd1_w4' or QUERY_SET == 'qs_guasa':
            
            for i in range(0,len(gt)):
                for j in range(0,len(gt[i])):
                    if(gt[i][j] == -1):
                        gt[i][j] = 1
                    else:
                        gt[i][j] = 0
            pred = []
            for i in range(0,len(final_ranking)):
                
                if( len(final_ranking[i]) == 1 ):
                    if(final_ranking[i][0][0] == -1):
                        pred.append([1])
                    else:
                        pred.append([0])
                else:
                    p=[]
                    for j in range(0, len(final_ranking[i])):
                        if(final_ranking[i][j][0] == -1):
                            p.append(1)
                        else:
                            p.append(0)
                    pred.append([p[0], p[1]])

            print(pred)

            gt = (np.hstack(gt))
            pred = (np.hstack(pred))

            print(gt)
            print(pred)

            pixelTP = np.sum(gt & pred)
            pixelFP = np.sum(gt & (pred==0))
            pixelFN = np.sum((gt==0) & pred)
            pixelTN = np.sum((gt==0) & (pred==0))

            pixel_precision = 0
            pixel_sensitivity = 0
            F1 = 0
            if (pixelTP+pixelFP) != 0:
                pixel_precision   = float(pixelTP) / float(pixelTP+pixelFP)
            if (pixelTP+pixelFN) != 0:
                pixel_sensitivity = float(pixelTP) / float(pixelTP+pixelFN)

            if (pixel_precision+pixel_sensitivity != 0):
                F1 = 2*pixel_precision*pixel_sensitivity/(pixel_precision+pixel_sensitivity)
            print('F1 score:')
            print(F1)


    ## WRITE OUTPUT FILES ##
    pickle.dump(final_ranking, open('../qs/' + QUERY_SET + '/actual_corresps.pkl','wb'))

if __name__== "__main__":
    main()