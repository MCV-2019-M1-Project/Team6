# Imports
import cv2 as cv
import numpy as np
import glob
import pickle
import ml_metrics
import math

## PARAMETERS ##
NBINS = 64
COLORSPACE = cv.COLOR_BGR2Lab
#COLORSPACE = cv.COLOR_RGB2YUV

## FUNCTIONS ##

def extract_features(img):

#Extracts feature vector from image. The returned vecor consists of the 1D histograms of
# each of the image channels concatenated.

    hists = []
    npx = img.shape[0] * img.shape[1]
    hist_1 = cv.calcHist([img], [0], None, [NBINS], [0, 255]) / npx
    hist_2 = cv.calcHist([img], [1], None, [NBINS], [0, 255]) / npx
    hist_3 = cv.calcHist([img], [2], None, [NBINS], [0, 255]) / npx
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
        img=cv.cvtColor(img, COLORSPACE)
        database.append(extract_features(img))
    print('Database has ' + str(len(database)) + ' images')

    queries = []
    for f in sorted(glob.glob('./qsd1_w1/*.jpg')):
        img = cv.imread(f, cv.IMREAD_COLOR)
        img = cv.cvtColor(img, COLORSPACE)
        queries.append(extract_features(img))
    print('Query set has ' + str(len(queries)) + ' images')

    gt = pickle.load(open('./qsd1_w1/gt_corresps.pkl','rb'))

    ## SEARCH FOR THE QUERIES IN THE DB ##
    final_ranking = search(queries, database, "euclidean")

    ## EVALUATION USING MAP@K ##
    gt_list = []
    for i in range(0,len(gt)):
        gt_list.append([gt[i][0][1]])
        
    mapk_ = ml_metrics.mapk(gt_list,final_ranking.tolist(),10)
    print('MAP@K = '+ str(mapk_))

    ## WRITE OUTPUT FILES ##
    pickle.dump(final_ranking.tolist(), open('./qsd1_w1/actual_corresps.pkl','wb'))

if __name__== "__main__":
  main()