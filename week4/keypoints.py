import cv2 as cv
import numpy as np
from compute_mask import compute_mask
from text_removal_mask2 import find_text
from matplotlib import pyplot as plt
import glob
import os
import pickle
import ml_metrics

def compute_SIFT_kp_and_des(img, bg_mask, text_mask, sift, size):

    # Resize to speed up execution
    img = cv.resize(img, (size,size))

    if bg_mask is not None:
        bg_mask = cv.resize(bg_mask, (size, size))
        text_mask = cv.resize(text_mask, (size, size))
        prod = cv.bitwise_not(text_mask) * bg_mask
        prod = prod.astype(np.uint8)
        kp, des = sift.detectAndCompute(img, prod)
    else:
        kp, des = sift.detectAndCompute(img, None)

    return kp, des

def SIFT_descriptor(im_gray, bg_mask, text_mask, sift, size, pre_list, flann):
    # Compute SIFT keypoints and descriptors
    kp, des_q = compute_SIFT_kp_and_des(im_gray, bg_mask, text_mask, sift, size)
    """
    cv.imshow("mask", bg_mask[m])
    cv.waitKey(0)
    cv.destroyAllWindows()

    cv.imshow("mask", text_mask[m])
    cv.waitKey(0)
    cv.destroyAllWindows()
    """

    matches_final = np.zeros(279)
    h = 0

    for f in sorted(glob.glob('../database/*.jpg')):
        """
        im_db = cv.imread(f, cv.IMREAD_COLOR)
        im_db = cv.resize(im_db, (512,512))
        im_db = cv.medianBlur(im_db, 3)
        img_gray_db = cv.cvtColor(im_db,cv.COLOR_BGR2GRAY)
        """

        matches = flann.knnMatch(descriptors_db[h], des_q, k=2)

        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for l in range(len(matches))]

        # ratio test as per Lowe's paper
        matches_good = 0

        for k,(m,n) in enumerate(matches):
            if m.distance < 0.55*n.distance:
                matchesMask[k]=[1,0]
                matches_good += 1

        # Number of "true" matches
        matches_final[h] = matches_good
        h+=1

    #pre_list.append([np.argmax(matches_final)])
    # Get K paintings from the database with more matches
    topK_matches = (-matches_final).argsort()[:K]
    pre_list.append(topK_matches.tolist())

    print("Query: " + str(q))
    print("Most similar image: " + str(np.argmax(matches_final)))
    print("Number of matches: " + str(np.amax(matches_final)))

    return pre_list

############### SIFT SETTINGS ###################
# Create SIFT object
sift = cv.xfeatures2d.SIFT_create()

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params,search_params)
######################################################

QUERY_SET = 'qsd1_w4'
qs_l = '../qs/' + QUERY_SET + '/*.jpg'

COLORSPACE = cv.COLOR_BGR2HSV
K = 5
size = 128

descriptors_db = []
keypoints_db = []
final_ranking = []

i = 0
for f in sorted(glob.glob('../database/*.jpg')):
        # Read image
        img = cv.imread(f, cv.IMREAD_COLOR)
        img = cv.medianBlur(img, 3)
        img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        kps, desc = compute_SIFT_kp_and_des(img_gray, None, None, sift, size)
        descriptors_db.append(desc)
        keypoints_db.append(kps)
        i+=1

q = 0

for f in sorted(glob.glob(qs_l)):
    # Read image 
    name = os.path.splitext(os.path.split(f)[1])[0]
    im = cv.imread(f, cv.IMREAD_COLOR)
    im = cv.medianBlur(im, 3)
    im_gray = cv.cvtColor(im,cv.COLOR_BGR2GRAY)
    im = cv.cvtColor(im, COLORSPACE)

     
     # Compute background and text masks
    bg_mask,_ = compute_mask(im, "prova" + name, 'qsd1w4')
    text_mask = find_text(im_gray, bg_mask, "provatext" + name)

    # Check whether the image contains two paintings
    length = np.shape(bg_mask)[0]

    if length > 2:
        length = 1
        bg_mask = [bg_mask]
        text_mask = [text_mask]

    pre_list = []
    for m in range(length):

        pre_list = SIFT_descriptor(im_gray, bg_mask[m], text_mask[m], sift, size, pre_list, flann)

    final_ranking.append(pre_list)
    q+=1

print(final_ranking)
gt = pickle.load(open('../qs/' + QUERY_SET + '/gt_corresps.pkl','rb'))
mapk_ = np.mean([ml_metrics.mapk([a],p,K) for a,p in zip(gt, final_ranking)])
print('MAP@K = '+ str(mapk_))