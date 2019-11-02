import cv2 as cv
import numpy as np
from compute_mask import compute_mask
from text_removal_mask2 import find_text
from matplotlib import pyplot as plt
import glob
import os

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

# Read image and its DDBB correspondence
im = cv.imread('../qs/qsd1_w4/00016.jpg', cv.IMREAD_COLOR)
im_db = cv.imread('../database/bbdd_00026.jpg', cv.IMREAD_COLOR)

# Remove salt & pepper noise and convert image to grayscale
filtered_im_q = cv.medianBlur(im,3)
filtered_im_db = cv.medianBlur(im_db,3)

img_gray_q = cv.cvtColor(filtered_im_q,cv.COLOR_BGR2GRAY)
img_gray_db = cv.cvtColor(filtered_im_db,cv.COLOR_BGR2GRAY)

# Compute background and text masks
bg_mask_q,_ = compute_mask(im, "try_mask1", 'qsd1w4')
text_mask_q = find_text(img_gray_q, bg_mask_q, "try_text1")

# Call SIFT creator
sift = cv.xfeatures2d.SIFT_create()

# Compute SIFT keypoints and descriptor
kp1, des1 = compute_SIFT_kp_and_des(img_gray_q, bg_mask_q[0], text_mask_q[0], sift, 512)
kp2, des2 = compute_SIFT_kp_and_des(img_gray_db, None, None, sift, 512)

# Variable to store descriptors
descriptors=[]
descriptors.append(des1)
descriptors.append(des2)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(descriptors[0],descriptors[1],k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

# ratio test as per Lowe's paper
matches_good = 0

for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
        matches_good += 1

# Number of "true" matches
print(matches_good)

# Matches drawing
draw_params = dict(matchColor = (0,255,0),
            singlePointColor = (255,0,0),
            matchesMask = matchesMask,
            flags = cv.DrawMatchesFlags_DEFAULT)

# Plot matches
img3 = cv.drawMatchesKnn(cv.resize(img_gray_q, (512,512)),kp1,cv.resize(img_gray_db, (512,512)),kp2,matches,None,**draw_params)
plt.imshow(img3,)
plt.show()
