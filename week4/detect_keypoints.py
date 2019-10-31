import cv2 as cv
import numpy as np
from compute_mask import compute_mask
from text_removal_mask2 import find_text

# Read image and its DDBB correspondence
im = cv.imread('../qs/qsd1_w4/00004.jpg', cv.IMREAD_COLOR)
im_db = cv.imread('../database/bbdd_00210.jpg', cv.IMREAD_COLOR)

# Remove salt & pepper noise and convert image to grayscale
filtered_im = cv.medianBlur(im,3)
img_gray = cv.cvtColor(filtered_im,cv.COLOR_BGR2GRAY)

# Compute background and text masks
bg_mask,_ = compute_mask(im, "try_mask", 'qsd1w4')
text_mask = find_text(img_gray, bg_mask, "try_text")

# Compute final mask
prod = cv.bitwise_not(text_mask[0]) * bg_mask[0]
prod = prod.astype(np.uint8)

# Compute keypoints and descriptors using SIFT 
sift = cv.xfeatures2d.SIFT_create()
kp, descriptor = sift.detectAndCompute(img_gray, prod)

# Final image with drawn keypoints
img = cv.drawKeypoints(img_gray, kp, im)

# Show keypoints
cv.imshow("image", img)
cv.waitKey(0)
cv.destroyAllWindows()
