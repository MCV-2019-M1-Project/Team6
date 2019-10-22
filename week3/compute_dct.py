# IMPORTS #
import cv2 as cv
import glob
import os
import numpy as np

im = cv.imread('../qs/qsd2_w3/00000.jpg', cv.IMREAD_COLOR)
img_gray =  cv.cvtColor(im, cv.COLOR_BGR2GRAY)

h, w = img_gray.shape[:2]

if (h % 2 == 1):
    h += 1

if (w % 2 == 1):
    w += 1

resized_img = cv.resize(img_gray, (w, h)) 

vis0 = np.zeros((h,w), np.float32)
vis0[:h, :w] = resized_img
vis1 = cv.dct(vis0)
img2 = cv.createMat(vis1.shape[0], vis1.shape[1], cv.CV_32FC3)
img3 = cv.cvtColor(cv.fromarray(vis1), img2, cv.CV_GRAY2BGR)

cv.imshow("image", img3)
cv.waitKey(0)
cv.destroyAllWindows()