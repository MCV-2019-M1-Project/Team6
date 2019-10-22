# IMPORTS #
import cv2 as cv
import glob
import os
import numpy as np

QUERY_SET = 'qsd2_w3'
qs_l = '../qs/' + QUERY_SET + '/*.jpg'
for f in sorted(glob.glob(qs_l)):
    # Read image
    name = os.path.splitext(os.path.split(f)[1])[0]
    im = cv.imread(f, cv.IMREAD_COLOR)
    img_cs =  cv.cvtColor(im, cv.COLOR_BGR2RGB)

    # Filter image with a median filter
    filtered_im = cv.medianBlur(img_cs, 3)
    image =  cv.cvtColor(filtered_im, cv.COLOR_RGB2BGR)
    cv.imwrite('try/' + name + 'prova.jpg', image)