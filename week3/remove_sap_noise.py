# IMPORTS #
import cv2 as cv
import glob
import os

QUERY_SET = 'qsd2_w3'
qs_l = '../qs/' + QUERY_SET + '/*.jpg'
for f in sorted(glob.glob(qs_l)):
    name = os.path.splitext(os.path.split(f)[1])[0]
    im = cv.imread(f, cv.IMREAD_COLOR)
    #cv.imshow("image", im)
    #cv.waitKey(0)
    #cv.destroyAllWindows()

    median = cv.medianBlur(im, 3)
    gauss = cv.GaussianBlur(im, (7,7), 0)

    cv.imwrite('try/' + name + 'prova_median.jpg', median)
    cv.imwrite('try/' + name + 'prova_gaussian.jpg', gauss)

