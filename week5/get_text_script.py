# IMPORTS #
import cv2 as cv
import glob
import os
import numpy as np
import pytesseract
import Levenshtein
from matplotlib import pyplot as plt
from text_removal_mask2 import find_text

QUERY_SET = 'qsd1_w2'

#Read text ground truth
gt = []
for f in sorted(glob.glob('../qs/' + QUERY_SET +'/*.txt')):
    # painters = []
    with open(f) as textfile:
        line = str(textfile.readline())
        line = line.strip("'()")
        line = line.split(', ')
        #print(line)
        gt.append(line[0].strip("'"))
print(gt)

#Loop through images
found_text = []
for f in sorted(glob.glob('../qs/' + QUERY_SET +'/*.jpg')):
    name = os.path.splitext(os.path.split(f)[1])[0]
    img = cv.imread(f, cv.IMREAD_COLOR)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.cvtColor(img, cv.COLOR_BGR2Lab)
    bg_mask = [np.zeros((img_gray.shape[0],img_gray.shape[1]))]
    masks = find_text(img_gray, bg_mask, name)
    if np.sum(masks[0]) == 0:
        masks[0] = np.ones((img_gray.shape[0],img_gray.shape[1]))  
    indices = np.where(masks[0] != [0])
    if(indices[0].size != 0 and indices[1].size !=0):
        cropped_image = img_gray[min(indices[0]):max(indices[0]),min(indices[1]):max(indices[1])] 
    ret,binary_image = cv.threshold(cropped_image,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

    # cv.imshow("binary_image", binary_image)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    text = pytesseract.image_to_string(binary_image, config="-c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ- ")
    print(text)
    found_text.append(text)
print(found_text)
dist = []
for i in range(len(gt)):
    #THIS IS THE DISTANCE MEASURE TO BE USED WHEN COMPARING STRINGS
    dist.append(Levenshtein.distance(found_text[i], gt[i]))
print(dist)
print(sum(dist)/len(dist))



