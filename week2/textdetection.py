"""
TEXT REMOVAL USING MORPHOLOGY
"""

# IMPORTS
import glob
import cv2 as cv
import numpy as np
import os
import pandas as pd

QUERY_SET='qsd1_w2'
images = glob.glob('./' + QUERY_SET + '/*.jpg')
print(str(QUERY_SET) + " has " + str(len(images)) + " images")

# =============================================================================
# img = cv.imread('qsd1_w2/00003.jpg', cv.IMREAD_COLOR)
# img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
# =============================================================================

# Structuring element
strel = np.ones((15,15), np.uint8)

for f in sorted(images):
    # Read image
    name = os.path.splitext(os.path.split(f)[1])[0]
    img = cv.imread(f, cv.IMREAD_COLOR)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Obtain image dimensions
    height,width = img_gray.shape[:2]
    
    # Bounds to decide whther opening or closing must be applied
    min_a = round(width/2)
    max_a = round(min_a + 2)
    
    for i in range(min_a, max_a):
        col = img_gray[:,i]
        values = pd.Series(col).value_counts().keys().tolist()
        counts = pd.Series(col).value_counts().tolist()
        
        if int(counts[0]) > 20:
            if int(values[0]) <= 200:
                opening = cv.morphologyEx(img, cv.MORPH_OPEN, strel)
                cv.imwrite('results/' + name + '_opening.png', opening)
            elif int(values[0]) > 200:
                closing = cv.morphologyEx(img, cv.MORPH_CLOSE, strel)
                cv.imwrite('results/' + name + '_closing.png', closing)

