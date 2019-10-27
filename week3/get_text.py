# IMPORTS #
import cv2 as cv
import glob
import os
import numpy as np
import pytesseract
import Levenshtein

#Returns a list of strings, one per each binary mask in input list masks
#The strings contain the closest author in the ddbb

def get_text(img_gray, masks, database_txt):
    found_text = []
    for i in range(np.shape(masks)[0]):
        # Apply the mask
        indices = np.where(masks[i] != [0])
        if(indices[0].size != 0 and indices[1].size !=0):
            img_gray = img_gray[min(indices[0]):max(indices[0]),min(indices[1]):max(indices[1])] 

        # Binarize and compute text
        ret, binary_image = cv.threshold(img_gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        text = pytesseract.image_to_string(binary_image, config="-c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ- ")
        text = text.lower()

        # Search for closest authors in the ddbb
        dists = []
        for db in database_txt:
            dists.append(Levenshtein.distance(text, db))

        # Return closest author
        index = np.argmin(dists)
        found_text.append(database_txt[index])
        print(text)
        print(database_txt[index])

    #print(found_text)
    return found_text


