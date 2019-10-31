# IMPORTS #
import cv2 as cv
import glob
import os
import numpy as np
import pytesseract
import Levenshtein
import random

#Returns the predicted queries, one per each binary mask in input list masks
#The strings contain the closest author in the ddbb

def get_text(img_gray, masks, database_txt):
    found_text = []
    for i in range(np.shape(masks)[0]):
        # Apply the mask
        """
        indices = np.where(masks[i] != [0])
        if(indices[0].size != 0 and indices[1].size !=0):
            img_gray = img_gray[min(indices[0]):max(indices[0]),min(indices[1]):max(indices[1])] 
        """
        img_gray = cv.bitwise_and(img_gray, img_gray, mask=masks[i])

        # Binarize and compute text
        ret, binary_image = cv.threshold(img_gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        if(binary_image is not None):
            text = pytesseract.image_to_string(binary_image, config="-c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ- ")
            text = text.lower()

            if( text == '' ):
                # Text is empty, nothing to do
                found_text.append('#################')
            else :
                # Search for closest authors in the ddbb
                dists = []
                for db in database_txt:
                    dists.append(Levenshtein.distance(text, db))

                # Return the first 10 paintings by the closest author
                index = np.argmin(dists)
                found_text.append(database_txt[index])

        else:
            found_text.append('#################')

    final_authors = []
    for author in found_text:
        authors = []
        j = 0
        for db in database_txt:
            if(db == author):
                authors.append(j)
            j += 1
        random.shuffle(authors)
        final_authors.append(authors[:10])

    #print(found_text)
    return final_authors, found_text


