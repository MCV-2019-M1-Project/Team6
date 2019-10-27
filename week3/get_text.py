# IMPORTS #
import cv2 as cv
import glob
import os
import numpy as np
import pytesseract

#Returns a list of strings, one per each binary mask in input list masks

def get_text(img_gray, masks)
    found_text = []
    for i in range(np.shape(masks)[0])
        if np.sum(masks[i]) == 0:
            masks[i] = np.ones((img_gray.shape[0],img_gray.shape[1]))  
        indices = np.where(masks[i] != [0])
        if(indices[0].size != 0 and indices[1].size !=0):
        cropped_image = img_gray[min(indices[0]):max(indices[0]),min(indices[1]):max(indices[1])] 
        ret,binary_image = cv.threshold(cropped_image,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        text = pytesseract.image_to_string(binary_image, config="-c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ- ")
        found_text.append(text)
    print(found_text)
    dist = []
    return found_text


