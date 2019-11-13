import cv2 as cv
import numpy as np
import math
import colorgram
import tamura

def art_features(img, image_name):

    img = cv.resize(img, (512,512))
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    height = img.shape[0]
    width = img.shape[1]
    npx = height*width
    
    #Mean saturation and brightness
    saturation = np.sum(img_hsv[:][:][2])/npx
    brightness = np.sum(img_hsv[:][:][2])/npx

    #Pleasure, arousal and dominance
    pleasure = 0.69*brightness +0.22*saturation 
    arousal = -0.31*brightness +0.60*saturation 
    dominance = 0.76*brightness +0.32*saturation

    #Mean hue and dispersion, with and without saturation
    a = 0
    a_s = 0
    b = 0
    b_s = 0
    for x in range(height):
        for y in range(width):
            a += math.cos(img_hsv[x][y][1])
            a_s += img_hsv[x][y][2]* math.cos(img_hsv[x][y][1])
            b += math.sin(img_hsv[x][y][1])
            b_s += img_hsv[x][y][2] * math.sin(img_hsv[x][y][1])
    
    mean_hue = math.atan2(b,a)
    hue_variance = 1 - math.sqrt(a*a+b*b)/npx
    mean_hue_saturation = math.atan2(b_s,a_s)
    hue_variance_saturation = 1 - math.sqrt(a_s*a_s+b_s*b_s)/(saturation*npx)

    #Colorfulness
    img_signature = np.zeros((64,4), np.float32)
    uniform_signature = np.zeros((64,4), np.float32)
    hist = cv.calcHist([img], [0, 1, 2], None, [4, 4, 4], [0, 255, 0, 255, 0, 255])/npx 
    i=0
    for x in range(0,4):
        for y in range(0,4):
            for z in range(0,4):
                img_signature[i][0] = hist[x][y][z]
                uniform_signature[i][0] = 1/64
                img_signature[i][1] = x
                uniform_signature[i][1] = x
                img_signature[i][2] = y
                uniform_signature[i][2] = y
                img_signature[i][3] = z
                uniform_signature[i][3] = z
                i+=1
    
    colorfulness = cv.EMD(uniform_signature, img_signature, cv.DIST_L2)[0]

    #Main colours
    main_colors = []
    colors = colorgram.extract(image_name, 3)
    for i in range(0,3):
        for j in range(0,3):
            main_colors.append(colors[i].hsl[j])    
        main_colors.append(colors[i].proportion)

    #Tamura texture features: coarseness, contrast and directionality
    coarseness = tamura.coarseness(img_gray, 3)
    contrast = tamura.contrast(img_gray)
    directionality = tamura.directionality(img_gray)

<<<<<<< HEAD
=======
    
>>>>>>> master
    #Colour features: saturation + brightness + pleasure + arousal + dominance + mean_hue + hue_variance + mean_hue_saturation + hue_variance_saturation + colorfulness + main_colors
    #Texture features: coarseness + contrast + directionality

image_name = '../database/bbdd_00241.jpg'
img = cv.imread(image_name, cv.IMREAD_COLOR)
art_features(img, image_name)