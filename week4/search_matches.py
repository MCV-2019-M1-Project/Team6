import cv2 as cv
import numpy as np

def search_matches_FLANN(queries, database, K):

    # Set up FLANN based matcher
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params,search_params)

    j=0
    matches_final = np.zeros(len(database))
    

    for i in range(len(database)):
        matches = flann.knnMatch(database[i], queries, k=2)
    
        #print(np.shape(database[i]))
        #print(np.shape(queries))

        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for l in range(len(matches))]

        matches_good = 0
        # ratio test as per Lowe's paper
        for k,(m,n) in enumerate(matches):
            if m.distance < 0.65*n.distance:
                matchesMask[k]=[1,0]
                matches_good += 1

        # Number of "true" matches
        matches_final[i] = matches_good
    
    #print(matches_final)
    topK_matches = (-matches_final).argsort()
    topK_dists = np.abs(sorted(-matches_final))

    if( (topK_dists[0] - topK_dists[1]) < 60 ):
        # Not belonging to DB
        topK_matches = []
        for n in range(0,K):
            topK_matches.append(-1)
    else:
        # Get K paintings from the database with more matches
        topK_matches = (-matches_final).argsort()[:K]
        topK_matches = topK_matches.tolist()

    return topK_matches

def search_matches_ORBS(queries, database, K):
    
    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    matches_final = np.zeros(len(database))
    
    for i in range(len(database)):
        # Match descriptors
        matches = bf.match(database[i],queries)

        # Sort matches in the order of their distance
        matches = sorted(matches, key = lambda x:x.distance)

        matches_good = 0
        for m in matches:
            if m.distance < 70.0:
                matches_good += 1
        print(matches_good)
        # Number of "true" matches
        matches_final[i] = matches_good

    # Get K paintings from the database with more matches
    topK_matches = (-matches_final).argsort()[:K]

    return topK_matches.tolist()

    