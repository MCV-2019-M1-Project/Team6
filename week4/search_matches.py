import cv2 as cv
import numpy as np

def search_matches(queries, database, K):

    # Set up FLANN based matcher
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params,search_params)

    print("Database")
    j=0
    matches_final = np.zeros(len(database))
    

    for i in range(len(database)):
        matches = flann.knnMatch(database[i], queries, k=2)
        print(np.shape(database[i]))
        print(np.shape(queries))

        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for l in range(len(matches))]

        matches_good = 0
        # ratio test as per Lowe's paper
        for k,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                matchesMask[k]=[1,0]
                matches_good += 1

        # Number of "true" matches
        matches_final[i] = matches_good

    # Get K paintings from the database with more matches
    topK_matches = (-matches_final).argsort()[:K]

    return topK_matches.tolist()
    