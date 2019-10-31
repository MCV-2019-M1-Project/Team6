import cv2 as cv
import numpy as np
import glob
import pickle
import ml_metrics
import math
import pandas as pd
import os
import yaml

def search(queries, database, distance, k):

# For each of the queries, searches for the K  most similar images in the database. The
#decision is based on the feature vectors and a distance or similarity measure (Euclidean
# distance and Hellinger Kernel similarity. Returns a 2D array containing the results of
#the search for each of the queries.

    final_ranking = np.zeros((len(queries), k), dtype=float)

    if(distance == "euclidean"):
        for i in range(0, len(queries)):
            ranking = np.ones((k, 2), dtype=float) * 9999
            for j in range(0, len(database)):
                # Compute the distance metric
                dist = sum(pow(abs(np.array(database[j]) - np.array(queries[i])), 2))
                # Check the ranking and update it
                if (dist < max(ranking[:, 1])):
                    # Add the distance and the id to the db
                    idx = np.argmax(ranking[:, 1])
                    ranking[idx, 0] = j
                    ranking[idx, 1] = dist
            # Store the closest K images
            for j in range(0, k):
                idx = np.argmin(ranking[:, 1])
                final_ranking[i, j] = ranking[idx, 0]
                ranking[idx, :] = [9999, 9999]

    if(distance == "chisq"):
            for i in range(0, len(queries)):
                ranking = np.ones((k, 2), dtype=float) * 9999
                for j in range(0, len(database)):
                    # Compute the distance metric
                    dist = sum( np.divide(pow(abs(database[j] - queries[i]), 2), (database[j] + queries[i]), out=np.zeros_like(database[j]), where=queries[i]!=0) )
                    # Check the ranking and update it
                    if (dist < max(ranking[:, 1])):
                        # Add the distance and the id to the db
                        idx = np.argmax(ranking[:, 1])
                        ranking[idx, 0] = j
                        ranking[idx, 1] = dist
                # Store the closest K images
                for j in range(0, k):
                    idx = np.argmin(ranking[:, 1])
                    final_ranking[i, j] = ranking[idx, 0]
                    ranking[idx, :] = [9999, 9999]

    if(distance == "hellinger"):
        for i in range(0, len(queries)):
            ranking = np.zeros((k, 2), dtype=float)
            for j in range(0, len(database)):
                # Compute the distance metric
                dist = np.sum(np.sqrt(np.multiply(np.array(database[j]),np.array(queries[i]))))
                # Check the ranking and update it
                if (dist > min(ranking[:, 1])):
                    # Add the distance and the id to the db
                    idx = np.argmin(ranking[:, 1])
                    ranking[idx, 0] = j
                    ranking[idx, 1] = dist
            # Store the closest K images
            for j in range(0, k):
                idx = np.argmax(ranking[:, 1])
                final_ranking[i, j] = ranking[idx, 0]
                ranking[idx, :] = [0, 0]

    final_list = final_ranking.tolist()[0]
    for i in range(len(final_list)):
        final_list[i] = int(final_list[i])

    return final_list