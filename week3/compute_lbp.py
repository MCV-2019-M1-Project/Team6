# IMPORTS #
import cv2 as cv
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

"""
radius = 2
n_points = 8 * radius
METHOD = 'uniform'
"""

def lbp_pattern_and_histogram(im, n_points, radius, METHOD):
    n_points = n_points * radius
    #lbp_pattern = local_binary_pattern(im, n_points, radius, METHOD)
    lbp_pattern = local_binary_pattern(im, 8, 1)
    n_bins = int(lbp_pattern.max() + 1)
    hist,_ = np.histogram(lbp_pattern, density = True, bins = n_bins, range = (0, n_bins))
    return hist, lbp_pattern

def list_to_list_of_lists(lst):
    final_list = []
    for elem in lst:
        #sub = elem.split(',')
        final_list.append([elem])
    return final_list

def compute_lbp(im, n_points, radius, METHOD):
    # Compute the LBP pattern for each channel
    hist1, lbp1 = lbp_pattern_and_histogram(im[:,:,0], n_points, radius, METHOD)
    hist2, lbp2 = lbp_pattern_and_histogram(im[:,:,1], n_points, radius, METHOD)
    hist3, lbp3 = lbp_pattern_and_histogram(im[:,:,2], n_points, radius, METHOD)
    # Concatenate all channels
    hists = np.concatenate((hist1,hist2,hist3))
    # Format settings
    hists_list = list_to_list_of_lists(hists)

    return hists_list