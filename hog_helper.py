# -*- coding: utf-8 -*-
"""
A collection of helper functions for object detection
Created on Fri Oct 27 23:16:41 2017

@author: Henry
"""

import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog


def convert_color(img, conv='RGB2YCrCb'):
    """convert from RGB other color spaces"""
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    """Calls hog function from skimage with or with visualization"""
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, block_norm='L1',
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, block_norm='L1',
                       visualise=vis, feature_vector=feature_vec)
        return features

def bin_spatial(img, size=(32, 32)):
    """Create a 1-D array to bin thee color channels. 
    output size is size[0]*size[1]*3"""
    
    #testing smaller feature vector, use only Y channel
    color1 = cv2.resize(img[:,:,0], size).ravel()
#    color2 = cv2.resize(img[:,:,1], size).ravel()
#    color3 = cv2.resize(img[:,:,2], size).ravel()
#    return np.hstack((color1, color2, color3))
    return np.hstack(color1)
                        
def color_hist(img, nbins=32, bins_range=(0, 256)):    #bins_range=(0, 256)
    """Computes histogram of each of the three color channels and concatenates 
    into 1D array. Output size 3*nbins """
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256), orient = 9, 
                        pix_per_cell = 8, cell_per_block = 2):
    """Computes feature vector of a list of images (imgs is list of filenames)
    feature vector is concatenation of spatial, color and HOG features"""
    
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: 
            feature_image = np.copy(image)   
            
        
            
        #use RGB on color histogram
        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # Apply color_hist() also with a color space option now
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
        
        # Apply hog_features:        
        # testing only using Y channel, a 1% drop in accuracy but a huge speed up
        hog1 = get_hog_features(feature_image[:,:,0], orient, pix_per_cell, cell_per_block, vis=False,feature_vec=False)
        #hog2 = get_hog_features(feature_image[:,:,1], orient, pix_per_cell, cell_per_block, vis=False,feature_vec=False)
        #hog3 = get_hog_features(feature_image[:,:,2], orient, pix_per_cell, cell_per_block, vis=False,feature_vec=False)
    
        hog_feat1 = hog1.ravel() 
        #hog_feat2 = hog2.ravel() 
        #hog_feat3 = hog3.ravel() 
        hog_features = np.hstack((hog_feat1))
#        hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
       
#        hog_features = get_hog_features(feature_image, orient, pix_per_cell, cell_per_block, 
#                        vis=False, feature_vec=True)
        
        # Append the new feature vector to the features list
#        features.append(np.concatenate((spatial_features, hist_features)))
        features.append(np.concatenate((spatial_features, hist_features,hog_features)))
#        features.append(hog_features)
        
    
    
    # Return list of feature vectors
    return features


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    """function to draw bounding boxes"""
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def add_heat(heatmap, bbox_list):
    """heatmap is a 2D array, +1 for each pixel in each bounding box"""
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap

def apply_threshold(heatmap, threshold):
    '''Zero out pixels below the threshold'''
    heatmap[heatmap <= threshold] = 0
    
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    """Draw bounding boxes onto img, labels has same dimension as img with 
    integer values corresponding to vehicle label or else 0"""
    
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        
        # Draw the box on the image if larger than some threshold 
        if np.abs(bbox[0][0]-bbox[1][0]) > 50 and np.abs(bbox[0][1]-bbox[1][1]) > 50:
           
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

