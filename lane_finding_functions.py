# -*- coding: utf-8 -*-
"""
Routines for detecting and drawing lane lines
Optimize to be real time
Created on Wed Nov  8 21:22:04 2017

@author: Henry
"""
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.lines import Line2D 

def calibrate_camera():
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    
    # Make a list of calibration images
    images = glob.glob('../camera_cal/calibration*.jpg')
    
    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
    
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
            cv2.imshow('img',img)
            cv2.waitKey(500)        
    cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (1280, 720), None, None)
    return ret, mtx, dist


def undistort_img(img, objpoints, imgpoints):
    """Using object and image points, undistort the raw images"""
#    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


def sobel_thresh(img, orient='x', kernel_size=3, s_thresh=(0, 255)):
    """perform a sobel filter then threshold, input and output images are single channel"""
    if orient =='x':
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0,ksize = kernel_size)
    else:
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1,ksize = kernel_size)
    abs_sobel = np.absolute(sobel)
    
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= s_thresh[0]) & (scaled_sobel <= s_thresh[1])] = 1    
    return sbinary

def mag_thresh(img, kernel_size=3, s_thresh=(0, 255)):
    """perform a sobel filter both x and y directions, use magnitude for treshold test, input and output images are single channel"""
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0,ksize = kernel_size)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1,ksize = kernel_size)
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    gradmag_scaled = np.uint8(255*gradmag/np.max(gradmag))    
    
    sbinary = np.zeros_like(gradmag_scaled)
    sbinary[(gradmag_scaled >= s_thresh[0]) & (gradmag_scaled <= s_thresh[1])] = 1
    
    return sbinary

def dir_thresh(img, kernel_size=3, s_thresh=(0, np.pi/2)):
    """perform a sobel filter both x and y directions, use direction for threshold test,input and output images are single channel"""
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0,ksize = kernel_size)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1,ksize = kernel_size)
    atanGrad = np.arctan2(np.absolute(sobely),np.absolute(sobelx))
    
    
    sbinary = np.zeros_like(atanGrad)
    sbinary[(atanGrad >= s_thresh[0]) & (atanGrad <= s_thresh[1])] = 1
    
    return sbinary

def color_thresh(img, sat_thresh = (5,100),light_thresh = (200,255), hue_thresh = (15,55)):
    """input img is a RGB output is single channel B/W image"""
    
    img_HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    img_Sat = img_HLS[:,:,2]
    img_Hue = img_HLS[:,:,0]    
    img_Light = img_HLS[:,:,1]  
    
    #yellow hue range is roughly 19 to 52, create a mask
    img_Hue_yellow = np.zeros_like(img_Hue)
    img_Hue_yellow[(img_Hue>hue_thresh[0] )&(img_Hue<=hue_thresh[1])]=1
    
    #whites can't be determined using Hue, and value/Lightness is too broad so use in combination with saturation
    img_Light_thresh = np.zeros_like(img_Light)
    img_Light_thresh[(img_Light>=light_thresh[0] )&(img_Light<light_thresh[1] )]=1
    
    img_Sat_thresh = np.zeros_like(img_Sat)
    img_Sat_thresh[(img_Sat>=sat_thresh[0] )&(img_Sat<sat_thresh[1] )]=1
    
    

    combined_binary = np.zeros_like(img_Sat_thresh)
    #combined_binary[( (img_Sat_thresh==1))|img_Hue_yellow==1] = 1
    #combined_binary[(img_Sat_thresh==1) ] = 1
    combined_binary[((img_Light_thresh==1) | (img_Sat_thresh==1))] = 1
    
    #combined_binary[( (img_Hue_yellow==1)&(img_Light_thresh==1))] = 1 #yellow lines are bright
    combined_binary[( (img_Hue_yellow==1))] = 1 #yellow lines are bright
    
    return combined_binary


def processImage(img,sat_thresh = (160,255), light_thresh = (230,255), hue_thresh = (19,50), sx_thresh = (45,255), dx_thresh = (0.6,1.3), kernel_size=5):
    """Processes a single frame of video. Lane thresholding"""
   
    
    # First perform thresholding in colorspace so that lane lines are clearly visable
    img_color_thresh = color_thresh(img, sat_thresh,light_thresh,hue_thresh)
    
    #perform various sobel filters
    img_HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    img_Sat = img_HLS[:,:,2]
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    gradx = sobel_thresh(img_gray,'x', kernel_size, sx_thresh)
    grady = sobel_thresh(img_gray, 'y', kernel_size, sx_thresh)
    mag_binary = mag_thresh(img_gray, kernel_size, sx_thresh)
    dir_binary = dir_thresh(img_gray, kernel_size, dx_thresh)
  
    
    #combine the thresholds
    combined_binary = np.zeros_like(gradx)
    combined_binary[((gradx == 1) ) | ((mag_binary == 1) & (dir_binary == 1))| (img_color_thresh == 1)] = 1
#    combined_binary[((gradx == 1)) | ((mag_binary == 1) & (dir_binary == 1))| (img_color_thresh == 1)] = 1
    #combined_binary[((gradx == 1)| (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
 

    #combined_binary[img_color_thresh == 1] = 1
        
    
    combined_255 = np.uint8(255*combined_binary/np.max(combined_binary))
    color_255 = np.dstack((combined_255, combined_255, combined_255))
    
    return color_255


def perspective_transform():
    

    #src_pts = np.float32([[190,706],[608,440],[670,440],[1150,706]])
    #dst_pts = np.float32([[190,706],[190,0],[900,0],[900,706]])
    
    src_pts = np.float32([[190,706],[590,450],[690,450],[1150,706]])
    dst_pts = np.float32([[280,706],[280,0],[1000,0],[1000,706]])
    
    
    #center of car in warped space
    warp_ratio = 1 #(dst_pts[3,0]-dst_pts[0,0])/(src_pts[3,0]-src_pts[0,0])
    #ctr_pt = (dst_pts[3,0]-dst_pts[0,0])*0.5 
    ctr_pt = (1280)*0.5 
    
    
    image_unprocessed = mpimg.imread('../test_images/'+test_images_list[0])
    image_undistorted = undistort_img(image_unprocessed, objpoints, imgpoints)
    
    #cv2.imshow('img',image_undistorted_lines)
    
    image_shape = (image_undistorted.shape[1], image_undistorted.shape[0])
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    Minv = cv2.getPerspectiveTransform(dst_pts, src_pts)
    image_transformed_lines = cv2.warpPerspective(image_undistorted, M, image_shape, flags=cv2.INTER_CUBIC)
    
    
    ux = src_pts[:,0]
    uy = src_pts[:,1]
    
    wx = dst_pts[:,0]
    wy = dst_pts[:,1]
    
    # Plot the result
    f, (ax1, ax2) = plt.subplots(1,2, figsize=(24, 9))
    f.tight_layout()
    
    ax1.imshow(image_undistorted)
    ax1.add_line(Line2D(ux,uy,linewidth=7, c='r'))
    
    ax1.set_title('Undistorted Image', fontsize=25)
    
    ax2.imshow(image_transformed_lines)
    ax2.add_line(Line2D(wx,wy,linewidth=7, c='r'))
    
    ax2.set_title('Warped Image', fontsize=25)
    
    
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
    
def find_lanes():
    histogram = np.sum(image_transformed_lines[image_transformed_lines.shape[0]*4//5:,:,0], axis=0)

    out_img = np.copy(image_transformed_lines)
    
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    print(leftx_base)
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    print(rightx_base)
    
    nwindows = 9
    
    window_height = np.int(image_transformed_lines.shape[0]/nwindows)
    
    nonzero = image_transformed_lines[:,:,0].nonzero()
    nonzeroy = np.array(nonzero[0]) #y indices of all non-zero pixels
    nonzerox = np.array(nonzero[1])
    
    
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    margin = 100
    minpix = 100
    
    leftlane_ind = []
    rightlane_ind = []
    
    for window in range(nwindows):
        
        win_y_low = image_transformed_lines.shape[0] - (window+1)*window_height
        win_y_high = image_transformed_lines.shape[0] - (window)*window_height
        
        win_x_leftlane_l = leftx_current - margin
        win_x_leftlane_r = leftx_current + margin
        
        win_x_rightlane_l = rightx_current - margin
        win_x_rightlane_r = rightx_current + margin
        
        
        #cv2.rectangle(out_img, (win_x_leftlane_l, win_y_low) ,(win_x_leftlane_r, win_y_high), (255,0,0),3)    
        #cv2.rectangle(out_img, (win_x_rightlane_l, win_y_low) ,(win_x_rightlane_r, win_y_high), (255,0,0),3)
        
        good_leftlane_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_x_leftlane_l) &(nonzerox < win_x_leftlane_r)).nonzero()[0]
        
        good_rightlane_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_x_rightlane_l) &(nonzerox < win_x_rightlane_r)).nonzero()[0]
        
        leftlane_ind.append(good_leftlane_inds)
        rightlane_ind.append(good_rightlane_inds)
        
        #count the number of nonzeros in each window,     
        if len(good_leftlane_inds)>minpix:
            leftx_current = np.int(np.mean(nonzerox[good_leftlane_inds])) #average of inside window
        if len(good_rightlane_inds)>minpix:
            rightx_current = np.int(np.mean(nonzerox[good_rightlane_inds])) #average of inside window
            
        
        
    # Concatenate the arrays of indices
    leftlane_ind = np.concatenate(leftlane_ind)
    rightlane_ind = np.concatenate(rightlane_ind)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[leftlane_ind]
    lefty = nonzeroy[leftlane_ind] 
    rightx = nonzerox[rightlane_ind]
    righty = nonzeroy[rightlane_ind] 
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
        
    plt.plot(histogram)
    plt.show()
    #plt.imshow(out_img)
    #plt.show()
    
    #sliding window with polynomial fit
    
    #visualize polynomial
    ploty = np.linspace(0, image_transformed_lines.shape[0]-1, image_transformed_lines.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    out_img = np.copy(image_transformed_lines)
    out_img[nonzeroy[leftlane_ind], nonzerox[leftlane_ind]] = [255, 0, 0]
    out_img[nonzeroy[rightlane_ind], nonzerox[rightlane_ind]] = [0, 0, 255]
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.show()

if __name__ == "__main__":
    #assumes camera is always the same and image size is always the same so only need to calibrate camera once
    
    