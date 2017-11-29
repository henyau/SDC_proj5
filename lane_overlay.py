# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 15:57:45 2017

@author: Henry
"""

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from matplotlib.lines import Line2D 

#%matplotlib qt

def calibrate_camera():
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    
    
    # Make a list of calibration images
    images = glob.glob('./data/camera_cal/calibration*.jpg')
    
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
#            img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
#            cv2.imshow('img',img)
#            cv2.waitKey(500)        
    cv2.destroyAllWindows()
    return objpoints, imgpoints



def undistort_img(img, objpoints, imgpoints, mtx, dist):
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
    combined_binary[((img_Light_thresh==1) | (img_Sat_thresh==1))] = 1
    combined_binary[( (img_Hue_yellow==1))] = 1 #yellow lines are bright
    
    return combined_binary

def findRoadCurvature(PolyCoeff):
    """returns the left and right radii of curvature"""    
    # need a conversion from screenspace to world space
    # because of the transformation, there is an additional factor
    # using https://mutcd.fhwa.dot.gov/htm/2009/part3/part3b.htm as a guide
    y_ppm = 95/9.144 # pixels per meter. gap between markings is 30 feet
    x_ppm = 419/3.7 # lane width is 3.7 meters

    
    Rcurvature = ((1+(2*PolyCoeff[0]*(y_ppm**2/x_ppm)+PolyCoeff[1]*(y_ppm/x_ppm))**2)**(3/2))/(2*PolyCoeff[0]*(y_ppm**2/x_ppm))
    
    return Rcurvature

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
    combined_binary[((gradx == 1) ) | ((mag_binary == 1) & \
                     (dir_binary == 1))| (img_color_thresh == 1)] = 1
    
    combined_255 = np.uint8(255*combined_binary/np.max(combined_binary))
    color_255 = np.dstack((combined_255, combined_255, combined_255))
    
    return color_255

def perspectiveWarp():
    src_pts = np.float32([[190,706],[590,450],[690,450],[1150,706]])
    dst_pts = np.float32([[280,706],[280,0],[1000,0],[1000,706]])
    
    #center of car in warped space
    warp_ratio = 1 #(dst_pts[3,0]-dst_pts[0,0])/(src_pts[3,0]-src_pts[0,0])
    ctr_pt = (1280)*0.5 
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    Minv = cv2.getPerspectiveTransform(dst_pts, src_pts)
    
    return M, Minv, ctr_pt, warp_ratio

def evalpoly(polycoeff, yeval):
    """return polynomial evaluated at yeval"""
    return polycoeff[0]*yeval**2 + polycoeff[1]*yeval + polycoeff[2]

def draw_overlay(img):
    """draw a green polygon over lane"""
    
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    # Create an image to draw the lines on    
    img_overlay = np.copy(img)

    newunwarp = np.zeros_like(img_overlay)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(newunwarp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(newunwarp, Minv, (img.shape[1], img.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(img_overlay, 1, newwarp, 0.3, 0)
    
    return result , newunwarp


class Line():
    """class keeps track of important line parameters"""
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        
        # average the past n x values
        self.n = 15         
        self.yextents = 720 # shouldn't hardcode...
        
        self.yrange = np.linspace(0, self.yextents-1, self.yextents)
        
    def evalpoly(self, polycoeff, yeval):
        """return polynomial evaluated at yeval"""
        return polycoeff[0]*yeval**2 + polycoeff[1]*yeval + polycoeff[2]
        
    def add_new_data(self, xpix_coord, ypix_coord):
        """update parameters given new detected lane pixels"""
        
        self.allx = xpix_coord
        self.ally = ypix_coord
        old_coeffs = self.current_fit
     
        self.current_fit = np.polyfit(self.ally, self.allx, 2)
        
        self.diffs = self.current_fit - old_coeffs
        
        xfitted = self.evalpoly(self.current_fit, self.yrange)

  
        if len(self.recent_xfitted) == self.n:
            #compare current radius to average curvature, if too extreme, don't include
            rad_cur = findRoadCurvature(self.current_fit)
            rad_diff = np.abs(self.radius_of_curvature-rad_cur)
            
            
            #if rad_diff<5000:
            
                # pop off and add new xfitted
            del self.recent_xfitted[0]
            self.recent_xfitted.append(xfitted)            
  
        else:
            self.recent_xfitted.append(xfitted)
#            self.current_fit.append(np.polyfit(self.ally, self.allx, 2))
            
        #take average of all x's
        self.bestx = sum(self.recent_xfitted)/len(self.recent_xfitted)
        
        #use the average evaluated x to come up with polynomial fit
        self.best_fit = np.polyfit(self.yrange, self.bestx, 2)
        
        self.radius_of_curvature = findRoadCurvature(self.best_fit)
    def reset(self):       
        self.detected = False  

def draw_overlay_LR(img, left_fitx, right_fitx,Minv):
    
    # Create an image to draw the lines on
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )#/y_ppm  
    
    img_overlay = np.copy(img)

    newunwarp = np.zeros_like(img_overlay)
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(newunwarp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(newunwarp, Minv, (img.shape[1], img.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(img_overlay, 1, newwarp, 0.3, 0)
    
    return result

 
def process_lane_overlay(imgOrigin, objpoints, imgpoints, mtx, dist, M, Minv, left_line, right_line, warp_ratio, ctr_pt):
    y_ppm = 95/9.144 # pixels per meter. gap between markings is 30 feet
    x_ppm = 419/3.7 # lane width is 3.7 meters

    img = np.copy(imgOrigin)
    # undistorts camera lense effects
    image_undistorted = undistort_img(img, objpoints, imgpoints, mtx, dist)
    
    image_shape = (image_undistorted.shape[1], image_undistorted.shape[0])
    
    # initial preprocessing to emphasize lane lines
    image_processed = processImage(image_undistorted,(160,255),(230,255),(19,50),(45,255),(0.6,1.3),21)
    # warps the image so that one point perspective parallel lines appear parallel
    image_transformed_lines = cv2.warpPerspective(image_processed, M, image_shape, flags=cv2.INTER_CUBIC)

    # check if line has been computed before, if not then compute a moving 
    # window to find a polynomial fit for the lane lines
    
    # already have line data, use existing polynomials
    if left_line.detected == True: 
        nonzero = image_transformed_lines[:,:,0].nonzero()
        nonzeroy = np.array(nonzero[0]) #y indices of all non-zero pixels
        nonzerox = np.array(nonzero[1])
        margin = 100
        

        
        #leftlane_ind = ((nonzerox>left_line.bestx -margin) &(nonzerox<left_line.bestx+margin))
        #rightlane_ind = ((nonzerox>right_line.bestx-margin) &(nonzerox<right_line.bestx+margin))
        leftlane_ind = ((nonzerox>evalpoly(left_line.best_fit,nonzeroy)-margin) &(nonzerox<evalpoly(left_line.best_fit,nonzeroy)+margin))
        rightlane_ind = ((nonzerox>evalpoly(right_line.best_fit,nonzeroy)-margin) &(nonzerox<evalpoly(right_line.best_fit,nonzeroy)+margin))

        
        #collect all the nonzero pixels for each lane region
        leftx = nonzerox[leftlane_ind]
        lefty = nonzeroy[leftlane_ind]
        rightx = nonzerox[rightlane_ind]
        righty = nonzeroy[rightlane_ind]
        
        if((len(leftlane_ind)<100) or (len(rightlane_ind)<100)):
            print(len(leftlane_ind))
            print(len(rightlane_ind))
            left_line.reset()
            right_line.reset()
        else:   
        
            left_line.add_new_data(leftx, lefty)
            right_line.add_new_data(rightx, righty)

#            rad_of_curvature = (left_line.radius_of_curvature+right_line.radius_of_curvature)*0.5
            dist_from_center = warp_ratio*(ctr_pt-(right_line.bestx[-1]+left_line.bestx[-1])*0.5)/x_ppm

            # if curvatures don't match or distance is not close to what a lane should be
            if (np.abs(dist_from_center>1) | (np.abs(left_line.radius_of_curvature-right_line.radius_of_curvature)>700)):
                left_line.reset()
                right_line.reset()

    # this is intentionally not an else if, if the attempt to use previous lines
    # goes badly, then the lines are are reset, and this is executed
    if left_line.detected == False: 
        histogram = np.sum(image_transformed_lines[image_transformed_lines.shape[0]*4//5:,:,0], axis=0)
        
        out_img = np.copy(image_transformed_lines)
        
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        nwindows = 12
        margin = 100
        minpix = 100        
        window_height = np.int(image_transformed_lines.shape[0]/nwindows)
        
        nonzero = image_transformed_lines[:,:,0].nonzero()
        nonzeroy = np.array(nonzero[0]) #y indices of all non-zero pixels
        nonzerox = np.array(nonzero[1])
        
        
        leftx_current = leftx_base
        rightx_current = rightx_base              
        
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

        leftlane_ind = np.concatenate(leftlane_ind)
        rightlane_ind = np.concatenate(rightlane_ind)  
        
        leftx = nonzerox[leftlane_ind]
        lefty = nonzeroy[leftlane_ind] 
        rightx = nonzerox[rightlane_ind]
        righty = nonzeroy[rightlane_ind] 

        #todo: should also reset the history of each line
        
        left_line.add_new_data(leftx, lefty)
        left_line.detected = True
        right_line.add_new_data(rightx, righty)  
        right_line.detected = True
        
        rad_of_curvature = (left_line.radius_of_curvature+right_line.radius_of_curvature)*0.5
        dist_from_center = warp_ratio*(ctr_pt-(right_line.bestx[-1]+left_line.bestx[-1])*0.5)/x_ppm

        
    output_img = draw_overlay_LR(img, left_line.bestx, right_line.bestx, Minv) 
    
    
#    rad_of_curvature = (left_line.radius_of_curvature+right_line.radius_of_curvature)*0.5
#    dist_from_center = (ctr_pt-(right_line.bestx[-1]+left_line.bestx[-1])*0.5)/x_ppm
    
#    radius_of_curvature_s = 'Radius of Curvature: '+'{:0.4f}'.format(rad_of_curvature)+'m'
#    dist_from_center_s = 'Distance from middle: '+'{:0.4f}'.format(dist_from_center)+'m'
    #output_img_txt = overlayText(output_img, radius_of_curvature_s, (20,20), 2)
#    cv2.putText(output_img, radius_of_curvature_s,(20,70), font, 2,(255,255,255),2,cv2.LINE_AA)
#    cv2.putText(output_img, dist_from_center_s,(20,120), font, 2,(255,255,255),2,cv2.LINE_AA)
    
    return output_img         

if __name__ == "__main__":
    objpoints, imgpoints = calibrate_camera()

    ret, mtx, dist, rvecs, tvecs = \
            cv2.calibrateCamera(objpoints, imgpoints, (1280, 720), None, None)
            
    M, Minv, ctr_pt, warp_ratio = perspectiveWarp()
        
    left_line = Line()    
    right_line = Line()
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    video_unprocessed = VideoFileClip("./data/test_videos/project_video.mp4").subclip(5,6)
    video_processed = video_unprocessed.fl_image(process_lane_overlay) 
    
    video_output = './output_videos/project_video_output_python.mp4'
    video_processed.write_videofile(video_output, audio=False)