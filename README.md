**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./writeup_images/car_hog.png
[image2]: ./writeup_images/noncar_hog.png
[image3]: ./writeup_images/vehicle_detection2.png
[image4]: ./writeup_images/vehicle_detection3.png
[image5]: ./writeup_images/bboxes_and_heat.png
[image6]: ./writeup_images/labels_map.png
[image7]: ./writeup_images/output_bboxes.png
[video1]: ./project_video.mp4

---

### Histogram of Oriented Gradients (HOG)

#### 1.Implementation and use

The Histogram of Oriented Gradients (HOG) method is a technique used to identify and label objects in an image or video. It can be seen as an extension to the edge oriented histogram. Though the concepts were developed several decades ago, HOG rose to prominence in 2005 as an effective method for detecting pedestrians in images by Dalal and Trigg.

In this project the `hog()` implementation from `skimage.feature` is used to detect vehicles. It is called from `get_hog_features()` in the Python file `hog_helper.py`. `get_hog_features()` is called from `extract_features()` also in the same file which concatenates the HOG features along with spatial and color features into a single feaure vector. 

HOG divides the image into cells, where the cell dimension in `hog()` is called `pixels_per_cell` a tuple of x and y pixel dimensions. As this is a histogram, there are a discrete number of orientation bins `orientations`. Each pixel contributes a weight proportional to its gradient magnitude to an orientation. Block normalization is also used to improve performance. By using overlapping boxes the descriptor is more robust to changes in illumination. In `hog()` this is is controlled by the parameter `cells_per_block`. Dalal and Trigg recommend either 2x2 or 3x3 cells per block. 

As an example, below are images of a vehicle and a non-vehicle with HOG applied each of their channels in `YCrCb` color space. With `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image1]

![alt text][image2]

#### 2. Parameter selection
The recommended number of orientation bins is between 9 to 12 while the recommended cells per block is 2x2 or 3x3. I settled on
`orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` after a little experimentation with satisfactory results.

We can also see that channels 1 and 2 do not add 

#### 3. Classifier training
This function `extract_features()`,implemented in `hog_helper.py`, combines the feature vectors of the HOG, spatial, and color information. The function is called from the second cell of the iPython notebook `Vehicle Detection.ipynb` to generate feature vectors for 8,798 images of vehicles and 8,971 images of non-vehicles. 

After rescaling the feature vectors with `StandardScaler()`, the scaled feature vectors are used to train a linear support vector classifier (SVC) using `sklearn.svm.LinearSVC()`.

We notice that the Y' component of a `YCrCb` image, the luma, contains all the spatial information. Using this fact, we can reduce the feature vector length by only using the Y' channel for the spatial and HOG features. 

The length of a feature vector is 4932. Running the classifier on a test set, we see an accuracy of 97.9%


### Sliding Window Search

#### 1. Sliding window search and parameter 

A sliding window search is used on a subsection of the image containing the road. This is implented in the function `find_cars()` in the `iPython` notebook. The parameter `cells_per_step` describes how much to move the window per step. The parameter `scale` describes how to scale the search window. Setting `scale = 1` uses a window of the same size as the training images (`64x64 pixels`). It is best to use a variety of window sizes to increase the likelihood a particular patch of pixels of a vehicle will be classified correctly. The search window is then resized to `64x64` and the feature vector is computed. Using the previously trained SVC, we determine whether the current window contains a vehicle using `svc.predict()`.

As it is quite likely that any given patch may be misclassified, we choose to accumulate the detected patches in a heatmap and only if the summed pixel value is above a threshold will it be classified as a positive detection. Then  `scipy.ndimage.measurements.label()` is used label each patch of connected pixels and finally a bounding box for each patch is drawn on the image.

#### 2. Examples and optimizations
The steps of the detection process are shown below.
![alt text][image3]

There are several misclassifications, however using threshold on the accumulation of positive classfications we see that the only the two vehicles are classified. To improve the performance, several search window sizes are used and the results are accumulated into the same heatmap. Windows with scalings of 1, 1.5, and 2 were used to generate the above image.

Using more search window sizes also means performing predictions, particularly if the scaling is `<1`. Performance optimizations to achive realtime classifications are discussed below.

![alt text][image4]

---

### Video Implementation

#### 1. Video output
[link to video result](./output_videos/project_video_output_short.mp4)


#### 2. Video pipeline
The video pipeline is much the same as the image pipeline except that heatmaps for the previous 10 frames are stored and summed. This reduces the likelihood of a false positive detection. This is described in the function `process_vehicle_overlay()`

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust? Performance optimization? +Lane detections

TODO


