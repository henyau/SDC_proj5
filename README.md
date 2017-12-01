### Vehicle Detection Project
**Henry Yau**


[//]: # (Image References)
[image1]: ./writeup_images/car_hog.png
[image2]: ./writeup_images/noncar_hog.png
[image3]: ./writeup_images/vehicle_detection2.png
[image4]: ./writeup_images/vehicle_detection3.png
[image5]: ./writeup_images/bboxes_and_heat.png
[image6]: ./writeup_images/labels_map.png
[image7]: ./writeup_images/output_bboxes.png
[image8]: ./writeup_images/vehicle_detection_fast.png

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
`orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(3, 3)` after a little experimentation with satisfactory results.

In addition to the HOG features, we also use spatial and histogram features, both with 8 bins. By reducing the size of the feature vector, we can speed up the classification so the training images are scaled down by 50%.

#### 3. Classifier training
The function `extract_features_augmented()`,implemented in `hog_helper_optimize.py`, combines the feature vectors of the HOG, spatial, and color information. The function is called from the second cell of the iPython notebook `VehicleDetection_Keras.ipynb` to generate feature vectors for 8,798 images of vehicles and 8,971 images of non-vehicles. The data sources are from KITTI (http://www.cvlibs.net/datasets/kitti/)

The function also creates additional training data by randomly zooming and translating the training data. The idea is to allow a single window size to be used to sweep the image rather than use multiple window sizes.

A neural network model was implemented using Keras with a Tensorflow backend. The network structure is input layer->32 fully connected nodes with ReLu activation funcitons->10 fully connected nodes with sigmoid activation functions->8 fully connected nodes with sigmoid->output node with a sigmoid activation function.

The network is trained to reduce the binary cross entropy with the Adams optimizer. The network has a greater than 99% testing accuracy after several epochs of training, this is not necessarily a good quality however. As the video subimages will not match the training data that closely, a high accuracy/low loss function would result in classifying everything one way or the other. Therefore dropout layers were added to prevent overfitting of the training data.


### Sliding Window Search

#### 1. Sliding window search and parameter 

A sliding window search is used on a subsection of the image containing the road. This is implented in the function `find_cars()` in the `iPython` notebook. The parameter `cells_per_step` describes how much to move the window per step. The parameter `scale` describes how to scale the search window. Setting `scale = 1` uses a window of the same size as the training images (`32x32 pixels`). It is best to use a variety of window sizes to increase the likelihood a particular patch of pixels of a vehicle will be classified correctly, however in the optimized version a single window size is used. The search window is then resized to `32x32` and the feature vector is computed. 

We determine whether the current window contains a vehicle using `.predict()` function of the classifier model.

As it is likely that any given patch may be misclassified, we choose to accumulate the detected patches in a heatmap and only if the summed pixel value is above a threshold will it be classified as a positive detection. Then  `scipy.ndimage.measurements.label()` is used label each patch of connected pixels and finally a bounding box for each patch is drawn on the image.

#### 2. Examples and optimizations
The steps of the detection process are shown below.
![alt text][image3]

There are several misclassifications, however using threshold on the accumulation of positive classfications we see that the only the two vehicles are classified. To improve the performance, several search window sizes are used and the results are accumulated into the same heatmap. Windows with scalings of 1, 1.5, and 2 were used to generate the above image. Each frame requires roughly 6 seconds to compute

Next the image shows the optimized implementation where only a single moving window is used.  Though the number of positive window detections is reduced, it is still accurate enough to classify the vehicles. This implementation also uses Tensorflow. The GPU acceleration provides a significant improvement in speed. Each frame takes less than a second to compute and could actually be significantly faster though with a perhaps unreasonable reduction in the accuracy.
![alt text][image8]




---

### Video Implementation

#### 1. Video output

This video result uses TensorFlow NN model with lane detection:
[link to video result](./output_videos/project_video_output_w_lane_multWindow_Final.mp4)


#### 2. Video pipeline
The video pipeline is much the same as the image pipeline except that heatmaps for the previous 4 frames are stored and summed. This reduces the likelihood of a false positive detection. This is described in the function `process_vehicle_overlay()`. The lane detection from a previous project is also added. 

---


### Discussion
An additional goal is to achieve near real time performance. Without any optimization, it takes a few seconds to process a single frame. To speed up the detection process several steps were optimized. First, the feature vector length was reduced as much as possible while maintaining a high test accuracy. This was accomplished by reducing the training image size by 50% and reducing the number of bins significantly. The feature vector length was shrunk from 4932 to 504. Next to reduce the number of search windows, the training set was augmented by randomly zooming in up to 25% and randomly translating the training images (lines 173-176 in hog_helper_optimize.py). The reasoning behind this is that a single smaller window size can capture and identify a region of a vehicle as a vehicle therefore using additional larger search windows are no longer needed. This data augmentation raises the prevalance of false positives in a single image, however when using the moving sum used in processing a video the false positives tend to not accumulated above the cutoff threshold.

A final speed optimization is simply skipping frames. As one frame is likely not significantly different from the prior frame in a 30fps video, we can skip several frames without much concern.

Lane detection from the previous project is also implemented here with a slight performance hit. The resulting optimized pipeline is show below. This pipeline can process roughly 1 frames per second on a laptop. If we skip 5 frames, we can get process roughly 5 frames per second. 



---



#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust? Performance optimization? 

Optimizing for speed took a significant amount of time. TensorFlow was used to reap the benefits of hardware acceleration and was significantly faster than using Scikit learn. However, developing a model which was able to classify the vehicle accurately was challenging. Changing a single parameter could cause the model with classify the entire image as a vehicle or as not a vehicle. Using dropout layers helped to prevent learning the training the data but it was still almost a crapshoot whether or not a model would work. The training data consisted of photos of the rear of the vehicles, so often the vehicles wont be detected unless the rear of the vehicle is visible. A CNN would have helped as the relationships within the 2D structure of the image would have been maintained. 

