# Classical Vehicle Detection on the Road

## Computer vision with OpenCV
## Machine Learning with Scikit-Learn

### Here we are going to apply a traditional computer vision approach to write a software pipeline to identify vehicles in a video from a front-facing camera on a car. The code will be incorporated into my [advanced lane line detection project](https://github.com/rzuccolo/rz-advanced-lane-detection). Thanks to Udacity Self-driving Car Nanodegree for providing me the basic skills set to get there!

### This classical approach, basically, requires all parameters to be tuned by hand, which gives a lot of intuition of how it works and why. There is an increasing adoption of deep learning implementations (e.g. [YOLO](https://pjreddie.com/darknet/yolo/) and [SSD](http://www.cs.unc.edu/~wliu/papers/ssd.pdf)) using Convolutional Neural Networks for obstacle and objects detection on the road. In many cases, deep learning has been showing better and more efficient results for the same tasks, but is still kind of a "black box", it is good to learn both techniques. 

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier, Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Normalize the features and randomize a selection for training and testing.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream and create a heat map of recurring detection frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


[//]: # (Image References)

[image1]: ./my_images/1-dataset.png "Dataset"
[image2]: ./my_images/2-hog_gray.png "HOG Grayscale"
[image3]: ./my_images/3-hog_grad.png "HOG gradients"
[image4]: ./my_images/4-hog_features.png "HOG Features"
[image5]: ./my_images/5-color_space_rgb.png "3D RGB Color Space"
[image6]: ./my_images/6-color_space_hsv.png "3D HSV Color Space"
[image7]: ./my_images/7-color_space_YCrCb.png "3D YCrCb Color Space"
[image8]: ./my_images/8-color_hist_img.png "Image Test"
[image9]: ./my_images/9-color_hist_hist.png "Color Histograms"
[image10]: ./my_images/10-color_hist_features.png "Color Features"
[image11]: ./my_images/11-spatial_img.png "Image Test"
[image12]: ./my_images/12-spatial_small.png "Bin Image"
[image13]: ./my_images/13-spatial_feature.png "Spatial Binning Features"
[image14]: ./my_images/14-training_img.png "Dataset Example"
[image15]: ./my_images/15-slide_w1.png "Sliding Window Search, Window 1"
[image16]: ./my_images/16-slide_w2.png "Sliding Window Search, Window 2"
[image17]: ./my_images/17-slide_w3.png "Sliding Window Search, Window 3"
[image18]: ./my_images/18-slide_w4.png "Sliding Window Search, Window 4"
[image19]: ./my_images/19-slide_all.png "Sliding Window Search, All Windows"
[image20]: ./my_images/20_search1.png "Sliding Window Search, Example 1"
[image21]: ./my_images/21_search2.png "Sliding Window Search, Example 2"
[image22]: ./my_images/22_search3.png "Sliding Window Search, Example 3"
[image23]: ./my_images/23_search4.png "Sliding Window Search, Example 4"
[image24]: ./my_images/24_final1a.png "Final Pipeline, Example 1a"
[image25]: ./my_images/25_final1b.png "Final Pipeline, Example 1b"
[image26]: ./my_images/26_final2a.png "Final Pipeline, Example 2a"
[image27]: ./my_images/27_final2b.png "Final Pipeline, Example 2b"


---

## Code Files & Functionality

### 1. Files:

Except for one addition (**vehicle_detection_helpers.py**), the structure and the files are the same as in the [advanced lane line detection project](https://github.com/rzuccolo/rz-advanced-lane-detection):

Specific for Finding Lane Lines:

* **camera_calibration.py**  is the script used to analyze the set of chessboard images, and save the camera calibration coefficients (mtx,dist). That is the first step in the project.
* **perspective_transform.py** is the script used to choose the appropriate perspective transformation matrices (M, Minv) to convert images to bird's-eye view. That is the second step in the project.
* **warp_transformer.py** contains the functions used to color transform, warp and create the binary images.
* **line.py** defines a class to keep track of lane line detection. This information helps the main pipeline code to decide if the current image frame is good or bad.

Specific for Vehicle Detection:

* **vehicle_detection_helpers.py**  several helpers functions: draw bounding boxes, compute color histogram features, compute binned color features, compute HOG (Histogram of Oriented Gradient), extract features from a list of images, convert color spaces, adds "heat" to a map for a list of bounding boxes, imposing a threshold to reject areas affected by false positives, and draw bounding boxes around the labeled vehicles regions.
* **Train_Classifier.ipynb**  jupyter notebook used to train the classifier.

General:

* **load_parameters.py** contains the functions used to load camera coefficients (mtx, dist), perspective matrices (M, Minv), and the Liner SVM trained classifier.
* **main.py** contains the script used to run the video pipeline and create the final annotated video.
* **[Annotated Project Video](https://vimeo.com/213638727)** Click on this link to watch the annotations for the project video.
* **writeup_classical-vehicle-detection.md** is the summary report of the results


Playground using Jupyter notebooks for all stages of the projects can be found here:
**[github repository](https://github.com/rzuccolo/rz-classical-vehicle-detection)**



### 2. Functional codes:

#### 2.1 Camera calibration:
Check on write-up for  [advanced lane line detection project](https://github.com/rzuccolo/rz-advanced-lane-detection).

#### 2.2 Perspective Transform Matrices:
Check on write-up for  [advanced lane line detection project](https://github.com/rzuccolo/rz-advanced-lane-detection).


#### 2.3 Video Pipeline:
Open the **load_parameters.py** and set the proper income directories (code lines 4-17).

Default configuration will:
* Read camera coefficients and perspective matrices from: `output_images/camera_cal`
* Read SVM Liner trained classifier from: `training_dataset`

Open the **main.py** and set the proper output directory and video source (code lines 521-522).

Default configuration will:
* Read video source from parent directory
* Save annotated video to: `output_images`

Execute the script as follow: 
```
python main.py
```

---

## Motivation and Challenge
Recognition of objects on a image is the essence of computer vision. When we look at the world with our own eyes, we are constantly detecting and classifying objects with our brain, and that perception of the world around us is important for driver-less car systems. 

There are a lot challenges behind image classification process. We don't know where in the image the objects will appear, or which size/shape it will be, which color, or how many of those it will show up at same time. Regarding self-driving cars, it applies to vehicles, pedestrians, signs and all other things showing up along the way.

For vehicle detection, it is important to identify and anticipate its position on the road,  how far it is from the reference car,  which way they are going to and how fast they are moving. Same way as we do with our own eye when we drive. 

Here are some of the characteristics that are useful for identifying objects on a image:
* Color
* Position within the image
* Shape
* Apparent size

---

## Training Dataset

Here are links to the labeled data for: [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and  [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip). This is a small dataset with 8,792 vehicles images and 8,968 non-vehicles images, and size of 64x64 pixels. The images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself. 

Udacity recently made available a bigger labeled [dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) with full resolution, which could be used to further augment and better train the classifier, but I decided to carry on the project using only the small dataset and focus on learning the techniques. I will let further improvements as discussed at the end of this writeup for future implementation.

Data exploration:

![alt text][image1]

---

## Methodology
First, we identify and extract the features from the image, and then use it to train a classifier. Next, we execute a window search on the image, on each frame from the video stream, to reliably identify and classify the vehicles. Finally, we must deal with false positives and estimate a bounding box for vehicles detected.

It all comes down to intensity and gradients of intensity of image raw pixels, and how these features capture the color and shape of the object. Here are the main features there are extracted and combined for this project: 

* Histogram of pixel intensity: it reveals the color characteristics of the vehicles
* Gradients of pixel intensity: it reveals the shape characteristics of the vehicles

---

## Features Extraction

### 1. How the Histogram of Oriented Gradients (HOG) are computed:

Code lines (59-97) in `vehicle_detection_helpers.py`. 

I have already explored some benefits of the gradient approach into the [advanced lane line detection project](https://github.com/rzuccolo/rz-advanced-lane-detection), which give us a better information regarding the object shape in the image.

The gradients in a specific direction, with respect to the center of the object image, will give us a "signature" of object shape. Here, it was implemented by the called Histogram of Oriented Gradients (HOG) method, which is well presented and explained [here](https://www.youtube.com/watch?v=7S5qXET179I). I have used [`skimage.hog()`](http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.hog) from scikit-image to execute it. The function takes in a single color channel or grayscaled image as input.

```
skimage.feature.hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), block_norm='L1', visualise=False, transform_sqrt=False, feature_vector=True, normalise=None)
```

![alt text][image2]

![alt text][image3]

![alt text][image4]

Objects on the image may belong at same class but show up in different colors, that is know as non-variance problem. To deal with that, we got evaluate how to better cluster the color information of same class objects. We look into different color spaces and observe how the object we are looking for, stand-out from the background. There are many color spaces out there such as HLV, HSV, LUV, YUV, YCrCb, etc.   **I have adopted YCrCb color space**, since I found it to be clustering the colors pretty well along the Y channel as shown below. **Hence, I have computed HOG futures only for Y channel**. 

Code lines (187-202) in `vehicle_detection_helpers.py`.

![alt text][image5]

![alt text][image6]

![alt text][image7]


### 2. How the histograms of pixel intensity (Color Features) are computed:

Code lines (25-40) in `vehicle_detection_helpers.py`. 

The idea here is to extract from the image the color "signature" for vehicles, so we can later train our classifier and then search for such signatures along the image frames. Basically, locations with similar distributions will point us to close matches. This technique give us some level of structure freedom, since it is not sensitive to a perfect arrange of pixel (cars may have different orientation view for example). Slightly different aspects and orientations will still give us a match. Variations in size are accommodated by normalizing the histograms.

Here is the basic approach to compute the histograms:

```
# Take histograms in R, G, and B
rhist = np.histogram(image[:,:,0])
ghist = np.histogram(image[:,:,1])
bhist = np.histogram(image[:,:,2])

# Concatenate
hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
```
Here is a visualization example:

![alt text][image8]

![alt text][image9]

![alt text][image10]


### 3. Spatial binning of color

Code lines (43-56) in `vehicle_detection_helpers.py`.

Here, we collect the image pixels itself as a feature vector. It can be inefficient to include three (3) color channels of a full resolution image, so we perform a called spatial binning on the image, where close pixels are lumped together to form a lower resolution image. During training we tune how lower we can go and still retain enough information to help in finding vehicles.

How we create this feature vector? We resize the image and convert it to one dimensional vector:

```
# Resize and convert to 1D
small_img = cv2.resize(image, (32, 32))
feature_vec = small_img.ravel()	
```

![alt text][image11]

![alt text][image12]

![alt text][image13]

---

## Training the Linear SVM Classifier

Code lines in `Train_Classifier.ipynb`.

We start by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image14]

Then, we build a function to extract the spatial binning, the color histogram and the HOG for a list of images. The final vectors are the concatenation of the three (3) pieces: spatial binning, color and HOG. Next, we use this function to extract the features for the whole dataset. A generator may be hand at this stage to avoid computer memory issues, but I was dealing with a small dataset, so I did not implement it, it would be a good future improvement.

Next, the feature vectors were normalized to deal with different magnitude of concatenated features. Unbalanced number of features between spatial binning, histogram of colors and HOG, were minimized by dropping the features that were not significantly contributing. All that was  accomplished by applying [`StandardScaler()`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) method, from Python's sklearn package:

```
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)
``` 

Next, we explore different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`) to come up with final tuned parameters.  Here are mine:
```
# Parameter tunning
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = (8,8) # HOG pixels per cell
cell_per_block = (2,2) # HOG cells per block
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [None, None] # Min and max in y to search in slide_window()
```

I have used a [Linear SVM classifier](https://en.wikipedia.org/wiki/Support_vector_machine#Linear_SVM). In past projects I have commented the importance and various aspects of data preparation before training; balanced classes, randomization, train/test splits and others. In this project the data was properly random shuffled, split into training and testing sets and normalized:

```
# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)
# Use a linear SVC 
svc = LinearSVC()
# Train
svc.fit(X_train, y_train)
# Accuracy
svc.score(X_test, y_test)
```

I was able to get 98.9% test accuracy using this small dataset. The SVM training took a process time of only 21.05 seconds.

---

## Sliding Window Search

Code lines (239-423) in `main.py`.

Now that we have a trained classifier, we got implement a way to search for objects on the image. We are going to split the images in subsets of images (windows), extract the same features as described above (binning, color histogram, HOG) and feed into the classifier for prediction.

Ideally, we need to cut the the image subset close to the contour of the object, so the "signature" would be easily detected by the classifier. But we don't know the size of the object that will show up on the image, so we need to search it in multiple scale windows. Here we need to be careful, because it can easily lead to excessive large number of windows to search on each image, which ultimately will make the pipeline inefficient and running slow.

First thing, I ignored the upper half of the image because we don't expect vehicle to show up in there, that is beyond the road horizon. Then, I watched roads video streams to get a sense of object size along the depth perspective, so I could better define the size of windows and the region of interest. 

In the end, I decided to use four (4) windows sizes, with a 75% overlap, searching within specific regions of interest as shown in the images below. With more experimentation, it could be further improved, but over all I am searching a total of 12+34+56+60=162 windows per frame. Here are the windows characteristics:

```
# Window 1
window = (320,240)
cells_per_step = (2,2)
pix_per_cell=(40,30)
ystart = 400
ystop = 700

# Window 2
window = (240,160)
cells_per_step = (2,2)
pix_per_cell=(30,20)
ystart = 380
ystop = 620

# Window 3
window = (160,104)
cells_per_step = (2,2)
pix_per_cell=(20,13)
ystart = 380
ystop = 536

# Window 4
window = (80,72)
cells_per_step = (2,2)
pix_per_cell=(10,9)
ystart = 400
ystop = 490
```

![alt text][image15]

![alt text][image16]

![alt text][image17]

![alt text][image18]

![alt text][image19]


Below there are some examples of of sliding window searches. But at this point the pipeline have multiple overlap windows at identified objects. Next we will apply a technique with heat-maps to estimate a bounding box.

![alt text][image20]

![alt text][image21]

![alt text][image22]

![alt text][image23]


---

## Heat-maps bounding boxes and false positives

Code lines (204-253) in `vehicle_detection_helpers.py`.

At this point the pipeline is getting multiple overlap windows on identified vehicles, and also showing false positives. False positives that are now properly filtered out could lead the driver-less system to take actions when it is not necessary and potentially cause an accident. So the task now is to bound the multiple detection on same object, and get rid of false positives by using a heat-map technique.

To make a heat-map, we simply add "heat" (+=1) for all pixels within windows where a positive detection is reported by the classifier. 

```
heatmap = np.zeros_like(image[:,:,0])
# Add += 1 for all pixels inside each bbox
# Assuming each "box" takes the form ((x1, y1), (x2, y2))
heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
```

To get rid of false positives, we apply a threshold on the heat-map:

```
# Zero out pixels below the threshold
heatmap[heatmap <= threshold] = 0
```

Here are some final examples:

![alt text][image24]

![alt text][image25]

![alt text][image26]

![alt text][image27]


---

## Road Test Video!

* **[Annotated Project Video](https://vimeo.com/213638727)** Click on this link to watch the annotations for the project video.


---	


## Discussion

The pipeline is good for the project video but I would like to see how it goes on other video streams, I will let it for future tests. There are still some eventual false positives and the bounding boxes are a bit "jittery", despite the fact I averaged the heat-maps from previous fifteen (15) frames for each new frame. Nevertheless, so far I am happy with the results! Those are good techniques that allowed me to build a strong understanding and solid base for the task.

It is surely a lot work to properly hand tuning all those parameters, but at same time it gives you a good sense on strengths and weakness of computer vision. The pipeline doesn’t detect cars driving in the opposite direction because of heat-maps averaging over time, which should be solved with more sophisticated techniques, better trained classifiers, or improved sliding search window algorithm. A deep learning classifier seems to be the next step down the learning road now!

---


## Acknowledgments

* [Udacity Self-Driving Car Nanodegree](https://www.udacity.com/drive)
