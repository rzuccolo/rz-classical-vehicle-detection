# Classical Vehicle Detection on the Road

## Computer vision with OpenCV
## Machine Learning with Scikit-Learn

### Here we are going to apply a traditional computer vision approach to write a software pipeline to identify vehicles in a video from a front-facing camera on a car. The code will be incorporated into my [advanced lane line detection project](https://github.com/rzuccolo/rz-advanced-lane-detection). 

### Thanks to Udacity Self-driving Car Nanodegree for providing me the basic skills set to get there!

### This classical approach, basically, requires all parameters to be tuned by hand, which gives a lot of intuition of how it works and why. There is an increasing adoption of deep learning implementations (e.g. [YOLO](https://pjreddie.com/darknet/yolo/) and [SSD](http://www.cs.unc.edu/~wliu/papers/ssd.pdf)) using Convolutional Neural Networks for obstacle and objects detection on the road. In many cases, deep learning has been showing better and more efficient results for the same tasks, but is still kind of a "black box". It is good to learn both techniques. 

---

**Vehicle Detection Project**

The goals/steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier, Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Normalize the features and randomize a selection for training and testing.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream and create a heat map of recurring detection frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


[//]: # (Image References)

[image1]: ./my_images/1-dataset.png "Dataset"
[image2]: ./my_images/24_final1a.png "Final Pipeline, Example 1a"
[image3]: ./my_images/25_final1b.png "Final Pipeline, Example 1b"



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

The default configuration will:
* Read video source from parent directory
* Save annotated video to `output_images`

Execute the script as follow: 
```
python main.py
```

---

## Motivation and Challenge
Recognition of objects on an image is the essence of computer vision. When we look at the world with our own eyes, we are constantly detecting and classifying objects with our brain, and that perception of the world around us is important for driverless car systems. 

There are a lot of challenges behind image classification process. We don't know where in the image the objects will appear, or which size/shape it will be, which color, or how many of those it will show up at the same time. Regarding self-driving cars, it applies to vehicles, pedestrians, signs and all other things showing up along the way.

For vehicle detection, it is important to identify and anticipate its position on the road,  how far it is from the reference car,  which way they are going to and how fast they are moving. Same way as we do with our own eye when we drive. 

Here are some of the characteristics that are useful for identifying objects on an image:
* Color
* Position within the image
* Shape
* Apparent size

---

## Training Dataset

Here are links to the labeled data for: [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and  [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip). This is a small dataset with 8,792 vehicles images and 8,968 non-vehicles images, and size of 64x64 pixels. The images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself. 

Udacity recently made available a bigger labeled [dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) with full resolution, which could be used to further augment and better train the classifier, but I decided to carry on the project using only the small dataset and focus on learning the techniques. I will let further improvements as discussed at the end of this write-up for future implementation.

Data exploration:

![alt text][image1]

---

## Methodology
First, we identify and extract the features from the image, and then use it to train a classifier. Next, we execute a window search on the image, on each frame from the video stream, to reliably identify and classify the vehicles. Finally, we must deal with false positives and estimate a bounding box for vehicles detected.

It all comes down to intensity and gradients of intensities of image raw pixels, and how these features capture the color and shape of the object. Here are the main features there are extracted and combined for this project: 

* Histogram of pixel intensity: it reveals the color characteristics of the vehicles
* Gradients of pixel intensity: it reveals the shape characteristics of the vehicles

---


Here are some final examples:

![alt text][image2]

![alt text][image3]


---

## Road Test Video!

* **[Annotated Project Video](https://vimeo.com/213638727)** Click on this link to watch the annotations for the project video.


---    


## Discussion

The pipeline is good for the project video but I would like to see how it goes on other video streams, I will let it for future tests. There are still some eventual false positives and the bounding boxes are a bit "jittery", despite the fact I averaged the heat-maps from previous fifteen (15) frames for each new frame. Nevertheless, so far I am happy with the results! Those are good techniques that allowed me to build a strong understanding and solid base for the task.

It is surely a lot of work to properly hand tuning all those parameters, but at the same time, it gives you a good sense of strengths and weakness of computer vision. The pipeline doesn’t detect cars driving in the opposite direction because of heat-maps averaging over time, which should be solved with more sophisticated techniques, better-trained classifiers, or improved sliding search window algorithm. A deep learning classifier seems to be the next step down the learning road now!

---


## Acknowledgments

* [Udacity Self-Driving Car Nanodegree](https://www.udacity.com/drive)
