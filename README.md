# Vehicle Detection

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[hog]: ./doc_imgs/hog.png
[bbox]: ./doc_imgs/bbox.png
[vid_heat]: ./doc_imgs/vid_heat.png

---

## 1. Histogram of Oriented Gradients (HOG)

### 1.1. HOG Explanation
First of all, we converted the image to use YUV color space. Then we calculate its HOG feature with the following parameters:

- orientation: 12
- pixels per cell: 16
  Both orientation and number of pixels per cell parameters were the result of some experimentation that resulted in the highest accuracy of SVC.
- cell per block: 2
  I used a small number of cells to allow more interleaving between bounding boxes. In the later step this is useful in reducing the number of false positives when combined with a higher number of overlapping threshold.

Here is an example of hog features of car and non-car images:

![HOG][hog]

I combined the features with color histogram (with histogram bins of 16) and spatial features sequenced as follows: (spatial features + histogram features + hog features). These settings improved the precision score from 0.95 to 0.99

**Location:** See the code under **1. HOG** section in `0. Initial Exploration.ipynb` document.

### 1.2. Training the classifier

I trained an SVCLinear classifier to tell apart cars and non cars in a 64x64 pixels image. In this training process, I scaled the features to zero mean with Standard Scaler and used a GridSearchCV to find the most optimal C parameter. I used Precision instead of Accuracy as the main score as I want to get the false positive errors as low as possible.

This is the final classifier's settings:

```
LinearSVC(C=0.1, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
     verbose=0)
```

Feature vector length: 7284

**Location:** Code blocks under **2.1. Training the classifier** section

## 2. Sliding Window Search

### 2.1. Overview of sliding window search technique

I implemented a sliding window search to create bounding boxes surrounding the images, make a heatmap based on the interleaving areas, and then grouped multiple overlapping sections.

**Location:** Section 2.2. to 2.5. in the notebook document covers this.

### 2.2. Optimization of classifier's performance

As well as using a GridSearchCV to find an optimal parameter, I also optimized the performance of my classifier by further adjusting how the bounding boxes are treated once they are predicted throughout the frames:

1. I set the search area to be between y position of 400 to 645 pixels, or in other words the bottom half of the image that contains the road.
2. I set the scales to be searched to be 1.5, 2.0 and 3.0 times of the original 64 x 64 pixels. This was especially helpful when the pipeline worked on areas with many shadows i.e. it less often mistakenly classified the shadows as cars (False Positives).
3. Overlapping thresholds was set to 3. This means there should be at least 4 bounding boxes overlapping against each other before we can consider an area to contain a car (that area being the overlapping section of these boxes).

These visualizations demonstrate this process and its results better:

![bbox][bbox]

In the left column, I drew all of the bounding boxes to see how many bounding boxes was detected given various scenarios from the video. In the middle column, I visualized the density of overlapping sections (the whiter, the denser). And finally, in the right column, I combined the overlapping boxes into single bounding boxes.

**Location:** See the code block under section **2.5. Combine the bboxes with heatmap**.


## 3. Video Implementation

When working on the video, I added one more optimization: Instead of using the bounding boxes from a single frame, I combined the boxes from two subsequent frames and calculate the bounding boxes from their heatmap output.

This heatmap visualization visualizes the positions of bounding boxes in the duration of 50 video (i.e. the project video):

![vid_heat][vid_heat]

From the heatmap it was clear that adding the overlapping threshold would help reducing the false positives. I came up with the number 6 from testing out various other values and eyeballing the results.

**Location:** Code block under section **3. Video**

## 4. Discussion

Reducing false positives had been the largest issue in this project. Although I have found the settings that minimized false positives, I feel this trial-and-error method to be not ideal in working on a larger scale project. I would like to complete term 2 and term 3 of Self-Driving Car Nanodegree to see if there is a way to improve the pipeline, and then going back and perfecting this project.

I was particularly interested in seeing if it is possible to create a 3D model from detected vehicles like what NVIDIA DriveWorks does. [This paper](https://arxiv.org/pdf/1411.6069.pdf) is promising, I'd be very much interested to work on this further at the end of this Nanodegree program.

The pipeline still fails in these scenarios:
- When there are cars going to our direction, the pipeline won't be able to consistently detect them, since the training data contain only the back sections of the cars.
- Another failure of the pipeline is documented in the `project_video_processed.mp4` starting from 23rd second onwards, where it failed to detect the white car right after the color of the road changes. It is possible to correct this by taking a screenshot of that particular moment and adjusting the parameters and/or retrain the model with some training data augmentation, but I feel like there are better methods out there to handle this problem in a more robust way, thus I refrained from tweaking the pipeline further.
- It would fail when there are cars outside of the searching area, obviously.