This was done as part of my B.Tech. project in Semester 6 under the guidance of [Prof. Rajeev Srivastava](http://www.iitbhu.ac.in/cse/index.php/people/faculty/35.html).

### Objective
Find objects in images from any of the 20 [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/) classes!

### Run

Open terminal in project path.
Run jupyter-notebook.
Open Detection.ipynb and run it.

### Code Explanation

1. #### Region Representation and Extraction

* Regions which are extracted from images using selective search:
A region is represented as a rectangular box (y0,x0,y1,x1) where (y0,x0) are top left coordinates and (y1,x1) are bottom right coordinates with y denoting row and x denoting column in image.
images_regns.npy: Regions extracted from Training images in PASCAL VOC.
test_regns.npy: Regions extracted from Testing images in PASCAL VOC.

* Regions which are given as annotation in PASCAL VOC dataset:
A region is represented as a rectangular box (x0,y0,x1,y1) where (x0,y0) are top left coordinates and (x1,y1) are bottom right coordinates with y denoting row and x denoting column in image.
ground_truth.npy: Regions extracted from Training annotation of images in PASCAL VOC.
test_ground_truth.npy: Regions extracted from Testing annotation of images in PASCAL VOC.

2. #### Feature Extraction
Postives and negatives (HoG features) for each class are extracted and stored in cats_vs_dogs_positives.npy and cats_vs_dogs_negatives.npy respectively.

3. #### Training (in Detection.ipynb)
The training is done using extracted features(after applying pca on them).
Tuning finds best gamma,C and components(for pca).

4. #### Testing (in Detection.ipynb)
* Using the test image annotation for boxes, test features are found. Classification report is made.
* Using test images, detection boxes are printed to files which are name as comp3_det_className.
* Detection files can be used to generate precision recall graps after submission to PASCAL VOC.
