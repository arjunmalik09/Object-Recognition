import skimage
import numpy as np
import cv2
from load import *
from extract import *

#loading regions from images from the PASCAL VOC training data.
#images_regns is a list of dicts with each dict having image name as key and 
#corresponding 100 region boxes as values sorted in decreasing order of perspective regions.
images_regns = load_regions()

print "\n\nRegions Loaded by Selective Search."

ground_truth = load_ground_truth(images_regns)
print "Loaded ground truth...\n"
#ground_truth is list of objects in each image 
#each element in list contains image name and list of dicts of object class and corresponding boxes.

#Get positive and negetive features of image boxes for each class.
[pos,neg] = get_positve_negative(images_regns,ground_truth)

# for v, (i0, j0, i1, j1) in regions:
# 	print 

# (0.1399329058660388, (0, 34, 25, 43))