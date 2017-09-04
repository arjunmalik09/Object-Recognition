import skimage
import numpy as np
import cv2

from extract import *
from load import *
from sklearn.externals import joblib	#for importing model

test_regns = load_regions('test')
ground_truth = load_ground_truth(test_regns,'test')

# def find_class(clfs,img,boxes=None):
# 	# if boxes is None:
# 	class_names = clfs.keys()
# 	for box in boxes:
# 		# print box
# 		input = extract_hog(img,box)/10000
# 		max = 0
# 		for name in class_names:
# 			output = clfs[name].predict_proba([input])
# 			# print name,output
# 			output = output[0][1]
# 			if output>max:
# 				max = output
# 				output_class = name
# 				# print name,output
			
# 		print 'Predicted Class',output_class

# # clfs = joblib.load('Classifier/clfs_neg_eq_class_neg.pkl')
# clfs = joblib.load('Classifier 2/clfs_pep_neg_eq_classes_pos.pkl')
# # #get dicts from numpy array
# # clfs = clfs[()]

# img1 = cv2.imread('000071.jpg')
# boxes1 = [(61,75,443,274)]
# find_class(clfs,img1,boxes1)

# img2 = cv2.imread('000001.jpg')
# boxes2 = [(48,240,195,371),(8,12,352,498)]
# find_class(clfs,img2,boxes2)

# img3 = cv2.imread('000002.jpg')
# boxes3 = [(139,200,207,301)]
# find_class(clfs,img3,boxes3)

# img4 = cv2.imread('000008.jpg')
# boxes4 = [(192,16,364,249)]
# find_class(clfs,img4,boxes4)

# # if __name__ == "__main__":