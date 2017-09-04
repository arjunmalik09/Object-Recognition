import skimage
import numpy as np
import cv2

from sklearn.svm import NuSVC
from sklearn.externals import joblib 	#for model persistence

def drop_features(lt,j=2):
	lt = [lt[i] for i in range(len(lt)) if i%j==0]
	return lt

def train_classifier(class_names,class_positives,class_negatives):
	#what can i do
	#1)tune parameters maybe model is underfitting 
	#2)give weights to classes
	#3)use multiclass approaches of sklearn.svm
	#4)use a different method 

	clfs = {}
	for num,name in enumerate(class_names):	
		X = []
		X.extend(class_positives[name])
		y = [1]*len(class_positives[name])
		# X.extend(class_negatives[name])
		# y.extend([0]*len(class_negatives[name]))
		for other_name in class_names:
			if other_name == name:
				continue
			X.extend(class_positives[other_name])
			y.extend([0]*len(class_positives[other_name]))

		X = np.array(X)
		y = np.array(y)
		print 'Training for',name
		print 'Feature Vectors:',len(X)
		clf = NuSVC(nu=0.01,probability=True,verbose=True,shrinking=False)
		clf.fit(X, y)
		clfs[name] = clf
		# # if num == 3:
		# # 	break
		# np.save('clf_'+name+'.npy',clfs[name])

	joblib.dump(clfs, 'Classifier 2/clfs_pep_neg_eq_classes_pos.pkl') 




class_positives = np.load('class_positives.npy')
class_negatives = np.load('class_negatives.npy')

#get dicts from numpy array
class_positives = class_positives[()]
class_negatives = class_negatives[()]
class_names = class_positives.keys()

#scale features
for name in  class_names:
	class_positives[name] = [p/10000 for p in class_positives[name]]
	class_negatives[name] = [n/10000 for n in class_negatives[name]]	

class_names = class_names[8:len(class_names)-3]
print class_names

#dropping some neagtives for following classes 
#'person','dog','car','chair'
class_negatives['person'] = drop_features(drop_features(class_negatives['person']))
class_positives['person'] = drop_features(drop_features(class_positives['person']))
class_negatives['car'] = drop_features(class_negatives['car'])
class_negatives['chair'] = drop_features(class_negatives['chair'])

#using one vs rest approach for classfication
#taking other class's positives as negatives 
#plus this class's negatives as negatives

print [[name,len(class_negatives[name])] for name in class_names]
print[len(class_positives[name]) for name in class_names]
