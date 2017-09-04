import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET

from constants import *		
from selective_search_py.selective_search import *

def load_regions(data='train'):
	""" Loads regions from images, which are loaded one by one simultaneously,
	into a npy matrix and returns it.
	"""
	print "loading every image's regions for "+data
	if data=='train':
		try:
			images_regns = np.load('images_regns.npy')		
		except IOError:
			images_regns = selective_search_regions(data)
	elif data=='test':
		try:
			images_regns = np.load('test_regns.npy')		
		except IOError:
			images_regns = selective_search_regions(data)
	else:
		raise Exception('Incorrect parameter data in load_regions()')
	
	print "Done loading regions."
	return images_regns

def selective_search_regions(data):
	if data=='train':
		folder = folder_train
		save_as = 'images_regns.npy'
	else:
		folder = folder_test
		save_as = 'test_regns.npy'

	images_regns = []
	not_loaded = []
	num = 1
	try:
		for filename in os.listdir(folder):
			path = os.path.join(folder,filename)
			print "loading",filename
			img = cv2.imread(path)
			if img is not None:
				try:
					regions = selective_search(img,color_spaces = ['rgb', 'hsv'],ks = [150, 300, 500],
	                   feature_masks = [(1, 1, 1, 1)],n_jobs = 2)
					regions = [region[1] for region in regions]
					images_regns.append({filename : regions})
					print str(num)+")loaded regions for image:",filename
				except Exception as e:
					print str(e)
					# raise e
					not_loaded.append(filename)
					print str(num)+")could not load regions for image:",filename
				num = num + 1
				try:
					if num==900:
						np.save('test_regns_partial.npy',images_regns)
				except:
					continue
	except Exception as e:
		print str(e)
		# raise e

	print len(not_loaded),"Images Not Loaded"
	np.save(save_as,images_regns)
	# np.save('not_loaded.npy',not_loaded)	
	return images_regns

def load_ground_truth(images_regns,data='train'):
	""" Loads ground truth regions from images names stored in images_regns(along with their regions)
	into a npy matrix and returns it.
	"""
	if data=='train':
		try:
			print "Loading train ground truth from ground_truth.npy"
			return np.load('ground_truth.npy')
		except IOError:
			return extract_ground_truth(images_regns,data)
	elif data=='test':
		try:
			print "Loading test ground truth from test_ground_truth.npy"
			return np.load('test_ground_truth.npy')
		except IOError:
			return extract_ground_truth(images_regns,data)
	else:
		raise Exception('Incorrect parameter data in load_ground_truth()')

def extract_ground_truth(images_regns,data):
	if data=='train':
		path = train_path
		save_as = 'ground_truth.npy'
	else:
		path = test_path
		save_as = 'test_ground_truth.npy'

	print "Extracting ground truth from",path
	ground_truth = []

	for regns in images_regns:
		print "Loading for",regns.keys()[0]
		root  = ET.parse(path+'/'+regns.keys()[0][:-4]+'.xml').getroot()
		truth_rgns = []
		for object in root.findall('object'):
			obj_class = object.find('name').text
			bndbox = object.find('bndbox')
			obj_region = tuple([int(bndbox[i].text) for i in range(4)])
			truth_rgns.append({obj_class : obj_region})
			# print obj_class
			# print obj_region
		ground_truth.append([regns.keys()[0],truth_rgns])

	np.save(save_as,ground_truth)
	return ground_truth

# for i in range(1,len(images_regns)):
# 	name = images_regns[i].keys()[0]
# 	images_regns[i][name] = [region[1] for region in images_regns[i][name]]


def load_regions_from_images(images=None):
	""" Loads regions from images, all of which are which are loaded before loading regions,
	into a npy matrix and returns it.
	"""
	try:
		print "loading every image's regions"
		images_regns = np.load('images_regns.npy')
	except IOError:
		if not images:
			images = load_images_from_folder()
		
		images_regns = []
		not_loaded = []

		for img in enumerate(images):
			try:
				regions = selective_search(img[1])
				images_regns.append(regions[:80])
				print "loaded regions for image:",img[0]
			except:
				print "could not load regions for image:",img[0]
				not_loaded.append(img[0])

		np.save('images_regns.npy',images_regns)

	print "Done loading regions."
	return images_regns

def load_images_from_folder():
	""" Loads images from a given folder into a npy matrix and returns it.
	"""
	try:
		print "loading all images"
		images = np.load('images.npy')
	except IOError:
	    images = []
	    for filename in os.listdir(folder):
	        path = os.path.join(folder,filename)
	        print "loading",filename
	        img = cv2.imread(path)
	        if img is not None:
	            images.append(img)
	    np.save('images.npy',images)

	print "Done loading images."
	return images
