import numpy as np
import cv2
import os
from skimage.feature import hog
#from skimage import data, color, exposure
import matplotlib.pyplot as plt

from constants import folder_train,class_num
from functions import visualize,show

def get_positve_negative(images_regns, ground_truth):
    """ For each class returns it's ground truth objects as positive features 
    and 20-50% overlapping boxes as negatives features.
    """
    # -->scale the image before calculating hog
    assert [images_regns[i].keys()[0] for i in range(len(images_regns))] == \
    [ground_truth[i][0] for i in range(len(ground_truth))],'Images in images_regns and ground_truth not equal'

    class_names = get_class_names(ground_truth)
    class_positives = {}
    class_negatives = {}
    for name in class_names:
        class_positives[name] = []
        class_negatives[name] = []

    p_error = 0
    n_error = 0
    #Iterating over each image and finding out positive and negatives.
    for i in range(len(ground_truth)):
        path = os.path.join(folder_train,ground_truth[i][0])
        print ground_truth[i][0]
        img = cv2.imread(path,0)
        #take each ground truth object as positive and 
        #boxes around it with 20-50% overlapping as negative
        for positive in ground_truth[i][1]:
            p_box = positive.values()[0]
            p_box = p_box[1::-1] + p_box[3:1:-1]    #taking row and column convention
            class_name = positive.keys()[0]
            try:
                class_positives[class_name].append(extract(img,p_box))
            except Exception as e:
                print str(e)
                p_error = p_error + 1
                continue

            negatives = []
            regions = images_regns[i].values()[0]
            for regn in regions:
                if filter_region(regn):
                    continue
                overlap = get_overlapping(regn,p_box)
                #print overlap
                if overlap <= 0.5 and overlap >= 0.2:
                    add_regn = True
                    for negative in negatives:
                        if get_overlapping(negative,regn)>=0.7:
                            add_regn = False
                            break
                    if add_regn:
                        negatives.append(regn)
            
            print 'negatives',negatives
            #visualize(ground_truth[i][0],[p_box],negatives)
            for n_box in negatives:
                try:
                    class_negatives[class_name].append(extract(img,n_box))
                except Exception as e:
                    print str(e)
                    n_error = n_error + 1
                    continue
    
    print 'p_error',p_error
    print 'n_error',n_error
    np.save('cat_vs_dogs_positives.npy',class_positives)
    np.save('cat_vs_dogs_negatives.npy',class_negatives)
    return class_positives,class_negatives

def filter_region(region):
    width = region[2] - region[0]
    height = region[3] - region[1]
    # Incorrect region
    if width==0 or height==0:
        print "Incorrect region",region
        return True
    # distorted rects
    if width / height > 4 or height / width > 4:
        return True
    return False
    # # excluding regions smaller than 2000 pixels

def extract(image,box=None):
    """Returns hog feature descriptor for an image.
    """
    if box is not None:
        image = image[box[0]:box[2],box[1]:box[3]]
    # show(image)
    #resize image
    print "Before resize image size:",image.shape
    image = resize(image)
    print "After resize image size:",image.shape
    
    #image = color.rgb2gray(img)
    fd = hog(image, orientations=8, pixels_per_cell=(image.shape[1]/16, image.shape[0]/16),
        cells_per_block=(1, 1), visualise=False, transform_sqrt=True)

    print len(fd)
    return fd

def resize(image):
    if image.shape[1]/image.shape[0] >= 2:
        image = cv2.resize(image,(128,64))
    elif float(image.shape[1])/image.shape[0] <= 0.5:
        image = cv2.resize(image,(64,128))
    else:
        image = cv2.resize(image,(128,128))
    return image


def get_class_names(ground_truth):
    """ Returns names of different categories of objects present in ground_truth.
    """
    class_names = set()
    for positive in ground_truth:
        for d in positive[1]:
            if d.keys()[0] not in class_names:
                class_names.add(d.keys()[0])
    print "Following {0} classes of objects present in images:".format(len(class_names))
    print class_names
    return class_names


def get_overlapping(box_1,box_2):
    """Box is a tuple specifying top left and bottom right points.
    Returns overlapping percentage between 2 boxes.
    """
    (XA1,YA1,XA2,YA2) = box_1
    (XB1,YB1,XB2,YB2) = box_2
    overlap_area = max(0, min(XA2, XB2) - max(XA1, XB1)) * max(0, min(YA2, YB2) - max(YA1, YB1))
    total_area = max(0,XA2-XA1)*max(0,YA2-YA1)+max(0,XB2-XB1)*max(0,YB2-YB1) - overlap_area
    assert total_area>0,"{0},{1}".format(box_1,box_2)
    #print overlap_area,total_area
    return overlap_area/float(total_area)


# def extract_hog(img,box=None):
#     if box is not None:
#         img = img[box[0]:box[2],box[1]:box[3]]    
    
#     gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
#     gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
#     mag, ang = cv2.cartToPolar(gx, gy)
#     bin_n = 16 # Number of bins
#     # quantizing binvalues in (0...16)
#     bin = np.int32(bin_n*ang/(2*np.pi))

#     # Divide to 16 sub-squares
#     bin_cells = []
#     mag_cells = []

#     cellx = img.shape[1]/4
#     celly = img.shape[0]/4

#     for i in range(0,img.shape[0]/celly):
#         for j in range(0,img.shape[1]/cellx):
#             print i*celly,i*celly+celly, j*cellx , j*cellx+cellx
#             bin_cells.append(bin[i*celly : i*celly+celly, j*cellx : j*cellx+cellx])
#             mag_cells.append(mag[i*celly : i*celly+celly, j*cellx : j*cellx+cellx])   

#     hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
#     hist = np.hstack(hists)
#     print(len(hist))
#     return hist


# def extract_hog(img,box=None):
#     """Returns hog feature descriptor for an image.
#     """
#     if box is not None:
#         img = img[box[0]:box[2],box[1]:box[3]]
#     image = color.rgb2gray(img)

#     fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
#                     cells_per_block=(1, 1), visualise=True)

#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

#     ax1.axis('off')
#     ax1.imshow(image, cmap=plt.cm.gray)
#     ax1.set_title('Input image')
#     ax1.set_adjustable('box-forced')

#     # Rescale histogram for better display
#     hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

#     ax2.axis('off')
#     ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
#     ax2.set_title('Histogram of Oriented Gradients')
#     ax1.set_adjustable('box-forced')
#     plt.show()
#     return fd

# regions = selective_search(img, \
#                            color_spaces = ['rgb', 'hsv'],\
#                            ks = [50, 150, 300],\
#                            feature_masks = [(1, 1, 1, 1)])
