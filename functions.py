import numpy as np
import cv2
from constants import folder_train,folder_test,class_num

def put_labels(img, boxes,labels):
    """ boxes, neg_boxes are lists of tuples with each tuple specifying a box.
    Draws image and object boxes for a class.
    """
    color = (0, 255, 0)
    thickness = 2
    for box in boxes:
        cv2.rectangle(img, box[:2], box[2:], color, thickness)

    for label,box in zip(labels,boxes):
        cv2.putText(img,label,(box[0],box[1]),cv2.FONT_HERSHEY_COMPLEX,0.7,255)

    cv2.imshow('Img', img)
    key = cv2.waitKey(0)
    if key == ord('d'):
        cv2.destroyAllWindows()

def visualize(image_name, pos_boxes, neg_boxes):
    """ pos_boxes, neg_boxes are lists of tuples with each tuple specifying a box.
    Draws image and object boxes for a class.
    """
    try:
        img = cv2.imread(folder_train + '/' + image_name)
    except:
        img = cv2.imread(folder_test + '/' + image_name)
        
    color = (0, 255, 0)
    thickness = 2
    for box in pos_boxes:
        cv2.rectangle(img, box[:2], box[2:], color, thickness)

    color = (0, 0, 255)
    for box in neg_boxes:
        cv2.rectangle(img, box[:2], box[2:], color, thickness)

    cv2.imshow('Img', img)
    key = cv2.waitKey(0)
    if key == ord('d'):
        cv2.destroyAllWindows()

def show(img):
    cv2.imshow('Img', img)
    key = cv2.waitKey(0)
    if key == ord('d'):
        cv2.destroyAllWindows()