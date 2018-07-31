#coding:utf-8
# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils

import load
import dlib
from align import AlignDlib
import cv2
import os
alignment = AlignDlib('shape_predictor_68_face_landmarks.dat')
def load_image(path):
    img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    return img[...,::-1]
def align_image(img):
    return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img),
                           landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
def crop_train(path,result_folder):
    metadata = load.load_metadata_train(path)
    for i, m in enumerate(metadata):
        img_path = m.image_path()
        img = load_image(img_path)
        rect = alignment.getAllFaceBoundingBoxes(img)
        if rect:
            img = align_image(img)
        else:
            img = cv2.resize(img, (250, 250), interpolation=cv2.INTER_CUBIC)
            img = img[77:173, 77:173]
        impt = img_path.split("/mnt/datasets/WebFace/first_round/first_round_train/")[1]
        label = impt.split("/")[0]
        dest_file_path = os.path.join(result_folder,impt)
        label_path = os.path.join(result_folder,label)
        if not os.path.exists(label_path):
            os.mkdir(label_path)
        cv2.imwrite(dest_file_path, img)
        img = (img / 255.).astype(np.float32)

def crop_test(path,result_folder):
    metadata = load.load_metadata_test(path)
    for i, m in enumerate(metadata):
        img_path = m.image_path()
        img = load_image(img_path)
        rect = alignment.getAllFaceBoundingBoxes(img)
        if rect:
            img = align_image(img)
        else:
            img = cv2.resize(img, (250, 250), interpolation=cv2.INTER_CUBIC)
            img = img[77:173, 77:173]
        impt = img_path.split("/mnt/datasets/WebFace/first_round/first_round_test/")[1]
        dest_file_path = os.path.join(result_folder,impt)
        cv2.imwrite(dest_file_path, img)
        img = (img / 255.).astype(np.float32)

def crop_validate(path,result_folder):
    metadata = load.load_metadata_validate(path)
    for i, m in enumerate(metadata):
        img_path = m.image_path()
        img = load_image(img_path)
        rect = alignment.getAllFaceBoundingBoxes(img)
        if rect:
            img = align_image(imggi)
        else:
            img = cv2.resize(img, (250, 250), interpolation=cv2.INTER_CUBIC)
            img = img[77:173, 77:173]
        impt = img_path.split("/mnt/datasets/WebFace/first_round/first_round_validate/")[1]
        label = impt.split("/")[0]
        dest_file_path = os.path.join(result_folder,impt)
        label_path = os.path.join(result_folder,label)
        if not os.path.exists(label_path):
            os.mkdir(label_path)
        cv2.imwrite(dest_file_path, img)
        img = (img / 255.).astype(np.float32)

if __name__ == '__main__':
    # aligned_db_folder_train = "/mnt/datasets/WebFace/first_round/first_round_train/"
    aligned_db_folder_test = "/mnt/datasets/WebFace/first_round/first_round_test/"
    aligned_db_folder_validate = "/mnt/datasets/WebFace/first_round/first_round_validate/"
    # train_folder = "/home/kesci/input/align/train/crop_images_DB"
    test_folder = "/home/kesci/input/align/test/crop_images_DB"
    validate_folder = "/home/kesci/input/align/validate/crop_images_DB"
    # crop_train(aligned_db_folder_train,train_folder)
    crop_test(aligned_db_folder_test,test_folder)
    crop_validate(aligned_db_folder_validate,validate_folder)
