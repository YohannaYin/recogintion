# -*- coding: utf-8 -*-
import cv2
import detect_face
import os
import tensorflow as tf
import numpy as np
import math
from scipy import misc
import imageio
TRAIN_PATH = "/home/yinhong/dataset/train/"
IMG_SIZE_1 = 55
IMG_SIZE_2 = 55
DATA_DIM = 1 * IMG_SIZE_1 * IMG_SIZE_2
#====================  CLASS_DIM 是人脸提取的人数 ========================
CLASS_DIM = 7403
BATCH_SIZE = 128
class IdentityMetadata():
    def __init__(self, base, name, file):
        # dataset base directory
        self.base = base
        # identity name
        self.name = name
        # image file name
        self.file = file

    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.name, self.file)


def load_metadata_train(path):
    metadata = []
    for i in os.listdir(path)[:CLASS_DIM]:
        if os.path.isdir(os.path.join(path, i)):
            for f in os.listdir(os.path.join(path, i)):
                # Check file extension. Allow only jpg/jpeg' files.
                ext = os.path.splitext(f)[1]
                if ext == '.jpg' or ext == '.jpeg':
                    metadata.append(IdentityMetadata(path, i, f))
    return np.array(metadata)

metadata_train = load_metadata_train(TRAIN_PATH)
total_images = len(metadata_train)

minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor
gpu_memory_fraction = 1.0
print('Creating networks and loading parameters')
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    # sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}, log_device_placement=True))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

count = 0
train_data = np.zeros((metadata_train.shape[0], IMG_SIZE_1, IMG_SIZE_2))
train_labels = np.zeros((metadata_train.shape[0]))
missing_count = -1
for i, m in enumerate(metadata_train):
    missing_count = missing_count + 1
    train_labels[missing_count] = metadata_train[i].name
    img = imageio.imread(m.image_path())
    try:
        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        nrof_faces = bounding_boxes.shape[0]#人脸数目
        print('找到人脸数目为：{}'.format(nrof_faces))
        if nrof_faces!=1:
            continue
        # print(bounding_boxes)
        crop_faces=[]
        for face_position in bounding_boxes:
                face_position=face_position.astype(int)
                print(face_position[0:4])
                cv2.rectangle(img, (face_position[0], face_position[1]), (face_position[2], face_position[3]), (0, 255, 0), 2)
                crop=img[face_position[1]:face_position[3],
                         face_position[0]:face_position[2],]
                crop = cv2.resize(crop, (55, 55), interpolation=cv2.INTER_CUBIC)
                crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
                img = np.expand_dims(crop, axis=0)
                img = (img / 255.).astype(np.float32)
                train_data[i] = img
    except:
            missing_count = missing_count - 1
            continue
print(count)
            # crop_faces.append(crop)
#     plt.imshow(crop)
#     plt.show()
#
# plt.imshow(img)
# plt.show()