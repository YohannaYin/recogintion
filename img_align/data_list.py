# coding=utf-8
import cv2
import numpy as np
import paddle.v2 as paddle
import random
from multiprocessing import cpu_count
# from utils import *

class MyReader:
    def __init__(self, imageSize, center_crop_size = 128):
        self.imageSize = imageSize
        self.center_crop_size = center_crop_size
        self.default_image_size = 128

    def train_mapper(self, sample):
        '''
        map image path to type needed by model input layer for the training set
        '''
        img, label = sample
        # sparse_label = [0 for i in range(1037)]
        # sparse_label[label] = 1

        # def crop_img(img, center_crop_size):
        #     img = cv2.imread(img, 0)
        #     if center_crop_size < self.default_image_size:
        #         side = (self.default_image_size - center_crop_size) / 2
        #         img = img[side: self.default_image_size - side - 1, side: self.default_image_size - side - 1]
        #     return img
        #
        # img = crop_img(img, self.center_crop_size)
        img = cv2.imread(img, 0)
        img = cv2.resize(img, (self.imageSize, self.imageSize))
        # print(label,img.shape)

        return img.flatten().astype('float32'), label
            # , sparse_label
    def val_mapper(self, sample):
        '''
        map image path to type needed by model input layer for the training set
        '''
        img, label = sample
        img = cv2.imread(img, 0)
        img = cv2.resize(img, (self.imageSize, self.imageSize))
        # print(label,img.shape)

        return img.flatten().astype('float32'), label

    def test_mapper(self, sample):
        '''
        map image path to type needed by model input layer for the test set
        '''
        img1, img2= sample
        img1 = paddle.image.load_image(img1)
        img1 = paddle.image.center_crop(img1, 128, is_color=True)
        img2 = paddle.image.load_image(img2)
        img3 = paddle.image.center_crop(img2, 128, is_color=True)

        img1 = cv2.resize(img1, (self.imageSize, self.imageSize))
        img2 = cv2.resize(img2, (self.imageSize, self.imageSize))
        return img1.flatten().astype('float32'), img1.flatten().astype('float32')

    def train_reader(self, train_list, buffered_size=1024):
        def reader():
            with open(train_list, 'r') as f:
                lines = [line.strip() for line in f]
                random.shuffle(lines)
                for line in lines:
                    line = line.strip().split('\t')
                    img_path = line[0]
                    img_label = line[1]
                    # print(img_path,img_label)
                    yield img_path, int(img_label)
        # print("cpu_count()=",cpu_count())
        return paddle.reader.xmap_readers(self.train_mapper, reader, cpu_count(), buffered_size)
    def val_reader(self, val_list, buffered_size=1024):
        def reader():
            with open(val_list, 'r') as f:
                lines = [line.strip() for line in f]
                random.shuffle(lines)
                for line in lines:
                    line = line.strip().split('\t')
                    img_path = line[0]
                    img_label = line[1]
                    # print(img_path,img_label)
                    yield img_path, int(img_label)
        # print("cpu_count()=",cpu_count())
        return paddle.reader.xmap_readers(self.val_mapper, reader, cpu_count(), buffered_size)

    def test_reader(self, test_list, buffered_size=1024):
        def reader():
            with open(test_list, 'r') as f:
                lines = [line.strip() for line in f]
                for line in lines:
                    img_path1, img_path2 = line.strip().split('\t')
                    yield img_path1, img_path2

        return paddle.reader.xmap_readers(self.test_mapper, reader,
                                          cpu_count(), buffered_size)