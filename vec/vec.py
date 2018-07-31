#! /usr/bin/python
import pickle
import numpy as np
from PIL import Image

def vectorize_imgs(img_path):
    with Image.open(img_path) as img:
        arr_img = np.asarray(img, dtype='float32')
        return arr_img

def read_list_file(list_file):
    x, y = [], []
    with open(list_file, "r") as f:
        lines = [line.strip() for line in f]
        for line in lines:
            path, label = line.strip().split('\t')
            x.append(vectorize_imgs(path))
            y.append(int(label))
    return np.asarray(x, dtype='float32'), np.asarray(y, dtype='int32')

def read_list_pair_file(pair_file):
    x1, x2, y = [], [], []
    with open(pair_file, "r") as f:
        for line in f.readlines():
            p1, p2, label = line.strip().split('\t')
            x1.append(vectorize_imgs(p1))
            x2.append(vectorize_imgs(p2))
            y.append(int(label))
    return np.asarray(x1, dtype='float32'), np.asarray(x2, dtype='float32'), np.asarray(y, dtype='int32')

def load_testX1():
    with open('/home/kesci/work/testX1.pkl', 'rb') as f:
        testX1 = pickle.load(f)
        return testX1
def load_testX2():
    with open('/home/kesci/work/testX2.pkl', 'rb') as f:
        testX2 = pickle.load(f)
        return testX2
def load_testY():
    with open('/home/kesci/work/testY.pkl', 'rb') as f:
        testY  = pickle.load(f)
        return testY
def load_validY():
    with open('/home/kesci/work/validX.pkl', 'rb') as f:
        validX = pickle.load(f)
        return validX
def load_validY():
    with open('/home/kesci/work/validY.pkl', 'rb') as f:
        validY = pickle.load(f)
        return validY
def load_trainX():
    with open('/home/kesci/work/trainX.pkl', 'rb') as f:
        trainX = pickle.load(f)
        return trainX
def load_trainY():
    with open('/home/kesci/work/trainY.pkl', 'rb') as f:
        trainY = pickle.load(f)
        return trainY
if __name__ == '__main__':
    testX1, testX2, testY = read_list_pair_file('/home/kesci/work/test_pair.list')
    validX, validY = read_list_file('/home/kesci/work/val.list')
    trainX, trainY = read_list_file('/home/kesci/work/train.list')

    print(testX1.shape, testX2.shape, testY.shape)
    print(validX.shape, validY.shape)
    print(trainX.shape, trainY.shape)
    with open('/home/kesci/work/testX1.pkl', 'wb') as f:
        pickle.dump(testX1, f, pickle.HIGHEST_PROTOCOL)
    with open('/home/kesci/work/testX2.pkl', 'wb') as f:
        pickle.dump(testX2, f, pickle.HIGHEST_PROTOCOL)
    with open('/home/kesci/work/testY.pkl', 'wb') as f:
        pickle.dump(testY , f, pickle.HIGHEST_PROTOCOL)
    with open('/home/kesci/work/validX.pkl', 'wb') as f:
        pickle.dump(validX, f, pickle.HIGHEST_PROTOCOL)
    with open('/home/kesci/work/validY.pkl', 'wb') as f:
        pickle.dump(validY, f, pickle.HIGHEST_PROTOCOL)
    with open('/home/kesci/work/trainX.pkl', 'wb') as f:
        pickle.dump(trainX, f, pickle.HIGHEST_PROTOCOL)
    with open('/home/kesci/work/trainY.pkl', 'wb') as f:
        pickle.dump(trainY, f, pickle.HIGHEST_PROTOCOL)