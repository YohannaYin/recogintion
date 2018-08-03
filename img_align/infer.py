# coding:utf-8
import gzip
import numpy as np
import pandas as pd
import paddle.v2 as paddle
import load
from model import *
from data_list import *
from align import *
from scipy.spatial.distance import cosine
alignment = AlignDlib('shape_predictor_68_face_landmarks.dat')
def load_image(path):
    img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    return img[...,::-1]
def align_image(img):
    return alignment.align(64, img, alignment.getLargestFaceBoundingBox(img),
                           landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

# Set constants
imageSize = 64
dim = 64
DATA_DIM = 1 * imageSize * imageSize
CLASS_DIM = 1037
BATCH_SIZE = 50
# PaddlePaddle init
paddle.init(use_gpu=False, trainer_count=2)

# Define input layers
image = paddle.layer.data(
    name="image", type=paddle.data_type.dense_vector(DATA_DIM))
# Load files,使用cost和erro最小的参数
with gzip.open('./params_pass_70.tar.gz', 'r') as f:
    parameters = paddle.parameters.Parameters.from_tar(f)
# Configure new intermediate neural network.
em, out, fc = resnet_baseline(image, CLASS_DIM)
# 存放所有的测试图片向量
test_data = []
# Get all files
all_file_list = [line.strip().split("\n")[0] for line in open("/home/kesci/work/test_pair.list")]


# Save files
def cosine_distance_numpy(vector1, vector2):
    vector1 = vector1.reshape([-1])
    vector2 = vector2.reshape([-1])
    cosV12 = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    return cosV12


def distance(emb1, emb2):
    return 1 / (1 + (np.sum(np.square(emb1 - emb2))))
embs = []
test_data = []
count = 0
# 将人脸进行对齐之后再进行infer
for image_file in all_file_list:
    # img = cv2.imread(image_file, 0)
    img = load_image(image_file)
    rect = alignment.getAllFaceBoundingBoxes(img)
    if rect:
        img = align_image(img)
    else:
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
    # img = cv2.resize(img, (imageSize, imageSize))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    n = img.flatten().astype('float32')
    test_data.append((n,))
pairs_id = pd.read_table('/mnt/datasets/WebFace/first_round/first_round_pairs_id.txt')
result = {}
count = 0
for pairs, i in zip(pairs_id['pairs_id'].values, pairs_id['pairs_id'].apply(lambda x: x.split('_'))):
    probs = paddle.infer(output_layer=em,
                         parameters=parameters,
                         input=test_data[count:count + 2])
    # print(pairs)
    # print(probs[0,:], probs[1, :],probs[2,:])
    result[pairs] = distance(probs[0, :], probs[1, :])
    count = count + 2
final = pd.DataFrame({'pairs_id': result.keys(), 'prob': result.values()})
final.to_csv('./mysubmission1.csv', index=False)