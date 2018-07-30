import numpy as np
import os.path
from PIL import Image
import matplotlib.pyplot as plt
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
    count = 0
    for i in os.listdir(path):
        count = count + 1
        if i != 'train.txt' and i != 'data_train.txt' and i!='data_validate.txt' and i!='validate.txt' and i!='.DS_Store':
            for f in os.listdir(os.path.join(path, i)):
                p = IdentityMetadata(path, i, f)
                # Check file extension. Allow only jpg/jpeg' files.
                metadata.append(p)
    return metadata

def load_metadata_test(path):
    metadata = []
    for i in os.listdir(path):
        if i != 'train.txt' and i != 'data_train.txt' and i!='data_validate.txt' and i!='validate.txt' and i!='.DS_Store':
                p = os.path.join(path, i)
                # Check file extension. Allow only jpg/jpeg' files.
                metadata.append(p)
    return metadata
def load_metadata_validate(path):
    metadata = []
    count = 0
    for i in os.listdir(path):
        count = count + 1
        if i != 'train.txt' and i != 'data_train.txt' and i!='data_validate.txt' and i!='validate.txt' and i!='.DS_Store':
            for f in os.listdir(os.path.join(path, i)):
                p = IdentityMetadata(path, i, f)
                # Check file extension. Allow only jpg/jpeg' files.
                metadata.append(p)
    return metadata
# load_metadata_train("/mnt/datasets/WebFace/first_round/first_round_train")
