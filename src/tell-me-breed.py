import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from os import listdir, makedirs
from os.path import join, exists, expanduser
from tqdm import tqdm
from sklearn.metrics import log_loss, accuracy_score
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications import xception
from keras.applications import inception_v3
from keras.applications.vgg16 import preprocess_input, decode_predictions
from sklearn.linear_model import LogisticRegression
import cv2

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

INPUT_SIZE = 224
data_dir = '../input'

labels = pd.read_csv(join(data_dir, 'labels.csv'))
sample_submission = pd.read_csv(join(data_dir, 'sample_submission.csv'))
print("Train size:", len(listdir(join(data_dir, 'train'))), len(labels))
print("Test size:", len(listdir(join(data_dir, 'test'))), len(sample_submission))

