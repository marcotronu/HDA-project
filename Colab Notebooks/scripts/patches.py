import cv2
import tensorflow as tf 
import os
import tarfile 
import pandas as pd
import xlrd
from functools import reduce
import numpy as np
import time
import cv2
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import random
import keras
from glob import glob
import re
import pydot
import scipy.misc

from tensorflow.keras import layers, Input
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from IPython.display import SVG
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from tensorflow.keras.metrics import mean_absolute_error
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from tensorflow.keras import Sequential
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc, accuracy_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from tensorflow.python.client import device_lib
from keras.applications.inception_v3 import preprocess_input

def create_patches(img):
    if img.shape != (560,560):
        img = cv2.resize(img, (560,560))  
    tensor_of_49_patches = np.array([img[int(i/7*(448 - 112)):int(224 + i/7*(448 - 112)),int(j/7*(448 - 112)):int(224 + j/7*(448 - 112))] for i in range(7) for j in range(7)])
    return tensor_of_49_patches



def train_subset(n=6000):
    tr=[]
    for img in os.listdir('./train-dataset-compress'):
        tr.append(img)
    tr_random = random.sample(tr, n)
    data5=[]
    for id_im in tr_random:
        for i in range(49):
            data5.append(str(id_im[:-4])+'_'+'{:02d}'.format(i+1)+'.png')
    traindf = traindf[traindf['id'].isin(data5)]
    return traindf




def download_train_patches():
    train_path='./train-dataset-compress'
    train_patch_path='./proj/train_patch'
    for img in tr_random:
        image=cv2.imread(train_path+'/'+str(img))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
        cl1 = clahe.apply(gray)
        patches = create_patches(cl1)
        for i in range(len(patches)):
            #cv2.imshow('img',patches[i])
            cv2.imwrite(train_patch_path+'/'+str(img[:-4])+'_'+'{:02d}'.format(i+1)+'.png', patches[i])
            cv2.destroyAllWindows()




def download_test_patches():
    test_path= './test-dataset-compress'
    test_patch_path = './proj/test_patch'

    for img in os.listdir(test_path):
        image= cv2.imread(test_path+'/'+str(img))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
        cl1 = clahe.apply(gray)
        #cl1 = cl1/255
        patches = create_patches(cl1)
        for i in range(len(patches)):
            #cv2.imshow('img',patches[i])
            cv2.imwrite(test_patch_path+'/'+str(img[:-4])+'_'+'{:02d}'.format(i+1)+'.png', patches[i])
            cv2.destroyAllWindows()