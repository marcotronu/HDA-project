"""
- python3
- Various useful function created for the project
"""
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


#############################################################################
def download_images(train = True,val = True,test = True):
    """
    Download images locally on the colab machine to avoid bottlenecks in the training
    Parameters:
        - train/val/test: (bool) if False images for the training/validation/test don't get downloaded
    """
    
    os.chdir('/content/')

    if train:
        os.system("cp '/content/drive/MyDrive/Colab Notebooks/train-dataset-compress.tar.xz' '/content/'")
        file =  tarfile.open('train-dataset-compress.tar.xz')
        file.extractall()
    if val:
        os.system("cp '/content/drive/MyDrive/Colab Notebooks/validation-dataset-compress.tar.xz' '/content/' ")
        file = tarfile.open('validation-dataset-compress.tar.xz')
        file.extractall()
    if test:
        os.system("cp '/content/drive/MyDrive/Colab Notebooks/test-dataset-compress.tar.xz' '/content/' ")
        file = tarfile.open('test-dataset-compress.tar.xz')
        file.extractall() 

    os.chdir('/content/drive/MyDrive/Colab Notebooks') 
#############################################################################


#############################################################################
def train_val_test_gen(): 
    """
    Return train, validation and test with some preprocessing
    """

    "train"
    train_df = pd.read_csv("train.csv")
    train_df['id'] = train_df['id'].apply(lambda x: str(x)+'.png') #we need that for flow_from_dataframe
    train_df['gender_01'] = train_df['male'].apply(lambda x: 1 if (x=='True') or (x==True) else -1)
    train_df.dropna(inplace = True)

    "validation"
    val_df = pd.read_csv("Validation Dataset.csv")
    val_df['id'] = val_df['Image ID'].apply(lambda x: str(x)+'.png')
    val_df['gender_01'] = val_df['male'].apply(lambda x: 1 if (x=='True') or (x==True) else -1)
    val_df.rename(columns={"Bone Age (months)": "boneage"},inplace = True)
    val_df.dropna(inplace = True)

    "test"    
    bone_truth = xlrd.open_workbook('/content/drive/MyDrive/Colab Notebooks/Bone-age-ground-truth.xlsx')
    worksheet = bone_truth.sheet_by_index(0)
    first_row = [] # The row where we stock the name of the column
    for col in range(worksheet.ncols):
        first_row.append( worksheet.cell_value(0,col) )
    # transform the workbook to a list of dictionnary
    data =[]
    for row in range(1, worksheet.nrows):
        elm = {}
        for col in range(worksheet.ncols):
            elm[first_row[col]]=worksheet.cell_value(row,col)
        data.append(elm)
    test_df = pd.DataFrame(data)
    test_df.rename(inplace = True,columns = {'Case ID':'id','Ground truth bone age (months)': 'boneage'})
    test_df['gender_01'] = test_df['Sex'].apply(lambda x: 1 if x == 'M' else -1)
    test_df['id'] = test_df['id'].apply(lambda x: str(int(x)) + '.png')
    test_df.dropna(inplace = True)

    return train_df, val_df, test_df
##############################################################################

##############################################################################
def create_embeddings2(genders, embeddings):
    """
    Take as input a list of 0 and 1, and transforms it 
    in a list of lists where each list has length equal 
    to embeddings.
    Parameters:
        - genders: (list) list of genders
        - embeddings: (int) length of the embedded list 
    """
    new_genders = []
    for gender in genders:
        new_genders.append([float(gender)] * int(embeddings))
    return np.array(new_genders)
##############################################################################

##############################################################################
def create_flow(data_gen,dff,batch_size,img_size,train = True,test=False,color='grayscale'):
    """
    Creates images flow for the training, the validation and the test set.
    Parameters:
        - dff: dataframe from which the function will take the IDs of the images;
        - batch_size: (int) number of images to return for each batch
        - img_size: (int) size of the imgage
        - train: (bool) if true it will return the flow for the training
        - test: (bool) if true it will return the flow for the test
        - if both train and test are False, it will return the flow for the validation '
        - color: (string) either 'grayscale' or 'rgb'
    """
    if train and test==False:
        flow = data_gen.flow_from_dataframe(
                                    dataframe=dff,
                                    directory = '/content/train-dataset-compress/boneage-training-dataset',
                                    x_col = 'id',
                                    y_col = 'boneage',
                                    batch_size=batch_size,
                                    shuffle=True,
                                    class_mode='raw',
                                    target_size=(img_size,img_size),
                                    color_mode = color,
                                    )
    elif train == False and test == False:
        flow = data_gen.flow_from_dataframe(
                            dataframe=dff,
                            directory = '/content/validation-dataset-compress/boneage-validation-dataset',
                            x_col = 'id',
                            y_col = 'boneage',
                            batch_size=batch_size,
                            shuffle=True,
                            class_mode='raw',
                            target_size=(img_size,img_size),
                            color_mode = color,
                        )
    elif train == False and test == True:
        flow = data_gen.flow_from_dataframe(
                            dataframe=dff,
                            directory = '/content/test-dataset/boneage-test-dataset',
                            x_col = 'id',
                            y_col = 'boneage',
                            batch_size=batch_size,
                            shuffle=False,
                            class_mode='raw',
                            target_size=(img_size,img_size),
                            color_mode = color,
                        )
    return flow
##############################################################################

##############################################################################
def get_indices_from_keras_generator(gen, batch_size):
    """
    Given a keras data generator, it returns the indices and the filepaths
    corresponding the current batch. 
    Parameters:
        - gen: the keras gnerator
        - batch_size: (int) the size of the batch 
    Return: both indices and filenames
    """

    idx_left = (gen.batch_index - 1) * batch_size
    idx_right = idx_left + gen.batch_size if idx_left >= 0 else None
    indices = gen.index_array[idx_left:idx_right]
    filenames = [gen.filenames[i] for i in indices]
    return indices, filenames
##############################################################################

##############################################################################
def myCustomGen(data_gen = None,dff = None,train = True,test=False,batch_size=None,img_size=None,embeddings=32,color='grayscale'):
    """
    Creates custom generators which yields, for each image, the gender and the corresponding boneage.
    Parameters:
        - data_gen: keras data generator
        - dff: dataframe from which the gen takes the images IDs
        - train/test (bool) same as create flow
        - batch_size: (int) size of the batch
        - img_size: (int) dimension of the image --> (N,N)
        - embeddings: (int) length of the gender embeddings
        - color: either 'grayscale' or 'rgb'
    """
    flow = create_flow(data_gen,dff,batch_size,img_size,train,test,color) 
    for x, y in flow:
        indices, filenames = get_indices_from_keras_generator(flow,batch_size)
        # boneages = my_val.loc[my_val['id'].isin(filenames)].values
        # boneages = reduce(pd.DataFrame.append, map(lambda i: dff[dff.id == i], filenames)).boneage.values
        genders = reduce(pd.DataFrame.append, map(lambda i: dff[dff.id == i], filenames)).gender_01.values
        genders = create_embeddings2(genders,embeddings)
        # if next_print:
        #   print(boneages,y)
        #   next_print = True

        if len(x) != len(genders):
            yield [x,genders[-len(y):]],y
        else:
            yield [x,genders],y
##############################################################################

##############################################################################
def schedule(epoch,lr):
    """
    Custom learning rate scheduler
    Parameters:
        - epoch (int)
        - lr (float): learning rate
    """
    return lr*0.95 if int(epoch) != 0 else lr
##############################################################################

##############################################################################
def mae_months(y_true,y_pred):
    """
    Return Mean Absolute Error in months (used in case boneages are normalized)
    """
    return mean_absolute_error(mu+sigma*y_true,mu+sigma*y_pred)
##############################################################################

##############################################################################
class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
##############################################################################

##############################################################################
def CreateCallbacks(ckp = None,training_log = None):
    """
    Create callbacks for training
    Parameters:
        - ckp: (string) path in which the weights will be stored (.h5 format)
        - training_log: (string) path in which the training log will be stored each epoch
        - batch_size: (int) size of the batch
    """
    if ckp == None:
        raise Exception('Please select a valid checkpoint path!')   
    if training_log == None:
        raise Exception('Please select a valid training log path!')
    
    early_stopping = EarlyStopping(monitor='val_loss',
                                min_delta=0,
                                patience=10,
                                verbose=0,
                                mode='auto')
    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', verbose = 1, factor=0.7, patience=2,mode='min', min_delta=0.0001, cooldown=2, min_lr=0.000001)

    my_cps = ModelCheckpoint(ckp,monitor='val_loss',mode='min',save_best_only=True)
    csv_logger = tf.keras.callbacks.CSVLogger(training_log,append=True) #append True is mandatory otherwise it overwrites everytime
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(schedule,verbose=1)
    time_callback = TimeHistory()
    return [early_stopping, reduceLROnPlat, my_cps,csv_logger]
##############################################################################

##############################################################################
def MAE(array1, array2):
    """
    Calculates Mean Absolute Error between the arrays array1 and array2
    """
    return np.mean(np.abs(array1 - array2))
##############################################################################

##############################################################################
def get_predictions(model,test_df,preprocess = True,embeddings = 1, iters = 10,width_shift_range = 0.25,height_shift_range=0.25):
    """
    From a model (passed as input) computes the predictions on the test set
    For each image, the prediction is done 10 times.
    Return a dataframe in which each column is an iteration on the Test Set
    Parameters:
        - model: tensorflow.keras model through which the predictions are made
        - preprocess: (bool) if true the images are preproceessed using the inceptionV3 preprocessing function( scaling between -1 and 1)
        - embeddings: (int) size of the embedded genders
        - iters: (int) how many iterations between which you do the averaging for each model
        - width_shift_range: (float)
        - height_shift_range: (float)
    """
    model_preds = {}
    if preprocess:
        preprocessing = preprocess_input
    else:
        preprocessing = None

    for i in range(iters):
        print('Iteration: {}'.format(i))
        test_data_gen = ImageDataGenerator(preprocessing_function = preprocessing,width_shift_range=width_shift_range, height_shift_range=height_shift_range, horizontal_flip = True)
        test_input =  myCustomGen(data_gen = test_data_gen, dff = test_df, train=False,batch_size = 20, img_size = 500, embeddings = embeddings,test = True)

        new_preds = model.predict(test_input,steps = len(test_df)/20,verbose=1) 
        new_preds = [pred[0] for pred in new_preds]
        model_preds[i] = new_preds
    preds = pd.DataFrame(model_preds)
    preds.columns = ['iter{}'.format(i) for i in np.arange(iters)]
    return preds
##############################################################################


##############################################################################

##############################################################################

##############################################################################

##############################################################################


