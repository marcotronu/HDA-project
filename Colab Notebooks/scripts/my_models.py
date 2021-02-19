"""
- python3
- Some of the models created for the project
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
import zipfile 

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
from tensorflow.keras.applications.inception_v3 import preprocess_input


#######################################################################
def create_EffNet(img_size = None,embeddings = 32):
    """
    Create efficent Net (B4)
    Parameters:
        - img_size (int) size of the image
        - embeddings: (int) length of the embedded gender
    """
    efficient_net = tf.keras.applications.EfficientNetB4(input_shape = (img_size,img_size, 1), include_top = False,
                                                    weights=None, pooling = 'avg')

    for layer in efficient_net.layers:
        layer.trainable = True

    # efficient_out = BatchNormalization()(efficient_net.output)
    efficient_out = Dropout(0.2)(efficient_net.output)
    
    #Gender embeddings
    gender = Input(shape=(embeddings,))

    dense_out = Dense(32,activation='sigmoid')(gender)

    #Concatenate with and Dense outputs
    concatenated = tf.keras.layers.Concatenate(axis = -1)([efficient_out, dense_out])

    #Add Dense 1000 layers
    out = Dense(1000,name='First-Dense-1000',activation = tf.keras.activations.swish)(concatenated) #we try with sigmoid as sigmoid(0) != 0
    # out = LeakyReLU(alpha=0.01)(out)
    out = Dropout(0.2)(out)

    #Add Dense 1000 layers
    out = Dense(1000,name='Second-Dense-1000', activation = tf.keras.activations.swish)(out)
    # out = LeakyReLU(alpha=0.01)(out)
    out = Dropout(0.2)(out)

    #We're considering the problem as a multiclass problem, with each month being a class (max age is 228 and min is 1)
    out = Dense(1, activation='linear', name = 'final-layer')(out)

    #Putting it all together:
    # my_model = Model(inputs=[model_v3.input,gender],outputs=out)
    my_model = Model(inputs = [efficient_net.input,gender],outputs = out)

    """
    Add l2 regularization
    """
    alpha  = 0.00001
    regularizer = tf.keras.regularizers.l2(alpha)
    for layer in my_model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l2(alpha)(layer.kernel))
        if hasattr(layer, 'bias_regularizer') and layer.use_bias:
            layer.add_loss(lambda: tf.keras.regularizers.l2(alpha)(layer.bias))  
        #Parameter
    # optim = tf.keras.optimizers.Adam(learning_rate = 0.0001)
    optim = tf.keras.optimizers.RMSprop(learning_rate=0.000001,momentum=0.8)
    # optim_acc = runai.ga.keras.optimizers.Optimizer(optim,6)
    mse = tf.keras.losses.MeanSquaredError()
    rmse = tf.keras.metrics.RootMeanSquaredError()
    mae = tf.keras.metrics.MeanAbsoluteError()
    my_model.compile(loss=mse,optimizer=optim,metrics=[rmse,mae])

    return my_model
#######################################################################


#######################################################################
def create_EffNet_noisy(img_size = None,embeddings = 32, lr = 0.0001):
    """
    Create efficent Net (B5) with noisy weights
    Parameters:
        - img_size (int) size of the image
        - embeddings: (int) length of the embedded gender
    """
    efficient_net = tf.keras.applications.EfficientNetB5(input_shape = (img_size,img_size, 3), include_top = False,
                                                    weights=None, pooling = 'avg')


    with zipfile.ZipFile('/content/drive/MyDrive/Colab Notebooks/noisy/efficientnet-b5_noisy-student_notop.h5.zip','r') as zip_ref:
        zip_ref.extractall('/content/drive/MyDrive/Colab Notebooks/noisy/')
    efficient_net.load_weights('/content/drive/MyDrive/Colab Notebooks/noisy/efficientnet-b5_noisy-student_notop.h5', by_name = True)

    """
    Set this to True when unfreezing the model (2nd stage of transfer learning)
    """
    efficient_net.trainable = True #set this to true when unfreezing
    """
    Uncomment this when unfreezing the model (2nd stage of transfer learning)
    """
    for layer in efficient_net.layers:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    # efficient_out = BatchNormalization()(efficient_net.output)
    efficient_out = Dropout(0.2)(efficient_net.output)
    
    #Gender embeddings
    gender = Input(shape=(embeddings,))

    dense_out = Dense(32,activation='sigmoid')(gender)

    #Concatenate with and Dense outputs
    concatenated = tf.keras.layers.Concatenate(axis = -1)([efficient_out, dense_out])

    #Add Dense 1000 layers
    out = Dense(1000,name='First-Dense-1000',activation = tf.keras.activations.swish)(concatenated) #we try with sigmoid as sigmoid(0) != 0
    # out = LeakyReLU(alpha=0.01)(out)
    out = Dropout(0.2)(out)

    #Add Dense 1000 layers
    out = Dense(1000,name='Second-Dense-1000', activation = tf.keras.activations.swish)(out)
    # out = LeakyReLU(alpha=0.01)(out)
    out = Dropout(0.2)(out)

    #We're considering the problem as a multiclass problem, with each month being a class (max age is 228 and min is 1)
    out = Dense(1, activation='linear', name = 'final-layer')(out)

    #Putting it all together:
    # my_model = Model(inputs=[model_v3.input,gender],outputs=out)
    my_model = Model(inputs = [efficient_net.input,gender],outputs = out)

    """
    Add l2 regularization
    """
    alpha  = 0.00001
    regularizer = tf.keras.regularizers.l2(alpha)
    for layer in my_model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l2(alpha)(layer.kernel))
        if hasattr(layer, 'bias_regularizer') and layer.use_bias:
            layer.add_loss(lambda: tf.keras.regularizers.l2(alpha)(layer.bias))  
        #Parameter
    # optim = tf.keras.optimizers.Adam(learning_rate = 0.0001)
    optim = tf.keras.optimizers.RMSprop(learning_rate=lr,momentum=0.8)
    # optim_acc = runai.ga.keras.optimizers.Optimizer(optim,6)
    mse = tf.keras.losses.MeanSquaredError()
    rmse = tf.keras.metrics.RootMeanSquaredError()
    mae = tf.keras.metrics.MeanAbsoluteError()
    my_model.compile(loss=tf.keras.losses.MeanAbsoluteError(),optimizer=optim,metrics=[rmse,mae])

    return my_model
#######################################################################



#######################################################################
def create_InceptionRes(img_size = None,embeddings = 32):
    efficient_net = tf.keras.applications.InceptionResNetV2(input_shape = (img_size,img_size, 1), include_top = False,
                                                    weights=None, pooling = 'avg')

    for layer in efficient_net.layers:
        layer.trainable = True

    # efficient_out = BatchNormalization()(efficient_net.output)
    efficient_out = Dropout(0.2)(efficient_net.output)

    #Gender embeddings
    gender = Input(shape=(embeddings,))

    dense_out = Dense(32,activation='linear')(gender)

    #Concatenate InceptionV3 and Dense outputs
    concatenated = tf.keras.layers.Concatenate(axis = -1)([efficient_out, dense_out])

    #Add Dense 1000 layers
    out = Dense(1000,name='First-Dense-1000',activation = tf.keras.activations.swish)(concatenated) #we try with sigmoid as sigmoid(0) != 0
    # out = LeakyReLU(alpha=0.01)(out)
    out = Dropout(0.2)(out)

    #Add Dense 1000 layers
    out = Dense(1000,name='Second-Dense-1000', activation = tf.keras.activations.swish)(out)
    # out = LeakyReLU(alpha=0.01)(out)
    out = Dropout(0.2)(out)

    #We're considering the problem as a multiclass problem, with each month being a class (max age is 228 and min is 1)
    out = Dense(1, activation='linear', name = 'final-layer')(out)

    #Putting it all together:
    # my_model = Model(inputs=[model_v3.input,gender],outputs=out)
    my_model = Model(inputs = [efficient_net.input,gender],outputs = out)

    alpha  = 0.0001
    regularizer = tf.keras.regularizers.l2(alpha)
    for layer in my_model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l2(alpha)(layer.kernel))
        if hasattr(layer, 'bias_regularizer') and layer.use_bias:
            layer.add_loss(lambda: tf.keras.regularizers.l2(alpha)(layer.bias))  
    #Parameter
    # optim = tf.keras.optimizers.Adam(learning_rate = 0.0001)
    optim = tf.keras.optimizers.RMSprop(learning_rate=0.0001,momentum=0.8)
    # optim_acc = runai.ga.keras.optimizers.Optimizer(optim,6)
    mse = tf.keras.losses.MeanSquaredError()
    rmse = tf.keras.metrics.RootMeanSquaredError()
    mae = tf.keras.metrics.MeanAbsoluteError()
    my_model.compile(loss=tf.keras.losses.MeanAbsoluteError(),optimizer=optim,metrics=[rmse,mae])

    return my_model
#######################################################################

#######################################################################
def create_InceptionRes_v2(img_size = None,embeddings = 32):
    efficient_net = tf.keras.applications.InceptionResNetV2(input_shape = (img_size,img_size, 1), include_top = False,
                                                    weights=None, pooling = 'avg')

    for layer in efficient_net.layers:
        layer.trainable = True

    # efficient_out = BatchNormalization()(efficient_net.output)
    efficient_out = Dropout(0.2)(efficient_net.output)

    #Gender embeddings
    gender = Input(shape=(embeddings,))

    # dense_out = Dense(32,activation='linear')(gender)

    #Concatenate InceptionV3 and Dense outputs
    concatenated = tf.keras.layers.Concatenate(axis = -1)([efficient_out, gender])

    #Add Dense 1000 layers
    out = Dense(1000,name='First-Dense-1000',activation = tf.keras.activations.sigmoid)(concatenated) #we try with sigmoid as sigmoid(0) != 0
    # out = LeakyReLU(alpha=0.01)(out)
    out = Dropout(0.2)(out)

    #Add Dense 1000 layers
    out = Dense(1000,name='Second-Dense-1000', activation = tf.keras.activations.sigmoid)(out)
    # out = LeakyReLU(alpha=0.01)(out)
    out = Dropout(0.2)(out)

    #We're considering the problem as a multiclass problem, with each month being a class (max age is 228 and min is 1)
    out = Dense(1, activation='linear', name = 'final-layer')(out)

    #Putting it all together:
    # my_model = Model(inputs=[model_v3.input,gender],outputs=out)
    my_model = Model(inputs = [efficient_net.input,gender],outputs = out)

    alpha  = 0.00001
    regularizer = tf.keras.regularizers.l2(alpha)
    for layer in my_model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l2(alpha)(layer.kernel))
        if hasattr(layer, 'bias_regularizer') and layer.use_bias:
            layer.add_loss(lambda: tf.keras.regularizers.l2(alpha)(layer.bias))  
    #Parameter
    optim = tf.keras.optimizers.Adam(learning_rate = 0.0001)
    # optim = tf.keras.optimizers.RMSprop(learning_rate=0.0001,momentum=0.8)
    # optim_acc = runai.ga.keras.optimizers.Optimizer(optim,6)
    mse = tf.keras.losses.MeanSquaredError()
    rmse = tf.keras.metrics.RootMeanSquaredError()
    mae = tf.keras.metrics.MeanAbsoluteError()
    my_model.compile(loss=tf.keras.losses.MeanAbsoluteError(),optimizer=optim,metrics=[rmse,mae])

    return my_model
#######################################################################

#######################################################################
def create_inception(img_size = None,embeddings = 32):

    model_v3 = tf.keras.applications.InceptionV3(input_shape = (img_size,img_size, 1), include_top = False,
                                                weights=None, pooling = 'avg')
    for layer in model_v3.layers:
        layer.trainable = True

    model_v3_out = Dropout(0.2)(model_v3.output)


    #Gender embeddings
    gender = Input(shape=(embeddings,))

    #Concatenate InceptionV3 and Dense outputs
    concatenated_v3_dense = tf.keras.layers.Concatenate(axis = -1)([model_v3_out, gender])

    #Add Dense 1000 layers
    out = Dense(1000,name='First-Dense-1000')(concatenated_v3_dense)
    out = LeakyReLU(alpha=0.01)(out)
    out = Dropout(0.2)(out)

    #Add Dense 1000 layers
    out = Dense(1000,name='Second-Dense-1000')(out)
    out = LeakyReLU(alpha=0.01)(out)
    out = Dropout(0.2)(out)

    #We're considering the problem as a multiclass problem, with each month being a class (max age is 228 and min is 1)
    out = Dense(1, activation='linear', name = 'final-layer')(out)

    #Putting it all together:
    # my_model = Model(inputs=[model_v3.input,gender],outputs=out)
    my_model = Model(inputs = [model_v3.input,gender],outputs = out)
    #Parameters
    # optim = tf.keras.optimizers.SGD(0.0001)
    optim = tf.keras.optimizers.Adam(0.0001)

    mse = tf.keras.losses.MeanSquaredError()
    rmse = tf.keras.metrics.RootMeanSquaredError()
    mae = tf.keras.metrics.MeanAbsoluteError()
    my_model.compile(loss=mse,optimizer=optim,metrics=[rmse,mae])

    return my_model
#######################################################################



###########################################################################
def create_InceptionRes_patches(img_size = None,embeddings = 32):
  efficient_net = tf.keras.applications.InceptionResNetV2(input_shape = (img_size,img_size, 1), include_top = False,
                                                   weights=None, pooling = 'avg')

  for layer in efficient_net.layers:
    layer.trainable = True

  # efficient_out = BatchNormalization()(efficient_net.output)
  efficient_out = Dropout(0.2)(efficient_net.output)
  
  #Gender embeddings
  gender = Input(shape=(embeddings,))

  dense_out = Dense(32,activation='linear')(gender)

  #Concatenate InceptionV3 and Dense outputs
  concatenated = tf.keras.layers.Concatenate(axis = -1)([efficient_out, dense_out])

  #Add Dense 1000 layers
  out = Dense(1000,name='First-Dense-1000',activation = tf.keras.activations.sigmoid)(concatenated) #we try with sigmoid as sigmoid(0) != 0
  # out = LeakyReLU(alpha=0.01)(out)
  out = Dropout(0.2)(out)

  #Add Dense 1000 layers
  out = Dense(1000,name='Second-Dense-1000', activation = tf.keras.activations.swish)(out)
  # out = LeakyReLU(alpha=0.01)(out)
  out = Dropout(0.2)(out)

  #We're considering the problem as a multiclass problem, with each month being a class (max age is 228 and min is 1)
  out = Dense(1, activation='linear', name = 'final-layer')(out)

  #Putting it all together:
  # my_model = Model(inputs=[model_v3.input,gender],outputs=out)
  my_model = Model(inputs = [efficient_net.input,gender],outputs = out)

  alpha  = 0.0001
  regularizer = tf.keras.regularizers.l2(alpha)
  for layer in my_model.layers:
      if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
          layer.add_loss(lambda: tf.keras.regularizers.l2(alpha)(layer.kernel))
      if hasattr(layer, 'bias_regularizer') and layer.use_bias:
          layer.add_loss(lambda: tf.keras.regularizers.l2(alpha)(layer.bias))  
    #Parameter
  # optim = tf.keras.optimizers.Adam(learning_rate = 0.0001)
  optim = tf.keras.optimizers.RMSprop(learning_rate=0.0001,momentum=0.8)
  # optim_acc = runai.ga.keras.optimizers.Optimizer(optim,6)
  mse = tf.keras.losses.MeanSquaredError()
  rmse = tf.keras.metrics.RootMeanSquaredError()
  mae = tf.keras.metrics.MeanAbsoluteError()
  my_model.compile(loss=mse,optimizer=optim,metrics=[rmse,mae])

  return my_model
  ##############################################################