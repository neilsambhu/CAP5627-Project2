# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 16:41:58 2019
@author: neils
"""

import os
import numpy as np
import cv2
import keras
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras import applications
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Reshape, Conv2D, LSTM, Input, Lambda, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import TimeDistributed
import sklearn
from sklearn.metrics import confusion_matrix
import datetime
from tqdm import tqdm
import fnmatch
from time import gmtime, strftime
from multiprocessing.dummy import Pool as ThreadPool
import itertools
from PIL import Image, ImageFilter
from scipy.ndimage.filters import gaussian_filter
from keras.preprocessing.image import ImageDataGenerator

def DataAugmentation(faceResized, label, fileName):
    x, y = [], []
    
    # data augmentation through ImageDataGenerator
    img_gen = ImageDataGenerator()
    # transformation types
    for theta_input in [-15,-10,-5,0,5,10,15]:
        for flip_horizontal_input in [False, True]:
            for flip_vertical_input in [False, True]:
                for channel_shift_intencity_input in [-100,0,100]:
                    faceTransform = img_gen.apply_transform(faceResized, 
                                            {'theta':theta_input,
                                             'flip_horizontal':flip_horizontal_input,
                                             'flip_vertical':flip_vertical_input,
                                             'channel_shift_intencity':channel_shift_intencity_input})
                    cv2.imwrite('{}_{}_{}_{}_{}.jpg'.format(fileName, 
                                theta_input, 
                                flip_horizontal_input,
                                channel_shift_intencity_input), faceTransform)
                    x.append(faceTransform)
                    y.append(label)
    
    # data augmetnation through OpenCV
#     augmented data: mirror (vertical flip)
    faceMirror = cv2.flip(faceResized, 1)
    cv2.imwrite('{}_Mirror.jpg'.format(fileName), faceMirror)
    x.append(faceMirror)
    y.append(label)

#     augmented data: Gaussian Blur
    faceBlur = gaussian_filter(faceResized, sigma=0.5)
    cv2.imwrite('{}_Blur.jpg'.format(fileName), faceBlur)
    x.append(faceBlur)
    y.append(label)
    
#     augmented data: mirror and Gaussian Blur
    faceBlurMirror = gaussian_filter(faceMirror, sigma=0.5)
    cv2.imwrite('{}_BlurMirror.jpg'.format(fileName), faceBlurMirror)
    x.append(faceBlurMirror)
    y.append(label)
    
    return x, y