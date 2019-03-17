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
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Reshape
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
import sklearn
import datetime
from tqdm import tqdm
import fnmatch
from time import gmtime, strftime
from multiprocessing.dummy import Pool as ThreadPool
import itertools
from PIL import Image, ImageFilter
from scipy.ndimage.filters import gaussian_filter

#detect face in image
def DetectFace(cascade, image, scale_factor=1.1):
    #convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)          
    #find face(s) in image
    faces = cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=3, minSize=(110,110))
    
    outputFace = None
    #crop image to face region
    for face in faces:
        if face is None:
            print('Face not detected in {}'.format(image))
        else:
            x,y,w,h = face
            outputFace = image[y:y+h, x:x+w]
            return outputFace, True
    return outputFace, False

def directorySearch(directory, label, dataName, dataAugmentation=False):
    print('Started directory search of {} at {}'.format(dataName, str(datetime.datetime.now())))
    x, y = [], []
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    time = strftime("%Y-%m-%d--%H-%M-%S", gmtime())
    if label is 0:
#        fileBadImages = open('{0}-BadImagesNoPain.txt'.format(time), 'w+')
#        fileBadFaces = open('{0}-BadFacesNoPain.txt'.format(time), 'w+')
        pass
    elif label is 1:
#        fileBadImages = open('{0}-BadImagesPain.txt'.format(time), 'w+')
#        fileBadFaces = open('{0}-BadFacesPain.txt'.format(time), 'w+')
        pass
    else:
        print('Error: label should be 0 or 1')
        return
    countBadImages = 0
    countBadFaces = 0
#    for file in tqdm(sklearn.utils.shuffle(os.listdir(directory))[0:10]):
    for file in tqdm(sklearn.utils.shuffle(os.listdir(directory))):
        if file.endswith('.jpg'):
            path = os.path.join(directory, file)
            img = cv2.imread(path)
            if img is None:
#                fileBadImages.write(file + '\n')
                countBadImages += 1
                pass
            else:
                face, faceDetected = DetectFace(face_cascade, img)
                if faceDetected:
                    faceResized = cv2.resize(face, (128, 128), interpolation = cv2.INTER_AREA)
#                    print(faceResized.shape)
#                    cv2.imwrite("Original.jpg", faceResized)
                    x.append(faceResized)
                    y.append(label)
                    
                    if dataAugmentation:
                        # augmented data: mirror (vertical flip)
                        faceMirror = cv2.flip(faceResized, 1)
    #                    print(faceMirror.shape)
    #                    cv2.imwrite("Mirror.jpg", faceMirror)
                        x.append(faceMirror)
                        y.append(label)
                    
                        # augmented data: Gaussian Blur
                        faceBlur = gaussian_filter(faceResized, sigma=0.5)
    #                    print(faceBlur.shape)
    #                    cv2.imwrite("Blur.jpg", faceBlur)
                        x.append(faceBlur)
                        y.append(label)
    #                    return
                else:
#                    fileBadFaces.write(file + '\n')
                    countBadFaces += 1
    if countBadImages > 0:
        print('Bad images count: {}'.format(countBadImages))
    if countBadFaces > 0:
        print('Bad faces count: {}'.format(countBadFaces))
    print('Ended directory search of {} at {}'.format(dataName, str(datetime.datetime.now())))
    return x, y

def verifyLength(list1, list2, list1Name, list2Name):
    if len(list1) != len(list2):
        print('Error: {0} length does not equal {1} length'.format(list1Name, list2Name))

# preprocessing
def readImages(pathData):
    # get test data
    pathTestNoPain = '{}Testing/No_pain/'.format(pathData)
    x_TestNoPain, y_TestNoPain = directorySearch(pathTestNoPain, 0, 'Test No Pain')
    verifyLength(x_TestNoPain, y_TestNoPain, 'x_TestNoPain', 'y_TestNoPain')
    pathTestPain = '{}Testing/Pain/'.format(pathData)
    x_TestPain, y_TestPain = directorySearch(pathTestPain, 1, 'Test Pain')
    verifyLength(x_TestPain, y_TestPain, 'x_TestPain', 'y_TestPain')

    # get train data
    pathTrainNoPain = '{}Training/No_pain/'.format(pathData)
    x_TrainNoPain, y_TrainNoPain = directorySearch(pathTrainNoPain, 0, 'Train No Pain', dataAugmentation=True)
    verifyLength(x_TrainNoPain, y_TrainNoPain, 'x_TrainNoPain', 'y_TrainNoPain')
    pathTrainPain = '{}Training/Pain/'.format(pathData)
    x_TrainPain, y_TrainPain = directorySearch(pathTrainPain, 1, 'Train Pain', dataAugmentation=True)
    verifyLength(x_TrainPain, y_TrainPain, 'x_TrainPain', 'y_TrainPain')
    # rebalance classes for training data
#    print('Training pain shape\nx: {}\ny: {}'.format(np.asarray(x_TrainPain).shape, np.asarray(y_TrainPain).shape))
#    print('Original training no pain shape\nx: {}\ny: {}'.format(np.asarray(x_TrainNoPain).shape, np.asarray(y_TrainNoPain).shape))
    x_TrainNoPain, y_TrainNoPain = sklearn.utils.shuffle(x_TrainNoPain, y_TrainNoPain)
    x_TrainNoPain, y_TrainNoPain = x_TrainNoPain[0:-11376], y_TrainNoPain[0:-11376]
#    print('New training pain shape\nx: {}\ny: {}'.format(np.asarray(x_TrainNoPain).shape, np.asarray(y_TrainNoPain).shape))

    # get val data
    pathValNoPain = '{}Validaiton/No_pain/'.format(pathData)
    x_ValNoPain, y_ValNoPain = directorySearch(pathValNoPain, 0, 'Val NoPain')
    verifyLength(x_ValNoPain, y_ValNoPain, 'x_ValNoPain', 'y_ValNoPain')
    pathValPain = '{}Validaiton/Pain/'.format(pathData)
    x_ValPain, y_ValPain = directorySearch(pathValPain, 1, 'Val Pain')
    verifyLength(x_ValPain, y_ValPain, 'x_ValPain', 'y_ValPain')

    # setup testing data
    test_x = np.asarray(x_TestNoPain + x_TestPain)
    test_y = np.asarray(y_TestNoPain + y_TestPain)
#    test_x, test_y = sklearn.utils.shuffle(test_x, test_y)
    
    # setup training data
    train_x = np.asarray(x_TrainNoPain + x_TrainPain)
    train_y = np.asarray(y_TrainNoPain + y_TrainPain)
#    train_x, train_y = sklearn.utils.shuffle(train_x, train_y)

    # setup validation data
    val_x = np.asarray(x_ValNoPain + x_ValPain)
    val_y = np.asarray(y_ValNoPain + y_ValPain)
#    val_x, val_y = sklearn.utils.shuffle(val_x, val_y)
    
    # normalize x data
    test_x = test_x.astype('float32')/255.0
    train_x = train_x.astype('float32')/255.0
    val_x = val_x.astype('float32')/255.0
    
    return test_x, test_y, train_x, train_y, val_x, val_y

def find_files(base, pattern):
    '''Return list of files matching pattern in base folder.'''
    return [n for n in fnmatch.filter(os.listdir(base), pattern) if
        os.path.isfile(os.path.join(base, n))]

def buildModel(pathBase):
    # create model
#    model = keras.models.Sequential()
#    model = keras.applications.nasnet.NASNetLarge(weights = "imagenet", include_top=False, input_shape=(128, 128, 3))
    model = keras.applications.Xception(weights = "imagenet", include_top=False, input_shape=(128, 128, 3))
#    from nasnet import NASNetLarge, NASNetMobile
#    model = NASNetLarge(input_shape=(128, 128, 3), dropout=0.5)
#    with tf.device('/cpu:0'):
#        model = Xception(weights=None, input_shape=(256, 256, 3), classes=2)

#    # 2 layers of convolution
#    model.add(keras.layers.Conv2D(64, 3, activation='relu', input_shape=(128,128,3)))
#    model.add(keras.layers.BatchNormalization())
#    # dropout
##    model.add(keras.layers.Dropout(0.50))
#    model.add(keras.layers.Conv2D(64, 3, activation='relu'))
#    model.add(keras.layers.BatchNormalization())
#    # dropout
##    model.add(keras.layers.Dropout(0.25))
#    
#    # max pooling
#    model.add(keras.layers.MaxPooling2D())
#    
#    # 2 layers of convolution
#    model.add(keras.layers.Conv2D(128, 3, activation='relu'))
#    model.add(keras.layers.BatchNormalization())
#    model.add(keras.layers.Conv2D(128, 3, activation='relu'))
#    model.add(keras.layers.BatchNormalization())
#    
#    # max pooling
#    model.add(keras.layers.MaxPooling2D())
#    
#    # 3 layers of convolution
#    model.add(keras.layers.Conv2D(256, 3, activation='relu'))
#    model.add(keras.layers.BatchNormalization())
#    model.add(keras.layers.Conv2D(256, 3, activation='relu'))
#    model.add(keras.layers.BatchNormalization())
#    model.add(keras.layers.Conv2D(256, 3, activation='relu'))
#    model.add(keras.layers.BatchNormalization())
#
#    # max pooling
#    model.add(keras.layers.MaxPooling2D())
    
#    # flatten
#    model.add(keras.layers.Flatten())
#    
#    # fully connected layer
#    model.add(keras.layers.Dense(2, activation='relu'))
#    
#    # dropout
##    model.add(keras.layers.Dropout(0.5))
#    
#    # final dense layer
#    model.add(keras.layers.Dense(1, activation='sigmoid', 
##                                 kernel_regularizer=regularizers.l2(0.01), 
##                                 activity_regularizer=regularizers.l1(0.01)
#                                 ))    
    # resume from checkpoint
#    savedModelFiles = find_files(pathBase, '2019-02-07--*.hdf5')
#    if len(savedModelFiles) > 0:
#        if len(savedModelFiles) > 1:
#            print('Error: There are multiple saved model files.')
#            return
#        print("Resumed model's weights from {}".format(savedModelFiles[-1]))
#        # load weights
#        model.load_weights(os.path.join(pathBase, savedModelFiles[-1]))

#    print('number of layers: {}'.format(len(model.layers)))
#    for layer in model.layers[:14]:
#    for layer in model.layers:
#        layer.trainable=False
##    #Adding custom Layers 
##    model.add(keras.layers.Reshape(1,2))
    x = model.output
#    x = Reshape(1,2)(x)
    x = Flatten()(x)
#    x = Dense(1024, activation="relu")(x)
#    x = Dropout(0.5)(x)
#    x = Dense(1024, activation="relu")(x)
#    x = Dropout(0.5)(x)
    predictions = Dense(2, activation="softmax")(x)
##    # creating the final model 
    model = Model(inputs = model.input, outputs = predictions)
    
    # multiple GPUs
    model = multi_gpu_model(model, gpus=16)
    # compile
#    model.compile(optimizer=keras.optimizers.Adam(lr=0.00001), loss=keras.losses.binary_crossentropy, metrics=['acc'])
    model.compile(loss = "sparse_categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])
    
    return model

if __name__ == "__main__":
    pathBase = 'pain_classification/'
    
    print('Image reading started at {}'.format(str(datetime.datetime.now())))
    test_x, test_y, train_x, train_y, val_x, val_y = readImages(pathBase)
    print('Image reading finished at {}'.format(str(datetime.datetime.now())))

#    print('Class balance started at {}'.format(str(datetime.datetime.now())))
#    unique, counts = np.unique(test_y, return_counts=True)
#    print('test_y: {}'.format(dict(zip(unique, counts))))
#    unique, counts = np.unique(train_y, return_counts=True)
#    print('train_y: {}'.format(dict(zip(unique, counts))))
#    unique, counts = np.unique(val_y, return_counts=True)
#    print('val_y: {}'.format(dict(zip(unique, counts))))
#    print('Class balance finished at {}'.format(str(datetime.datetime.now())))
# original split (with augmentation)
#    test_y: {0: 1342, 1: 2536}
#    train_y: {0: 34317, 1: 22941}
#    val_y: {0: 2912, 1: 2331}

    print('Model building started at {}'.format(str(datetime.datetime.now())))
    model = buildModel(pathBase)
    print('Model building finished at {}'.format(str(datetime.datetime.now())))
    
    print('Model evaluation started at {}'.format(str(datetime.datetime.now())))
    # fit model to data
    time = strftime("%Y-%m-%d--%H-%M-%S", gmtime())
    checkpoint = ModelCheckpoint('{0}{1}_{{epoch:02d}}-{{val_acc:.2f}}.hdf5'.format(pathBase, time), 
								 monitor='val_loss', verbose=1, save_best_only=True, mode='max')
    earlyStop = EarlyStopping('val_loss',0.001,5)
    callbacks_list = [checkpoint, earlyStop]
    model.fit(x=train_x, y=train_y, batch_size=64, epochs=10, verbose=2, 
              callbacks=callbacks_list,
              validation_data=(val_x, val_y),
              initial_epoch=0)    
    print(model.evaluate(test_x, test_y))
    print('Model evaluation finished at {}'.format(str(datetime.datetime.now())))