# A multi classes image classifier, based on convolutional neural network using Keras and Tensorflow. 
# A multi-label classifier (having one fully-connected layer at the end), with multi-classification (32 classes, in this instance)
# Largely copied from the code https://github.com/kallooa/MSDA_Capstone_Final/tree/master/3_Model_Training/Tile_Level_Model_Training
# classifying 32 cancer types (thumbnail images of WSI slides) downloaded from digitalslidearchive.emory.edu
# Will implement/include data manipulating functionalities based on Girder (https://girder.readthedocs.io/en/latest/)
# Used Keras.ImageDataGenerator for Training/Validation data augmentation and the augmented images are flown from respective directory
# Environment: A docker container having Keras, TensorFlow, Python-2 with GPU based execution

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Convolution2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.callbacks import Callback
from keras.optimizers import adam
from keras.models import Model
from keras.applications.vgg19 import VGG19
import datetime
import time
import os
import sys
import tarfile
import numpy as np
import h5py
import matplotlib as plt
plt.use('Agg')
import matplotlib.pyplot as pyplot
pyplot.figure
import pickle 
import matplotlib.image as img
from pickle import load
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import glob 
import datetime
import subprocess
import pandas as pd 
import json
import io
import random
import math

try:
    to_unicode = unicode
except NameError:
    to_unicode = str

now = datetime.datetime.now()
colormode = 'rgb'

img_width, img_height = 64, 64

train_data_dir = '/data/train' 
test_data_dir = '/data/test' 

nb_train_samples = 0
for root, dirs, files in os.walk(train_data_dir):
    nb_train_samples += len(files)
    print nb_train_samples

nb_test_samples = 0
for root, dirs, files in os.walk(test_data_dir):
    nb_test_samples += len(files)
    print nb_test_samples

epochs = 15
batch_size = 32

# Model definition
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255.0,
    shear_range=0.2,
    zoom_range=0.2,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    fill_mode = 'nearest',
    vertical_flip=True,
    horizontal_flip=True)

# Only rescaling for testing
test_datagen = ImageDataGenerator(rescale=1. / 255.0)

# Flows the data directly from the directory structure, resizing where needed
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode=colormode,
    class_mode='categorical')
    
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode = colormode,
    class_mode='categorical')

NumLabels = len(test_generator.class_indices)

#################################################################################################################################
# create the base pre-trained model
base_model = VGG19(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer
predictions = Dense(NumLabels, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = True

for layer in model.layers:
    layer.trainable = True

 # compile the model (should be done *after* setting layers to non-trainable)
#################################################################################################################################
model.compile(optimizer='adam', loss='binary_crossentropy',  metrics=['accuracy']) #create model with for binary output with the adam optimization algorithm

# Timehistory callback to get epoch run times
class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

time_callback = TimeHistory()

# Model fitting and training run
# Captures GPU usage
subprocess.Popen("timeout 120 nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv -l 1 | sed s/%//g > ./GPU-stats.log",shell=True)

simpsonsModel = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=nb_test_samples // batch_size,
    callbacks=[time_callback])    

# To write the each epoch run time into a json file
now = datetime.datetime.now()

filetime = str(now.year)+str(now.month)+str(now.day) 
times = time_callback.times
epochfilename ='VGG19_ClassificationEpochRuntime_'+filetime+'.json'
df=pd.DataFrame(times)
df.to_json(epochfilename, orient='records', lines=True)


modelfilename=str(now.year)+str(now.month)+str(now.day) 
modelfilename='VGG19_Model_'+modelfilename+'.h5'
model.save(modelfilename)

#historyfilename ='ClassificationModelhistory_'+filetime+'.json'
#pandas.DataFrame(simpsonsModel.history).to_json(historyfilename)

historyfilename ='VGG19_Modelhistory_'+filetime+'.json'
df=pd.DataFrame(simpsonsModel.history)
df.to_json(historyfilename, orient='records', lines=True)

# saving Confusion Matrix and Classification Report to a text file for human vision
target_names = test_generator.class_indices
optfile = 'VGG19_Modeloutput_'+filetime+'.txt'
file = open(optfile, "a+")
Y_pred = model.predict_generator(test_generator, nb_test_samples // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
ptropt= 'Confusion Matrix' 
print >> file, ptropt
cnf_matrix = confusion_matrix(test_generator.classes, y_pred)
print >>file, cnf_matrix
ptropt = 'Classification Report'
print >> file, ptropt
cls_rpt = classification_report(test_generator.classes, y_pred, target_names=target_names) 
print >> file, cls_rpt


# Serialize confusion matrix and prediction/probabilities matrix stores in json file
cmfile='VGG19_ConfusionMatrix_'+filetime+'.json' 
df=pd.DataFrame(cnf_matrix)
df.to_json(cmfile, orient='records', lines=True)

predjson='VGG19_Prediction_'+filetime+'.json' 
df=pd.DataFrame(y_pred)
df.to_json(predjson, orient='records', lines=True)

rptjson = 'VGG19_ClassificationReport_'+filetime+'.json' 
df=pd.DataFrame(cnf_matrix)
df.to_json(rptjson, orient='records', lines=True)

sysoptfile ='VGG19_ClassificationSystemEnvironment_'+filetime+'.txt'
import subprocess
sysopt = (subprocess.check_output("lscpu", shell=True).strip()).decode()
with open(sysoptfile,"a+") as f:
    for line in sysopt:
        f.write(line)

from tensorflow.python.client import device_lib
LOCAL_DEVICES = device_lib.list_local_devices()
file = open(sysoptfile, "a+")
print >> file, LOCAL_DEVICES

#Confusion Matrix is shown on a Plot
pyplot.figure(figsize=(8,8))
cnf_matrix =confusion_matrix(test_generator.classes, y_pred)
classes = list(target_names)
pyplot.imshow(cnf_matrix, interpolation='nearest')
pyplot.colorbar()
tick_marks = np.arange(len(classes))  
_ = pyplot.xticks(tick_marks, classes, rotation=90)
_ = pyplot.yticks(tick_marks, classes)
plotopt= 'VGG19_ClassificationModelImage_'+filetime+'.png'
pyplot.savefig(plotopt)

#To plot GPU usage
gpu = pd.read_csv("./GPU-stats.log")   # make sure that 120 seconds have expired before running this cell
gpuplt=gpu.plot()
gpuplt=pyplot.show()
gpuplt= 'VGG19_ClassificationGPUImage_'+filetime+'.png'
pyplot.savefig(gpuplt)