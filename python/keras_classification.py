# A multi classes image classifier, based on convolutional neural network using Keras and Tensorflow. 
# A multi-label classifier (having one fully-connected layer at the end), with multi-classification (18 classes, in this instance).
# Largely copied from the code https://gist.github.com/seixaslipe
# This is based on these posts: https://medium.com/alex-attia-blog/the-simpsons-character-recognition-using-keras-d8e1796eae36
# Data downloaded from Kaggle 
# Will emulate the image classification functionlities for Neuro Pathology images/slides (WSI-Whole Slide images)
# Will implement/include data manipulating functionalities based on Girder (https://girder.readthedocs.io/en/latest/)
# Has 6 convulsions, filtering start with 64, 128, 256 with flattening to 1024
# Used Keras.ImageDataGenerator for Training/Validation data augmentation and the augmented images are flown from respective directory
# Environment: A docker container having Keras, TensorFlow, Python-2 with GPU based execution

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.callbacks import Callback
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
import pandas
from sklearn.metrics import classification_report, confusion_matrix
import glob 
import datetime
import subprocess
import pandas as pd 
import json
import io

try:
    to_unicode = unicode
except NameError:
    to_unicode = str

now = datetime.datetime.now()

# Creating folder structure in image classification data, output, reports, model, figures
datarootdir = '/data'
optdir = './output'
rptdir=optdir+'/reports'
modeldir=optdir+'/models'
figuresdir=optdir+'/figures'
codedir='~/code'
train_data_dir = '/data/train' 
validation_data_dir = '/data/val' 
test_data_dir = '/data/test'


if not os.path.isdir(optdir):
    os.makedirs(optdir)

if not os.path.isdir(rptdir):
    os.makedirs(rptdir)

if not os.path.isdir(modeldir):
    os.makedirs(modeldir)    

if not os.path.isdir(figuresdir):
    os.makedirs(figuresdir)    

if not os.path.isdir(codedir):
    os.makedirs(codedir)    

if not os.path.isdir(train_data_dir):
    os.makedirs(train_data_dir)    

if not os.path.isdir(validation_data_dir):
    os.makedirs(validation_data_dir)    

if not os.path.isdir(test_data_dir):
    os.makedirs(test_data_dir)    

'''
# Execute Training Data Download from Girder
execfile("downloadBRCAImageSet.py")

# Copies downloaded images into docker train, val and test folder
# can be deleted if dont need to store in the downloaded folder
# For testing purposes copying to a persistent storage media

import shutil
import glob

for filename in glob.glob(os.path.join(trainingOutputDir, '*.*')):
    shutil.copy(filename, train_data_dir)

for filename in glob.glob(os.path.join(testingOutputDir, '*.*')):
    shutil.copy(filename, test_data_dir)

for filename in glob.glob(os.path.join(validationOutputDir, '*.*')):
    shutil.copy(filename, validation_data_dir)
'''
img_width, img_height = 64, 64

nb_train_samples = 0
for root, dirs, files in os.walk(train_data_dir):
    nb_train_samples += len(files)
    print nb_train_samples

nb_validation_samples = 0
for root, dirs, files in os.walk(validation_data_dir):
    nb_validation_samples += len(files)
    print nb_validation_samples

nb_test_samples = 0
for root, dirs, files in os.walk(test_data_dir):
    nb_test_samples += len(files)
    print nb_test_samples

epochs = 2
batch_size = 8

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
    horizontal_flip=True)

# Only rescaling for validation
valid_datagen = ImageDataGenerator(rescale=1. / 255.0)

# Flows the data directly from the directory structure, resizing where needed
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = valid_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

NumLabels = len(validation_generator.class_indices)

'''
6-conv layers - added on 06/21, Raj
'''
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(256, (3, 3), padding='same')) 
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(NumLabels, activation = 'softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


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
#subprocess.Popen("timeout 120 nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv -l 1 | sed s/%//g > rptdir+'/'+ GPU-stats.log",shell=True)
simpsonsModel = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=[time_callback])    

# To write the each epoch run time into a json file
now = datetime.datetime.now()

filetime = str(now.year)+str(now.month)+str(now.day) 
times = time_callback.times
epochfilename =rptdir+'/'+'classificationEpochRuntime_'+filetime+'.json'
df=pd.DataFrame(times)
df.to_json(epochfilename, orient='records', lines=True)

modelfilename=str(now.year)+str(now.month)+str(now.day) 
modelfilename=modeldir+'/'+'classificationModel_'+modelfilename+'.h5'
model.save(modelfilename)

#historyfilename ='ClassificationModelhistory_'+filetime+'.json'
#pandas.DataFrame(simpsonsModel.history).to_json(historyfilename)

historyfilename =rptdir+'/'+'ClassificationModelhistory_'+filetime+'.json'
df=pd.DataFrame(simpsonsModel.history)
df.to_json(historyfilename, orient='records', lines=True)

# saving Confusion Matrix and Classification Report to a text file for human vision
target_names = validation_generator.class_indices
optfile = rptdir+'/'+'classificationModeloutput_'+filetime+'.txt'
file = open(optfile, "a+")
Y_pred = model.predict_generator(validation_generator, nb_validation_samples // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
ptropt= 'Confusion Matrix' 
print >> file, ptropt
cnf_matrix = confusion_matrix(validation_generator.classes, y_pred)
print >>file, cnf_matrix
ptropt = 'Classification Report'
print >> file, ptropt
cls_rpt = classification_report(validation_generator.classes, y_pred, target_names=target_names) 
print >> file, cls_rpt


# Serialize confusion matrix and prediction/probabilities matrix stores in json file
cmfile=rptdir+'/'+ 'confusionMatrix_'+filetime+'.json' 
df=pd.DataFrame(cnf_matrix)
df.to_json(cmfile, orient='records', lines=True)

predjson=rptdir+'/'+ 'prediction_'+filetime+'.json' 
df=pd.DataFrame(y_pred)
df.to_json(predjson, orient='records', lines=True)

rptjson = rptdir+'/'+'classificationReport_'+filetime+'.json' 
df=pd.DataFrame(cnf_matrix)
df.to_json(rptjson, orient='records', lines=True)

sysoptfile =rptdir+'/'+ 'classificationSystemEnvironment_'+filetime+'.txt'
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
cnf_matrix =confusion_matrix(validation_generator.classes, y_pred)
classes = list(target_names)
pyplot.imshow(cnf_matrix, interpolation='nearest')
pyplot.colorbar()
tick_marks = np.arange(len(classes))  
_ = pyplot.xticks(tick_marks, classes, rotation=90)
_ = pyplot.yticks(tick_marks, classes)
plotopt= figuresdir+'/'+'classificationModelImage_'+filetime+'.png'
pyplot.savefig(plotopt)

#To plot GPU usage
gpu = pd.read_csv("./GPU-stats.log")   # make sure that 120 seconds have expired before running this cell
gpuplt=gpu.plot()
gpuplt=pyplot.show()
gpuplt= figuresdir+'/'+'classificationGPUImage_'+filetime+'.png'
pyplot.savefig(gpuplt)