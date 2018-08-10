from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import os
import sys
import tarfile
import numpy as np
import syncImageDataFromGirder as ImageSync


data_root = os.sep+os.path.join('tmp', 'simpsons')
dest_filename = os.path.join(data_root, 'simpsons_dataset.tar.gz')


useGirder=True

# filename = '/home/dagutman/dev/KerasSimpsons_Tensorflow/simpsons_dataset.tar.gz'

# def maybe_extract(filename, force=False):
#   root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
#   if os.path.isdir(root) and not force:
#     print('%s already present - Skipping extraction of %s.' % (root, filename))
#   else:
#     print('Extracting data for %s. This may take a while. Please wait.' % root)
#     tar = tarfile.open(filename)
#     sys.stdout.flush()
#     tar.extractall(root)
#     tar.close()
#   data_folders = [
#     os.path.join(root, d) for d in sorted(os.listdir(root))
#     if os.path.isdir(os.path.join(root, d))]
#   print(data_folders)
#   return data_folders
# data_folders = maybe_extract(dest_filename)


# data_folders = maybe_extract(filename)


# print "--PRINTING THE OUTPUT FOR DATA FOLDERS NOW----"
# print data_folders
# print len(data_folders)
# sys.exit()

rawImageRootDir = "/home/dagutman/devel/KerasSimpsons_Tensorflow/rawImageData"


### This should grab the training and testing data from Girder; this onl downloads images your missing
if useGirder:
  ImageSync.syncImageDataFromGirder( rawImageRootDir)

### 
img_width, img_height = 64, 64
train_data_dir = os.path.join(rawImageRootDir,'training')
validation_data_dir = os.path.join(rawImageRootDir,'validation')

# data_folders =  [
#     os.path.join(, d) for d in sorted(os.listdir(root))
#     if os.path.isdir(os.path.join(root, d))]

data_folders = os.listdir(train_data_dir)

nb_train_samples = 30000
nb_validation_samples = 990
epochs = 50
batch_size = 32


# Model definition
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

### this corresponds to the number of classes/characters we have in our TRAINING set
NumLabels = 42
## TO DO ; make this just equal to the number of directories in the training_data_dir


    
model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(NumLabels))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


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
    validation_data_dir ,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

simpsonsModel = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)
	
