'''
# Creating folder structure in image classification data, output, reports, model, figures
datarootdir = '/data'
optdir = '/data/output'
rptdir=optdir+'/reports'
modeldir=optdir+'/models'
figuresdir=optdir+'/figures'
codedir='/data/code'
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

# Execute Training Data Download from Girder
# execfile("downloadBRCAImageSet.py")

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
