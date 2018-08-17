Computer Vision Algorithms
Histo-pathology whole slide images multi-class (4 classes) Classification usings Convolutional Neural Networks Algorithms (Google InceptionV3 architectures with change in parameters)
Uses Keras ImageDataGenerator, Docker Images with Python3 and GPU optimization.

Data Summary:
Training#1708, Validation#425, Testing#744
Physical image size: 256
Model input image size =150 x 150 (Recommended/used by VGG19, InceptionV3 architectures)
classes:4 (brca, coad, gbm, lgg)
Model C- InceptionV3, Epoch-100  #batches-25, #Fully Connected N= 1024, Training Accuracy from Model-99.96%, Confusion Matrix Precision-91%
Model E- InceptionV3, Epoch-100, #batches- 5, #Fully Connected N=10000, Training Accuracy from Model-99.64%, Confusion Matrix Precision-82%
