# Ship Classification 
----------------

## Part 1. Introduction

### 1.1. The Goal of The Project

Detection and classification of ships using satellite images is extremely important and useful for maritime surveillance activities. Thus, governments can keep track of what types of ships are in the sea areas and what they are doing (fishing, drilling, exploration, cargo transportation, tourism, etc.). Detection of ships is an extremely useful and important task to monitor their impact on safety, security, economy or the environment.

In this work, we will develop a convolutional neural network (CNN) to classify vessels extracted from Sentinel-1 data. A VGG16 Convolutional Neural Network model will be trained with the help of the OpenSARShip dataset, a benchmarking dataset, to develop applicable and adaptive ship interpretation algorithms.

### 1.2. About Dataset

The data file to be used here is .NPZ. This file format is a compressed version of .NPY files and is often used for Machine Learning applications, especially when multiple large images (numpy arrays) need to be saved in a compressed format. The data to be used here includes 2805 images with a size of 128x128 pixels. Each image consists of a ship extracted from a VV-polarized Sentinel-1 image.

* 0 - Bulk Carrier,
* 1 - Container Ship,
* 2 - Tanker 

### 1.3. About Algorithm
Here we will use the VGG16 model, a Deep Learning algorithm used to classify and detect features in an image. The model was proposed by K. Simonyan and A. Zisserman of Oxford University in the paper 'Deep Convolutional Networks for Large-Scale Image Recognition'. This model is one of the simpler convolutional neural networks and uses only basic convolutions and pooling operations. 

**Key Features of VGG16:**
* it is also called the OxfordNet model, named after the Visual Geometry Group from Oxford
* Number 16 refers to the fact that it has a total of 16 layers that have some weights uses always a 3x3 kernel for convolution
* it only has Conv and pooling layers

For more detail information, please look at **Further Resources**.


## Part 2. Further Resources

[OpenSARSHİP](https://ieeexplore.ieee.org/document/8067489)

[Very Deep CNN For Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

[Merchant Vessel Classification Based on Scattering Component Analysis](https://ieeexplore.ieee.org/abstract/document/6451119)

