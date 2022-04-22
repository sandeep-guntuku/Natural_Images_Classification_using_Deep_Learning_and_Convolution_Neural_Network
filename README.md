# Natural_Images_Classification_using_Deep_Learning_and_Convolution_Neural_Network
Natural Images Classification using Deep Learning and Convolution Neural Network


### Introduction

Deep learning is a great approach to deal with unstructured data such as text, sound, video and image. There are a lot of implementations of deep learning in image classification and image detection, such as classifying image of dog or cats, detecting different objects in an image or do facial recognition.

On this article, we will try to build a simple image classification that will classify whether the presented image is an airplane, car, cat, dog, flower, fruit, motorbike or a person.


### Natural Images

This dataset contains 6,899 images from 8 distinct classes compiled from various sources. The classes include airplane, car, cat, dog, flower, fruit, motorbike and person. The dataset is available at Kaggle.

The link to the dataset is https://www.kaggle.com/datasets/prasunroy/natural-images?datasetId=42780&language=null

### **Exploratory Data Analysis**

Let's explore the data first before building the model. In image classification problem, it is a common practice to put each image on separate folders based on the target class/labels. For example, inside the train folder in our data, you can that we have 7 different folders, respectively for `airplane`, `car`, `cat`, `dog`,
`flower`, `fruit`, `motorbike`, `person`.

## **Data Preprocessing**

Data preprocessing for image is pretty simple and can be done in a single step in the following section.


Since we have a good of training set, we don't need to build artificial data using method called `Data Augmentation`. 

Data augmentation is one useful technique in building models that can increase the size of the training set without acquiring new images. But, here we are not using Data Augmentation as we have enough data for building and training the deep learning convnet model.

Here we are reading the images and converting them to tensors while rescaling the pixel values to [0,1] interval.

## **Model Architecture**

Convnet model has been built for the deep learning classification.


- Convolutional layer to extract features from 2D image with `relu` activation function
- Max Pooling layer to downsample the image features
- Flattening layer to flatten data from 2D array to 1D array
- Dense layer to capture more information
- Dense layer for output with `softmax` activation function


Model is built already in R script and saved so that it can be loaded again.


## Now we are loading the saved model in R Markdown

## **Model Fitting**

## Visualising and Evaluating the test data

Plotting the loss and accuracy for training and validation data.
The accuracy on the test data on the model is around 98%.

### **Conclusion:**

1. I have built the convnet model for predicting 8 classes from the natural images dataset.

2. From the visulaization of loss and accuracy with regards to epoch , I observed that accurary of the model will improve with more epochs.

3. I could observe over fitting issue based on the loss and epoch variation, and by using data augmentation technique we can overcome the overfitting.

4. With around 30 epochs, the model gave an accuracy of around 98% on the testing data.
