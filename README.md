# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./learning.png "Learning History"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is the Nvidia Model, and the data is normalized in the model using a Keras lambda layer (code line 104). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 117,119,121). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py lines 129-133). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 127).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to create a strong but fast so that the model can run in real time

My first step was implement the Nvidia architecture since it is well-developed and dedictated to the self-driving cars.

I collect the data by 1) driving the car around the center of road;

After the implementation, I test the car in the simulation but noticed that the car ran into the lake for the first sharp turn
 

To combat the overfitting, I modified the epochs from 5 to 3 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track when 1)encounter the  sharp turn and 2) turning has dirt surface. To improve the driving behavior in these cases,  I collect data by: 2) recovering from the edge of the road; 3)driving counter clockwise

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 103-123) consisted of a convolution neural network with the following layers and layer sizes
Layer (type)          Output Shape     		Param #   
=============================================================
lambda_1 (Lambda)        (None, 64, 64, 3)         0         
_________________________________________________________________
conv2d_1 (Conv2D)        (None, 30, 30, 24)        1824      
_________________________________________________________________
activation_1 (Activation)  (None, 30, 30, 24)          0         
_________________________________________________________________
conv2d_2 (Conv2D)        (None, 13, 13, 36)        21636     
_________________________________________________________________
activation_2 (Activation)   (None, 13, 13, 36)         0         
_________________________________________________________________
conv2d_3 (Conv2D)        (None, 5, 5, 48)          43248     
_________________________________________________________________
activation_3 (Activation)   (None, 5, 5, 48)          0         
_________________________________________________________________
conv2d_4 (Conv2D)        (None, 3, 3, 64)          27712     
_________________________________________________________________
activation_4 (Activation)  (None, 3, 3, 64)           0         
_________________________________________________________________
conv2d_5 (Conv2D)       (None, 1, 1, 64)          36928     
_________________________________________________________________
activation_5 (Activation)  (None, 1, 1, 64)           0         
_________________________________________________________________
flatten_1 (Flatten)       (None, 64)              0         
_________________________________________________________________
dense_1 (Dense)          (None, 80)            5200      
_________________________________________________________________
dropout_1 (Dropout)       (None, 80)              0         
_________________________________________________________________
dense_2 (Dense)          (None, 40)             3240      
_________________________________________________________________
dropout_2 (Dropout)       (None, 40)              0         
_________________________________________________________________
dense_3 (Dense)          (None, 16)             656       
_________________________________________________________________
dropout_3 (Dropout)       (None, 16)              0         
_________________________________________________________________
dense_4 (Dense)          (None, 10)             170       
_________________________________________________________________
dense_5 (Dense)          (None, 1)               11        
=============================================================


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I recorded two laps on track one using center lane driving

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover to center when car is running close to edges.

I also ran the vehicle on the track two for half a lap to generalize the model.

To better handle the sharp turns, I also collect more data for sharp turns. 

To augment the data set, I also flipped images and angles and randomize the brightness of the image input.


After the collection process, I had 238328 number of data points. I then preprocessed this data by 1)nomalize the data and crop the sky and vehicle front of the image.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the learning history shown in below. 
![alt text][image1]

I used an adam optimizer so that manually training the learning rate wasn't necessary.
