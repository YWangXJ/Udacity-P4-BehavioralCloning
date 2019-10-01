# CarND-Behavioral-Cloning-P3
import os
import csv
import math 

###read driving img paths and mearsurements
samples = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        samples.append(line)
print('total number of samples',len(samples))
### split the samples
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle
import random

### Generate random brightness function, produce darker transformation 
def random_brightness(image):
    #Convert 2 HSV colorspace from RGB colorspace
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    #Generate new random brightness
    rand = random.uniform(0.3,1.0)
    hsv[:,:,2] = rand*hsv[:,:,2]
    #Convert back to RGB colorspace
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return new_img 

### Crop image to remove the sky and driving deck, resize to 64x64 dimension (tried a version without resizing but not good. not sure what's the reason)
def crop_resize(image):
  cropped = cv2.resize(image[60:140,:], (64,64))
  return cropped

### use generator to save memory and create trainng and validation data
def generator(samples, batch_size=32,correction=0.3):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            augmented_images, augmented_angles = [],[]
            for batch_sample in batch_samples:
                center_name = '../data/IMG/'+batch_sample[0].split('/')[-1]
                left_name = '../data/IMG/'+batch_sample[1].split('/')[-1]
                right_name = '../data/IMG/'+batch_sample[2].split('/')[-1]
#                 print('center_name',center_name)
                ### augment, crop and resize the image inputs
                center_image = crop_resize(random_brightness(cv2.imread(center_name)))
                left_image = crop_resize(random_brightness(cv2.imread(left_name)))
                right_image = crop_resize(random_brightness(cv2.imread(right_name)))
#                 center_image = cv2.imread(center_name)
#                 left_image = cv2.imread(left_name)
#                 right_image = cv2.imread(right_name)
                center_angle = float(batch_sample[3])
                left_angle = center_angle + correction
                right_angle = center_angle - correction            
                
                            
                images.append(center_image)          
                images.append(left_image)
                images.append(right_image)
                angles.append(center_angle)
                angles.append(left_angle)
                angles.append(right_angle)

            ### flip the img on vertical axis          
            for image,angle in zip(images,angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                augmented_images.append(cv2.flip(image,1))
                augmented_angles.append(angle*-1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# Set our batch size
batch_size=64

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)


from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Lambda, Activation, MaxPooling2D, Dropout, Cropping2D
from keras.layers import Convolution2D
from keras.regularizers import l2

#NVIDIA architecture
input_shape = (64,64,3)

model = Sequential()
model.add(Lambda(lambda x: x/172.5- 1, input_shape=input_shape)) #nomalize the data
model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample =(2,2), W_regularizer = l2(0.001)))
model.add(Activation('relu'))
model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample =(2,2), W_regularizer = l2(0.001)))
model.add(Activation('relu'))
model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample = (2,2), W_regularizer = l2(0.001)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, border_mode='same', subsample = (2,2), W_regularizer = l2(0.001)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample = (2,2), W_regularizer = l2(0.001)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(80, W_regularizer = l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(40, W_regularizer = l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(16, W_regularizer = l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(10, W_regularizer = l2(0.001)))
model.add(Dense(1, W_regularizer = l2(0.001)))
model.summary()


model.compile(loss='mse',optimizer='adam')
# model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=5)
history_object = model.fit_generator(train_generator,
            steps_per_epoch=math.ceil(len(train_samples)/batch_size), 
            validation_data=validation_generator, 
            validation_steps=math.ceil(len(validation_samples)/batch_size), 
            epochs=3, verbose=1)

model.save('model.h5')
print('Congrats! Model saved')

### print the keys contained in the history object
print(history_object.history.keys())
import matplotlib.pyplot as plt
### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('learning.png')
exit()
    