# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 08:58:40 2017

@author: pcomitz
"""

###############
# Paul Comitz
# Project 3 
# renamed to model.py for submission 8/12 
##############

import csv
import cv2
import numpy as np

#cv2 wont display images? 
#import matplotlib.image as img
import matplotlib.pyplot as plt
import random

lines = [] 
lineNum = 0; 
file = 'Z:\\proj3OnDisk\\data\\data\\driving_log_no_header.csv'

with open(file) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        lineNum = lineNum+1
        
print("number of lines in ",file, " is " , lineNum)    
    
images = []
measurements = []
#Step 12 adding right and left camera
for line in lines: 
    for i in range(3):
        source_path = line[i]
        next = source_path.split('/')
        filename = next[1]
        current_path = 'Z:\\proj3OnDisk\\data\\data\\IMG\\' + filename
        image = cv2.imread(current_path)
        # BGR to RGB - key step!
        imgOut = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(imgOut)
        # adjust correction from 0.2 to 0.1 
        if(i == 0):
            measurement = float(line[3]) 
        elif(i == 1):
            measurement = float(line[3]) + 0.1
        elif (i ==2): 
            measurement = float(line[3]) - 0.1
        # add to measurements
        measurements.append(measurement)    
                
#Step 11 Data Augmentation
augmented_images, augmented_measurements = [],[]
for(image,measurement) in zip(images, measurements):    
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)
    
#test image display
#display a few random images to test 
index = random.randint(0, len(augmented_images))
plt.figure(figsize=(1,1))
fig = plt.figure()
title = 'first:' + str(augmented_measurements[index])
a=fig.add_subplot(1,3,1)
a.set_title(title)
testImage= augmented_images[index].squeeze()
plt.imshow(testImage)

#next image
a=fig.add_subplot(1,3,2)
title = 'second:' + str(augmented_measurements[index+1])
a.set_title(title)
testImage= augmented_images[index+1].squeeze()
plt.imshow(testImage)

#next image
a=fig.add_subplot(1,3,3)
title = 'third:' + str(augmented_measurements[index+2])
a.set_title(title)
testImage= augmented_images[index+2].squeeze()
plt.imshow(testImage)


# prepare for Keras with augmented data from Step 11
# Keras requires numpy arrays
X_train= np.array(augmented_images)
y_train = np.array(augmented_measurements)

# included to illustrate iterative developementS
# Step 7 build simplest network possible
# flattened image connected to single output node
# single output node will predict steering angle
# makes this a regression network
# for classification network might apply softmax 
# activation function to the output layer
# 

from keras.models import Sequential
#Step 13 add cropping
from keras.layers import Flatten, Dense, Lambda, Cropping2D

#from Step 10 add 2D Convolution
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()

#from step 9 Data Preprocessing
model.add(Lambda(lambda x: (x/255.0)-0.5, input_shape=(160,320,3)))

# Step 13 Crop 70 pix from top, 25 from bottom
# from keras docuentation
# If tuple of 2 tuples of 2 ints: interpreted as 
# ((top_crop, bottom_crop), (left_crop, right_crop))
model.add(Cropping2D(cropping=((70,25),(0,0))))

#Step 10
"""
This was subsequently removed. It is included 
in this source ile to illustrate the 
iterative nature of the development. 
model.add(Convolution2D(6,5,5, activation = "relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5, activation = "relu"))
model.add(MaxPooling2D())
model.add(Dense(120))
model.add(Dense(84))S
model.add(Flatten())
model.add(Dense(1))
"""

###
# Step 14 nvidia architecture
# 5 convolutions followed by 4 fully connected
###
model.add(Convolution2D(24,5,5,subsample=(2,2), activation = "relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2), activation = "relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2), activation = "relu"))
model.add(Convolution2D(64,3,3,activation = "relu"))
model.add(Convolution2D(64,3,3,activation = "relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#use mse rather than cross entropy
model.compile(loss='mse', optimizer='adam')

#shuffle and split off 20% for validation
EPOCHS = 4
print("starting to train for ",EPOCHS, " epochs")
model.fit(X_train,y_train, validation_split = 0.2, shuffle=True, nb_epoch = EPOCHS)

#save the model 
print("saving the model")
#model.save('model.h5.lambda.conv.aug.crop.lr.rgb.nvidia.4epochs')
model.save('model.h5')