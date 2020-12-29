#import libraries and pckages
from tensorflow.keras.models import Sequential
#Sequential is used to initialize our neural netwotrk
#two way to initialize our neural netwotrk
# 1.sequence of a layer
# 2.as a graph
from tensorflow.keras.layers import Convolution2D
#image 2 dimension but video 3 dimension
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
#dense is used to add fully connected layer in a classic
#(classic) artificial neural network

#initializing the CNN
classifier=Sequential()

#step 1: Convolution  that means
#add a convolutional layer which takes the input image:
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))
#number of filter=32,kernel size is 3x3 that means row=3,column=3
#2D image shape 256x256x2  3D image shape 256x256x3
#here for small calculation but geeting good accuracy we use 64x64x3(color image)
classifier.add(Convolution2D(64,3,3,activation='relu'))
#as we specify input_shape=(64,64,3) at the above so no need to specify another time 


#step 2: Maxpooling
classifier.add(MaxPooling2D(pool_size=(2,2)))
#subtble size or pool_size=(2,2)(khata dekho) and stride automatically 2

#step 3: Flattening
classifier.add(Flatten())

#step 4: Full connection(fully connected layer)
#there are two important task of full connection
#  1) inputlayer er sathe fully connected layer er relation kora
#  2) output layer theke final prediction koraa
#image classification is a non-inear task
#binary outcome er activation='sigmoid'
#more than 2 (binary) binary outcome er activation='sigmoid'
classifier.add(Dense(units= 128 ,activation='relu'))
#Dense is a standard layer of the neural network in which each 
#neuron is connected to each neuron of the next layer.
classifier.add(Dense(units=1 ,activation='sigmoid'))
#units:Positive integer, dimensionality of the output space.that 
#means number of node in the hidden/fully connected layer


#Compiling the CNN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#if the outcome is just 2, lossfunction will be loss='binary_crossentropy'
#if the outcome is more than 2, lossfunction will be loss='categorical_crossentropy'


#part -2 Ftting the mages to the images
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        'training_set',
        target_size=(64, 64),
        #since input size is 64x64
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'test_set',
        target_size=(64, 64),
        #since input size is 64x64
        batch_size=32,
        class_mode='binary')


classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        #not samples_per_epoch=8000
        #steps_per_epoch=8000,
        
        epochs=3,
        #not nb_epoch=10,
        validation_data=test_set)
        #we dont use  nb_val_samples=800
        #since test_set has 2000 images






