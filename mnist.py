import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import re 
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.datasets import mnist    
from keras.utils import np_utils
from keras.models import load_model

(X_train,y_train),(X_test, y_test) = mnist.load_data()

X_train = X_train.astype('f')    
X_test = X_test.astype('f')

X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, num_classes=10).astype('i')
Y_test = np_utils.to_categorical(y_test, num_classes=10).astype('i')

y_train_return = Y_train.argmax(axis=1)

batch_size = 300
n_epoch = 10
model = Sequential()
model.add(Flatten(input_shape = (28,28)))


model.add(Dense(900))
model.add(Activation('relu'))
model.add(Dense(1000))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy']) 

hist = model.fit(X_train,Y_train,
               epochs = n_epoch,
               validation_data = (X_test,Y_test),
               verbose = 1,
               batch_size = batch_size)


                
model.save("model_mnist.h5")