from keras.utils.visualize_util import model_to_dot
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adagrad

import numpy as np
import os


import cPickle as pickle


name = "targetSubsample.pickle"
with open(name, 'rb') as f:
    y = pickle.load(f)
    
name = "trainSubsample.pickle"
with open(name, 'rb') as f:
    X = pickle.load(f)

#Angry 1, 0, 0, 0
#Sad 0, 1, 0, 0
#Tense 0, 0, 1, 0
#Happy 0, 0, 0, 1

print y.shape
print X.shape


w = 128
h = 733
nclasses = 4
samplesCount = 800



nKernel = 10


model = Sequential()

model.add(Convolution2D( nKernel, 64, 4, border_mode='same', activation='relu', init='glorot_uniform', input_shape=(1, 128, 733)))
model.add(MaxPooling2D(pool_size = (1, 2)))
model.add(Flatten())

model.add(Dense(100, init='glorot_uniform', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(25, init='glorot_uniform', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nclasses, init='glorot_uniform', activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

def shuffle(x, y, samplesCount):
    indexes = np.arange(samplesCount)
    np.random.shuffle(indexes)
    x_shuffeled = np.array(x)
    y_shuffeled = np.array(y)


    for i in range(len(x)):
        x_shuffeled[i] = x[indexes[i]]
        y_shuffeled[i] = y[indexes[i]]

    return x_shuffeled, y_shuffeled

X, y = shuffle(X, y, samplesCount)


countInTrain = 200
model.fit(X.reshape(samplesCount, 1, 128, 733)[:countInTrain], y.reshape(samplesCount, 4)[:countInTrain],\
          batch_size=20, nb_epoch=10, validation_split=0.1, show_accuracy=True, shuffle=True, verbose=1)


pred = model.predict(X.reshape(samplesCount, 1, 128, 733))

def SetMaxProbabilityToOne(array):
    return map(lambda x: 1 if(x == max(array)) else 0, array)

original = y[:countInTrain]
predict = pred[:countInTrain]
countTrue = 0

original = y[countInTrain:samplesCount]
predict = pred[countInTrain:samplesCount]
for i in range(len(predict)):
    predict[i] = SetMaxProbabilityToOne(predict[i])
    countTrue += 1 if (sum(predict[i] == original[i]) == nclasses) else 0
print countTrue,
print len(predict)