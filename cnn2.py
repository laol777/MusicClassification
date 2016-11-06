from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adagrad

from PIL import Image, ImageDraw, ImageColor
import numpy as np
import os

def normalize (x):
      m = 0;
      w = x.shape[0]
      h = x.shape[1]

      for i in range(w):
            for j in range(h):
                  m = max(abs(x[i][j]), m)
      res = np.array(x, dtype=np.float32)
      res = np.reshape(res, x.shape)     
      for i in range(w):
            for j in range(h):
                  res[i][j] = x[i][j] / m
      return res

      
def drawImage(x, path):
      im = Image.new('RGB', x.shape)
      draw = ImageDraw.Draw(im)
      w = x.shape[0]
      h = x.shape[1]

      for i in range(w):
            for j in range(h):
                  c = ImageColor.getrgb('hsl(' + str(int(270 + x[i][j] * 90)) + ', 100%, 50%)')
                  draw.point((i, j), c)   
      del draw

      # write to stdout
      im.save(path, "PNG")

def loadTrainingData(x, y, ganre, path, w, h, dtype):
      f = open(path, "rb")
      
      # get file size
      f.seek(0, os.SEEK_END)
      samplesCount = int(f.tell() / w / h / np.dtype(dtype).itemsize)
      f.seek(0, os.SEEK_SET)
      
      buf = np.fromfile(f, dtype=dtype, count=-1)
      
      f.close()
      
      x = np.append(x, buf)         
      y = np.append(y, np.repeat([ganre], samplesCount, axis=0))

      return samplesCount, x, y

def shuffle(x, y, samplesCount):
      indexes = np.arange(samplesCount)
      np.random.shuffle(indexes)
      x_shuffeled = np.array(x)
      y_shuffeled = np.array(y)

      
      for i in range(len(x)):
            x_shuffeled[i] = x[indexes[i]]
            y_shuffeled[i] = y[indexes[i]]

      return x_shuffeled, y_shuffeled


w = 64 # data width
h = 128 # data height
nclasses = 9 # number of classes in dataset

x = np.array([], dtype=np.float32)
y = np.array([], dtype=np.float32)      

print('start data read...')
totalSamplesCount = 0;
samplesCount, x, y = loadTrainingData(x, y, [1, 0, 0, 0, 0, 0, 0, 0, 0], "dataset2/classical.bin", w, h, np.float32)
totalSamplesCount = totalSamplesCount + samplesCount
samplesCount, x, y = loadTrainingData(x, y, [0, 1, 0, 0, 0, 0, 0, 0, 0], "dataset2/drum.bin", w, h, np.float32)
totalSamplesCount = totalSamplesCount + samplesCount
samplesCount, x, y = loadTrainingData(x, y, [0, 0, 1, 0, 0, 0, 0, 0, 0], "dataset2/dubstep.bin", w, h, np.float32)
totalSamplesCount = totalSamplesCount + samplesCount
samplesCount, x, y = loadTrainingData(x, y, [0, 0, 0, 1, 0, 0, 0, 0, 0], "dataset2/hiphop.bin", w, h, np.float32)
totalSamplesCount = totalSamplesCount + samplesCount
samplesCount, x, y = loadTrainingData(x, y, [0, 0, 0, 0, 1, 0, 0, 0, 0], "dataset2/jazz.bin", w, h, np.float32)
totalSamplesCount = totalSamplesCount + samplesCount
samplesCount, x, y = loadTrainingData(x, y, [0, 0, 0, 0, 0, 1, 0, 0, 0], "dataset2/metal.bin", w, h, np.float32)
totalSamplesCount = totalSamplesCount + samplesCount
samplesCount, x, y = loadTrainingData(x, y, [0, 0, 0, 0, 0, 0, 1, 0, 0], "dataset2/pop.bin", w, h, np.float32)
totalSamplesCount = totalSamplesCount + samplesCount
samplesCount, x, y = loadTrainingData(x, y, [0, 0, 0, 0, 0, 0, 0, 1, 0], "dataset2/reggae.bin", w, h, np.float32)
totalSamplesCount = totalSamplesCount + samplesCount
samplesCount, x, y = loadTrainingData(x, y, [0, 0, 0, 0, 0, 0, 0, 0, 1], "dataset2/rock.bin", w, h, np.float32)
totalSamplesCount = totalSamplesCount + samplesCount

x = np.reshape(x, (totalSamplesCount, 1, w, h))
y = np.reshape(y, (totalSamplesCount, nclasses))

print('start shuffling...')
x, y = shuffle(x, y, totalSamplesCount)

print('start creating model...')
model = Sequential()
 
model.add(Convolution2D(256, 4, 128, border_mode='valid', activation='relu', init='glorot_uniform', input_shape=(1, 64, 128))) # 1
# 256 kernels 4x128
model.add(MaxPooling2D(pool_size = (2, 1))) # 2
model.add(Reshape(dims=(1, 256, 30))) # 3
model.add(Dropout(0.25))
model.add(Convolution2D(128, 256, 2, border_mode='valid', activation='relu', init='glorot_uniform')) # 4
# 128 kernels 256x2
model.add(MaxPooling2D(pool_size=(1, 2))) # 5
model.add(Flatten()) # 6
model.add(Dropout(0.25))
model.add(Dense(1024, init='glorot_uniform', activation='relu')) # 7
model.add(Dropout(0.5)) # 7
model.add(Dense(128, init='glorot_uniform', activation='relu')) # 8
model.add(Dropout(0.5)) # 8
model.add(Dense(nclasses, init='glorot_uniform', activation='softmax')) # 9


print('start compile...')
#model.load_weights('weightsDropOut.h5')



model.compile(loss='categorical_crossentropy', optimizer='sgd')
#print(model.layers[0].get_weights()[0].shape)
#for i in range(256):
#      drawImage(model.layers[0].get_weights()[0][i][0], str(i) + '.png')
#from keras import callbacks
#remote = callbacks.RemoteMonitor(root='http://localhost:9000')
#for i in range(totalSamplesCount):
#      wprev = model.layers[0].get_weights()[0][0][0]
#      model.fit(x[[i]], y[[i]], batch_size=1, nb_epoch=1, validation_split=0.0, shuffle=True, show_accuracy=True, verbose=1,callbacks=[remote])
#      wnext = model.layers[0].get_weights()[0][0][0]
#      print(wprev - wnext)
#      input('Press enter to continue...')

#indexes = range(20)
while (True):
      print('start shuffling...')
      x, y = shuffle(x, y, totalSamplesCount)
      print('start fiting...')
      hist = model.fit(x, y, batch_size=16, nb_epoch=1, validation_split=0.1, shuffle=True, show_accuracy=True, verbose=1)
      model.save_weights('weightsDropOut.h5', overwrite = True)
      print(hist.history)
#      print('start testing...')
#      indexes = range(1, totalSamplesCount, int(totalSamplesCount/10))
#      x_test = np.array(x[indexes])
#      y_test = np.array(y[indexes])
#      score = model.predict(x_test)
#      print(score)
#      print(y_test)

