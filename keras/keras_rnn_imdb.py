import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers
from keras.datasets import imdb
from keras.preprocessing import sequence

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())



epochs = 10
batch_size = 32
length = 250
embdim = 32
vocsize = 2000

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocsize)

x_train = sequence.pad_sequences(x_train, length)
x_test = sequence.pad_sequences(x_test, length)

model = keras.Sequential()
initializer = initializers.RandomUniform(-0.05, 0.05)
model.add(layers.Embedding(vocsize, embdim, embeddings_initializer=initializer))
model.add(layers.SimpleRNN(32))
model.add(layers.Dense(256))
model.add(layers.Activation('relu'))
model.add(layers.Dense(1))
model.add(layers.Activation('sigmoid'))

model.summary()

keras.utils.plot_model(model, to_file="keras_rnn_model.png")

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['binary_accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))

score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)

print('Test score: ', score)
print('Test accuracy: ', acc)
