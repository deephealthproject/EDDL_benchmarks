import os
from time import time
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, ReLU, Flatten, Input, Reshape, Softmax, BatchNormalization
from tensorflow.keras.optimizers import RMSprop

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Training parameters
batch_size = 100
num_classes = 10
epochs = 1

(x_train, y_train), (x_test, y_test) = mnist.load_data()  # Load dataset

# Reshape the images to start from the same format than the EDDL mnist dataset
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalize images
x_train /= 255.0
x_test /= 255.0
# Convert labels to one hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Build the model
model = Sequential()
model.add(Input(shape=(784)))
model.add(Reshape(target_shape=(28, 28, 1)))
model.add(Conv2D(32, (3, 3), padding="same"))
model.add(BatchNormalization())
model.add(ReLU())
model.add(MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding="same"))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(BatchNormalization())
model.add(ReLU())
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(BatchNormalization())
model.add(ReLU())
model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Conv2D(256, (3, 3), padding="same"))
model.add(BatchNormalization())
model.add(ReLU())
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(num_classes))
model.add(Softmax())

model.summary()

model.compile(loss='categorical_crossentropy',
    optimizer=RMSprop(0.01),
    metrics=['accuracy'])

start = time()
hist = model.fit(x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test),
    verbose=1)
end = time()
print(f"Total training time ({epochs} epochs): {end - start:.2f}s")

start = time()
score = model.evaluate(x_test, y_test, verbose=1)
end = time()
print(f"Total evaluation time: {end - start:.2f}s")
print(f"Test loss: {score[0]:.3f}")
print(f"Test acc: {score[1]:.3f}")
