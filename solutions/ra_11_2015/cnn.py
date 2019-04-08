import keras
import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib
import matplotlib.pyplot as plt
from keras.layers import merge
from keras.layers.core import Lambda
from keras.models import Model
import tensorflow as tf
import os
from keras.models import load_model
from keras.datasets import mnist
from keras.layers import LeakyReLU


batch_size = 256
num_classes = 10
epochs = 4


seed = 7
numpy.random.seed(seed)

img_rows, img_columns = 28, 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
print(x_train.shape[0])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(LeakyReLU(alpha=0.2))
model.add(Dense(50, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

tbCallBack.set_model(model)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(), metrics=['accuracy'])  #Previous optimizer Adadelta()

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2, shuffle='True', validation_data=(x_test, y_test),
                    callbacks=[tbCallBack])


score = model.evaluate(x_test, y_test)

print("Test loss: ", score[0])
print("Test accuracy: ", score[1])

fig = plt.figure()
plt.subplot(2, 1, 1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')

plt.subplot(2, 1, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')

plt.tight_layout()

plt.show(fig)
model.save('anmodel.h5')
