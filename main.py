import sys

import numpy as np

import matplotlib.pyplot as plt

import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split

import characters as c

X = np.load('./dataset/xdata.npy').reshape(-1, 32, 32, 1)
Y = np.load('./dataset/ydata.npy')
Y = to_categorical(Y).reshape(-1, 1, 1, 47)

x_train, x_test, y_train, y_test = train_test_split(X, Y)
del X, Y

if len(sys.argv) == 1:
    print('Usage:\npython3 main.py {train/test}')
    exit()
if sys.argv[1] == 'train':
    model = Sequential()
    model.add(
        Conv2D(
            4,
            kernel_size=(5, 5),
            strides=(1, 1),
            input_shape=(32, 32, 1),
        ))
    model.add(Conv2D(4, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same'))
    model.add(Conv2D(8, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(
        Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(
        Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(
        Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(
        Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(
        Conv2D(47, kernel_size=(1, 1), strides=(1, 1), activation='softmax'))
    print(model.summary())
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    filepath = "./weights/temp/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='val_acc',
        verbose=1,
        save_best_only=True,
        mode='max')
    callbacks_list = [checkpoint]
    model.fit(
        x_train,
        y_train,
        batch_size=200,
        epochs=50,
        validation_split=0.33,
        callbacks=callbacks_list)
    model.save('./weights/model.h5')
    print(model.evaluate(x_test, y_test))
elif sys.argv[1] == 'test':
    try:
        model = load_model('./weights/model.h5')
        print('Model Loaded')
    except:
        print('Cannot Load Model. Train first.')
    else:
        images = x_test[:100]
        predictions = model.predict(images)
        predictions = predictions.argmax(axis=-1).reshape(-1, 1)
        for (image, prediction) in zip(images, predictions):
            plt.imshow(image.reshape(32, 32), cmap='gray')
            plt.show()
            print(c.characters[int(prediction) - 1])
else:
    print('Invalid argument')