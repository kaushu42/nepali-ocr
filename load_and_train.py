import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint
import keras
np.random.seed(1)

def load_data():
    X = np.load('dataset/xdata.npy')
    X = X / 255
    Y = np.load('dataset/ydata.npy')
    from sklearn.model_selection import train_test_split
    xtr, xte, ytr, yte = train_test_split(X, Y, test_size = 0.2, random_state = 1)
    return xtr, ytr, xte, yte

x_train, y_train, x_test, y_test = load_data()

from sklearn.preprocessing import OneHotEncoder
y_train = OneHotEncoder().fit_transform(y_train)
y_test = OneHotEncoder().fit_transform(y_test)

model = Sequential()
model.add(Dense(2000, input_shape = (1024,), activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1000, activation = 'relu'))
model.add(Dense(500, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(250, activation = 'relu'))
model.add(Dense(125, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(46, activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

filepath="weights-improvement-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.load_weights('weights-improvement-02.hdf5')

model.fit(x_train, y_train, validation_split = 0.2, callbacks = callbacks_list, epochs = 100, batch_size = 10)

score = model.evaluate(x_test, y_test)
print("Test score:", score[0])
print('Test accuracy:', score[1])

model.save_weights('./weights.h5')
