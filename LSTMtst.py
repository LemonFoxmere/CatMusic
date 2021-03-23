import tensorflow as tf
import keras
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

mnist = tf.keras.datasets.mnist
(x, y), (tx, ty) = mnist.load_data()

sample = x[0].flatten()

sample.shape
sample = tf.reshape(sample,(1,784,1))
sample.shape

model = Sequential()

model.add(LSTM(5, batch_input_shape=((None,784, 1)), activation='relu', return_sequences=True))

model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=1e-3), metrics=['accuracy'])

model.predict(sample).shape
