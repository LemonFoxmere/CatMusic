import tensorflow as tf
import keras
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
mnist = tf.keras.datasets.mnist
(x, y), (tx, ty) = mnist.load_data()

type(x)
x.shape
type(y)
y.shape

sample = x[0].flatten()

sample.shape
sample = tf.reshape(sample,(1,784,1))
sample.shape

x.shape
bruhx = x[:300]
xx = x[300:600]
f_x = tf.reshape(bruhx,(300,1,784))
more_x = tf.reshape(xx,(300,1,784))

a = [[0.,0.,0.,0.,0.,0.,1.,0.],[0.,0.,0.,0.,0.,0.,1.,0.],[0.,0.,0.,0.,0.,0.,1.,0.]]
b = [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]

def getCustomLoss(train_in, gen_in):
    match = 0
    unmatch = 0
    min_thre = 0.3
    max_thre = 0.9


tf.losses.binary_crossentropy(a, b).numpy()
np.average(tf.losses.mean_squared_error(a, b).numpy())

f_x.shape
more_x.shape

model = Sequential()

model.add(LSTM(784, batch_input_shape=((None, 1, 784)), activation='tanh', return_sequences=True))
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=1e-3), metrics=['accuracy'])

sample_output = model(np.array([f_x, more_x]))
sample_output.shape
