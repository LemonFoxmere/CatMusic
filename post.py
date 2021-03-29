import os
import math
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio
from numpy.fft import fft, ifft
import time as systemClock
from tqdm import tqdm
import tensorflow as tf
import keras
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import keras.backend as K
from keras.optimizers import Adam, SGD, RMSprop
from tensorflow.python.client import device_lib
from datetime import datetime

compression_size = 10

absolute_path = '/home/lemonorange/catRemix/'
attempt = 'attempt4'
data_path = os.path.join(absolute_path, 'LSTMmodels', attempt, 'firsttry.h5')
loss_data_path = os.path.join(absolute_path, 'LSTMmodels', attempt, 'loss_time_graph.txt')
fin = open(loss_data_path)
loss_data = list(map(lambda a : np.float32(a.strip('>').strip(']').split(', ')[-1].split('=')[-1].strip('>')), fin.readline().strip('[').strip(']').strip('\n').split(', <')))

loss_data1 = list(map(lambda a : np.float32(a.strip('>').strip(']').split(', ')[-1].split('=')[-1].strip('>')), fin.readline().strip('[').strip(']').strip('\n').split(', <')))

loss_data2 = list(map(lambda a : np.float32(a.strip('>').strip(']').split(', ')[-1].split('=')[-1].strip('>')), fin.readline().strip('[').strip(']').strip('\n').split(', <')))

loss_data3 = list(map(lambda a : np.float32(a.strip('>').strip(']').split(', ')[-1].split('=')[-1].strip('>')), fin.readline().strip('[').strip(']').strip('\n').split(', <')))

x = np.arange(len(loss_data))
zero = np.zeros(x.shape[0])
plt.plot(x, loss_data, color='blue', label='Train BINCE')
plt.plot(x, loss_data1, color='green', label='Train MSE')
plt.plot(x, loss_data2, color='orange', label='Val BINCE')
plt.plot(x, loss_data3, color='magenta', label='Val MSE')
plt.plot(x, zero, color='red', label='optimal loss')
plt.legend()
plt.show()

def read_input(path):
    fin = open(path)
    # assuming above command runs successfully
    data = np.array(list(map(int, fin.readline().split())))
    fin.close()
    return data

model = keras.models.load_model(data_path)
input_path = os.path.join(absolute_path, 'LSTMreadyData', 'input')
input_file_name = os.listdir(input_path)[0]
input_file_path = os.path.join(input_path, input_file_name)
input_data = read_input(input_file_path)
input_data = input_data[:220500]
input_data = tf.reshape(input_data, (int(input_data.shape[0]/compression_size), 1, compression_size))

gen_out = model(input_data)
sample = gen_out[10]
sample = np.reshape(sample, [-1])
plt.plot(np.arange(sample.shape[0]), sample, color='red', label='Val MSE')
sample = sample.numpy()
sample = np.reshape(sample, (128))

threshold = 0.9

for i in range(sample.shape[0]):
    if sample[i] < threshold:
        sample[i] = 0
    else:
        sample[i] = 1

sample
