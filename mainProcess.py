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

# METHOD DEFINITIONS
def read_input(path):
    fin = open(path)
    # assuming above command runs successfully
    data = np.array(list(map(int, fin.readline().split())))
    fin.close()
    return data

def read_output(path): # note this will return a (n,128) shaped ndarray, for that the output must be in one hot encoding format.
    fin = open(path)
    fin.readline() # read in tempo. TODO: do detection with this tempo, but for now all data are assumed to have temp=5000000
    fullData = []
    for line in fin:
        dat = list(map(int, line.split(',')[1].strip('\n').strip('{').strip('}').split('x')))
        # create onehot encoding
        onehot = np.zeros(128, dtype=np.float32)
        for i in dat:
            onehot[i] = 1
        fullData.append(onehot)
    fin.close()
    return np.array(fullData)

# DATA STRUCTURE:
# rawWav_section : rawMid_section
# all rawWav are 44.1khz, and all rawMid are 500,000 tempo
absolute_path = '/home/lemonorange/catRemix/'

lost_data = 0
input_path = os.path.join(absolute_path, 'LSTMreadyData', 'input')
# read in all datasets
input_file_name = os.listdir(input_path)[0]
# if(input_file_name.split('.')[1] == 'txt'): continue
input_file_path = os.path.join(input_path, input_file_name)
meta = input_file_name.replace('-', '.').split('.') # meta: name, ins, section, type

output_path = os.path.join(absolute_path, 'LSTMreadyData', 'output')
output_file_name = meta[0].split('_')[0] + '_' + meta[2] + '.rawOut'
output_file_path = os.path.join(output_path, output_file_name)

input_data = read_input(input_file_path)[:-1]
output_data = read_output(output_file_path)
input_data = tf.reshape(input_data, (int(input_data.shape[0]/1), 1, 1))

model = Sequential()
model.add(LSTM(784, batch_input_shape=((None, 1, 1)), activation='relu', return_sequences=True))
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=1e-3), metrics=['accuracy'])

model.predict(input_data).shape
