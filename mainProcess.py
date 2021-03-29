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

# CONSTANT DEFINITIONS
t_g = np.float64(1/44100) # gen_out time constant per sample, this needs to be as precise as possible
t_t = np.float64(((500000/1000000)/24)/16) # train_out time constant per sample, this needs to be as precise as possible
period_constant = np.float64(t_t/t_g) # this needs to be as precise as possible
compression_size = 2

# METHOD DEFINITIONS
def match_gen_output_shape(gen_out, train_out):
    time_index = 0
    final_data = []
    for data_point in train_out:
        time_index += 1
        extration_time = np.int32((period_constant*time_index)/compression_size)-1
        if (extration_time < gen_out.shape[0]):
            final_data.append(gen_out[extration_time])
            continue
        final_data.append(gen_out[gen_out.shape[0]-1])
    return final_data

def read_input(path):
    fin = open(path)
    # assuming above command runs successfully
    data = np.array(list(map(int, fin.readline().split())))
    fin.close() # meant to save your ram buddy
    return data

def read_output(path): # note this will return a (n,128) shaped ndarray, for that the output must be in one hot encoding format.
    fin = open(path)
    fin.readline() # read in tempo. TODO: do detection with this tempo, but for now all data are assumed to have temp=5000000
    fullData = []
    for line in fin:
        dat = list(map(int, line.split(',')[1].strip('\n').strip('{').strip('}').split('x')))
        # create onehot encoding
        onehot = np.negative(np.ones(128, dtype=np.float32))
        for i in dat:
            onehot[i] = 1
        fullData.append(onehot)
    fin.close() # meant to save your ram buddy
    return np.array(fullData)

def make_model():
    m = Sequential()
    m.add(LSTM(128, batch_input_shape=((None, 1, compression_size)), activation='tanh', return_sequences=True))
    return m

# DATA STRUCTURE:
# rawWav_section : rawMid_section
# all rawWav are 44.1khz, and all rawMid are 500,000 tempo
model = make_model()
model.compile(metrics=['accuracy'])
opt = Adam(lr=1e-3, decay=1e-3)

def get_bin_loss(orig_data, gen_pred):
    return tf.losses.binary_crossentropy(orig_data, gen_pred)

def get_mse_loss(orig_data, gen_pred):
    return tf.losses.mean_squared_error(orig_data, gen_pred)

def train_step(orig_in, orig_out, val_in, val_out, epochs):
    with tf.GradientTape() as tape:
        for i in (range(epochs)):
            print("[DEBUG] Predicting...")
            gen_out = model(orig_in) # make prediction
            print("[DEBUG] Validating...")
            gen_val_out = model(val_in)
            print("[DEBUG] Reshaping training data...")
            gen_out = tf.reshape(gen_out, (gen_out.shape[0],128))
            gen_out = match_gen_output_shape(gen_out, orig_out)
            print("[DEBUG] Reshaping validation data...")
            gen_val_out = np.reshape(gen_val_out, (gen_val_out.shape[0],128))
            gen_val_out = np.array(match_gen_output_shape(gen_val_out, val_out))

            # print("[DEBUG] flattening training data...")
            # orig_out = tf.reshape(orig_out, [-1])
            # gen_out = tf.reshape(gen_out, [-1])
            # print("[DEBUG] flattening validation data...")
            # val_out = np.reshape(val_out, [-1])
            # gen_val_out = np.reshape(gen_val_out, [-1])

            print("[DEBUG] Calculating loss...")
            train_bin_loss = get_bin_loss(gen_out, orig_out)
            train_mse_loss = get_mse_loss(orig_out, gen_out)
            val_bin_loss = get_bin_loss(val_out, gen_val_out)
            val_mse_loss = get_mse_loss(val_out, gen_val_out)

            loss_over_time.append(train_bin_loss)
            loss_over_time1.append(train_mse_loss)
            loss_over_time2.append(val_bin_loss)
            loss_over_time3.append(val_mse_loss)

            print("[DEBUG] Calculating Gradient...")
            grad = tape.gradient(train_bin_loss, model.trainable_variables)

            print("[DEBUG] Applying Gradient...")
            opt.apply_gradients(zip(grad, model.trainable_variables))

            avg_loss = np.mean(train_bin_loss)
            avg_loss1 = np.mean(val_bin_loss)
            print("[DEBUG] SUCCESS; model loss: ", avg_loss, '; dimension: ' + str(train_bin_loss.shape))
            print("[DEBUG] -------- validation loss: ", avg_loss1, '; dimension: ' + str(val_bin_loss.shape))

absolute_path = '/home/lemonorange/catRemix/'
save_path = os.path.join(absolute_path, 'LSTMmodels')

quick_save_models = []

lost_data = 0
processed_data = 0
loss_over_time = []
loss_over_time1 = []
loss_over_time2 = []
loss_over_time3 = []
input_path = os.path.join(absolute_path, 'LSTMreadyData', 'input')
# read in all datasets
save_time = 1
start_time = systemClock.time()
input_length = len(os.listdir(input_path)[:500])
for input_file_name in tqdm(os.listdir(input_path)[:500]):
    if(input_file_name.split('.')[1] == 'txt'): continue # ignore all txt files
    if(lost_data > 50): print("[WARNING] Over 50 datasets have been lost or corrupted. Are you sure this is normal?")
    input_file_path = os.path.join(input_path, input_file_name)
    val_file_path = os.path.join(input_path, os.listdir(input_path)[-1])
    meta = input_file_name.replace('-', '.').split('.') # meta: name, ins, section, type
    meta_val = os.listdir(input_path)[-1].replace('-', '.').split('.') # meta_val: name, ins, section, type

    output_path = os.path.join(absolute_path, 'LSTMreadyData', 'output')
    output_file_name = meta[0].split('_')[0] + '_' + meta[2] + '.rawOut'
    output_file_path = os.path.join(output_path, output_file_name)

    val_output_file_name = meta_val[0].split('_')[0] + '_' + meta_val[2] + '.rawOut'
    val_output_file_path = os.path.join(output_path, val_output_file_name)

    print('[DEBUG] Attempting to read in input_file [' + input_file_name + ']')
    try:
        input_data = read_input(input_file_path)
    except:
        print('[WARNING] File [' + input_file_name + '] failed to read. Moving on')
        lost_data += 1
        continue
    val_input_data = read_input(val_file_path)
    val_input_data = val_input_data[:220500]

    print('[DEBUG] Attempting to trim [' + input_file_name + ']')
    if (input_data.shape[0] > 220500): # trim
        input_data = input_data[:220500]
    elif (input_data.shape[0] < 220500):
        lost_data += 1
        continue

    print('[DEBUG] Attempting to read in output_file [' + output_file_name + ']')
    try:
        output_data = read_output(output_file_path)
    except:
        print('[WARNING] File [' + input_file_name + '] failed to read. Moving on')
        continue

    val_output_data = read_output(val_output_file_path)

    print('[DEBUG] Reshaping input...')
    input_data = tf.reshape(input_data, (int(input_data.shape[0]/compression_size), 1, compression_size))
    val_input_data = tf.reshape(val_input_data, (int(val_input_data.shape[0]/compression_size), 1, compression_size))

    print('[DEBUG] Training network...')
    train_step(input_data, output_data, val_input_data, val_output_data, 1)
    # loss = train_step(input_data, output_data)

    processed_data += 1

    print('[DEBUG] Currently ' + str(processed_data) + ' files processed. Lost ' + str(lost_data) + ' files. Only ' + str(input_length-processed_data-lost_data) + ' to go!')
    lap_time = systemClock.time()
    print('[DEBUG] Program has been running non-stop for ' + str(int(lap_time - start_time)/60) + ' minutes now. You should probably go to sleep.')

    if(processed_data >= 25*save_time):
        save_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f") + '_net_atempt.h5'
        # save current archetechture as h5 in the appointed directory
        model.save(os.path.join(save_path, save_name))
        quick_save_models.append(model)
        save_time += 1
        print('[DEBUG] Network Attempt just saved! Glad you came so far ;)')
end_time = systemClock.time()
save_name = 'final_net_atempt.h5'
model.save(os.path.join(save_path, save_name))

print('[DEBUG] Phew! Program finished with ' + str(processed_data) + ' files processed. Lost ' + str(lost_data) + ' files. It only ' + str(int(lap_time - start_time)/(60**2)) + ' hours!')
print('[DEBUG] Final loss: ' + str(loss_over_time[-1]))
fout = open(os.path.join(save_path, 'loss_time_graph.txt'), 'w')
fout.write(str(loss_over_time) + '\n')
fout.write(str(loss_over_time1) + '\n')
fout.write(str(loss_over_time2) + '\n')
fout.write(str(loss_over_time3) + '\n')
print('[DEBUG] Program is now finished and will automatically shutdown. Goodbye.')
