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
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
import keras.backend as K
from keras.optimizers import Adam, SGD, RMSprop, Adagrad
from tensorflow.python.client import device_lib
from datetime import datetime

np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float16': lambda x: "{0:0.3f}".format(x)})

# CONSTANT DEFINITIONS
sample_rate = 44100 # sample rate
sample_duration = 5 # sample duration in seconds
compression_size = 2756 # batch input size, 1/32 note at 44100hz sr
sample_size_per_sample = 220480 # sample size per sample

t_g = np.float64(1/sample_rate)*compression_size # gen_out time constant per sample, this needs to be as precise as possible
t_t = np.float64(((500000/1000000)/24)/16) # train_out time constant per sample, this needs to be as precise as possible
period_constant = np.float64(t_t/t_g) # this needs to be as precise as possible

# METHOD DEFINITIONS
# def match_gen_output_shape(gen_out, train_out):
#     time_index = 0
#     final_data = []
#     for data_point in train_out:
#         time_index += 1
#         extration_time = np.int32((period_constant*time_index)/compression_size)-1
#         if (extration_time < gen_out.shape[0]):
#             final_data.append(gen_out[extration_time])
#             continue
#         final_data.append(gen_out[gen_out.shape[0]-1])
#     # print(final_dclearata)
#     return final_data

def match_gen_output_shape(gen_out, train_out):
    time_index = 0
    final_data = []
    for data_point in train_out:
        extration_time = math.floor((time_index*t_t)/t_g)
        if (extration_time < gen_out.shape[0]):
            final_data.append(gen_out[extration_time])
            time_index += 1
            continue
        final_data.append(gen_out[gen_out.shape[0]-1])
        time_index += 1
    # print(final_dclearata)
    return final_data

# def match_gen_output_shape(gen_out, train_out):
#     final_data = []
#     semi_final_data = []
#     generation_sample_time = sample_duration / gen_out.shape[0]
#     midi_sample_time = sample_duration / train_out.shape[0]
#
#     # initialize array of sets
#     for i in range(gen_out.shape[0]):
#         semi_final_data.append({})
#
#     for i in tqdm(range(train_out.shape[0]-1)):
#         x = math.floor((i * midi_sample_time)/generation_sample_time)
#         # after we get the index, add stuff to the hash map. Because np.ndarray are not hashable, we need its raw string format
#         try:
#             semi_final_data[x][str(train_out[i])[1:-1]] += 1
#         except:
#             semi_final_data[x][str(train_out[i])[1:-1]] = 1
#
#     # now with an array of sets, find the maximum of each and add them to final data
#     for dataset in semi_final_data:
#         max_key = ''
#         max_val = 0
#         for key in dataset.keys():
#             if(max_val < dataset[key]):
#                 max_val = dataset[key]
#                 max_key = key
#         # retrieve maximum entity
#         data = np.array(max_key.split(' '), dtype=np.float32)
#         final_data.append(data)
#     # print(final_data)
#     return final_data

def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    max_arr = max(arr)
    min_arr = min(arr)
    diff_arr = max_arr - min_arr
    for i in arr:
        temp = (((i - min_arr)*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return np.array(norm_arr)

def read_input(path):
    fin = open(path)
    # assuming above command runs successfully
    data = np.array(list(map(int, fin.readline().split())))
    # normalize all data from -1 to 1
    data = normalize(data, -1, 1)
    fin.close()
    return data

def read_output(path): # note this will return a (n,88) shaped ndarray, for that the output must be in one hot encoding format.
    fin = open(path)
    fin.readline() # read in tempo. TODO: do detection with this tempo, but for now all data are assumed to have temp=5000000
    fullData = []
    for line in fin:
        dat = list(map(int, line.split(',')[1].strip('\n').strip('{').strip('}').split('x')))
        # create onehot encoding
        # onehot = np.negative(np.ones(88, dtype=np.float32))
        onehot = np.zeros(88, dtype=np.float32)
        for i in dat:
            onehot[i-21] = 1
        fullData.append(onehot)
    fin.close() # meant to save your ram buddy
    return np.array(fullData)

def make_model():
    m = Sequential()
    ##### DENSE MODEL
    # m.add(LSTM(200, batch_input_shape=((None, 1, compression_size)), activation='selu', return_sequences=True))
    # m.add(BatchNormalization())
    # m.add(Dense(200))
    # m.add(Dropout(0.05))
    # m.add(Dense(1000, activation='relu'))
    # m.add(Dropout(0.1))
    # m.add(Dense(700, activation='relu'))
    # m.add(Dropout(0.15))
    # m.add(Dense(250, activation='relu'))
    # m.add(Dropout(0.2))
    # m.add(Dense(88, activation='sigmoid'))

    ##### BLSTM MODEL
    # m.add(LSTM(500, batch_input_shape=((None, 1, compression_size)), activation='relu', return_sequences=True))
    # m.add(LSTM(500, return_sequences=True, go_backwards=True, activation='relu'))
    # m.add(BatchNormalization())
    # m.add(LSTM(500, return_sequences=True, activation='relu'))
    # m.add(LSTM(500, return_sequences=True, go_backwards=True, activation='relu'))
    # m.add(BatchNormalization())
    # # m.add(LSTM(400, return_sequences=True, activation='relu'))
    # # m.add(LSTM(400, return_sequences=True, go_backwards=True, activation='relu'))
    # # m.add(BatchNormalization())
    # m.add(LSTM(300, return_sequences=True, activation='relu'))
    # m.add(LSTM(300, go_backwards=True, activation='relu'))
    # m.add(BatchNormalization())
    #
    # m.add(Dense(120, activation='relu'))
    # m.add(Dense(88, activation='sigmoid'))

    ##### PLAIN LSTM MODEL
    m.add(LSTM(500, batch_input_shape=((None, 1, compression_size)), activation='relu', return_sequences=True))
    m.add(BatchNormalization())
    m.add(LSTM(500, return_sequences=True, activation='relu'))
    m.add(BatchNormalization())
    m.add(LSTM(300, return_sequences=True, activation='relu'))
    m.add(BatchNormalization())
    m.add(LSTM(200, activation='relu'))
    m.add(BatchNormalization())

    m.add(Dense(120, activation='relu'))
    m.add(Dense(88, activation='sigmoid'))

    return m

# DATA STRUCTURE:
# rawWav_section : rawMid_section
# all rawWav are 44.1khz, and all rawMid are 500,000 tempo
model = make_model()
model.compile()
# opt = Adam(lr=1e-3)
opt = Adam(lr=1e-3)

# get_custom_loss = lambda inp, oup : tf.math.multiply(tf.math.multiply(tf.losses.binary_crossentropy(inp, oup), 1), tf.keras.losses.categorical_crossentropy(inp, oup))
get_custom_loss = lambda inp, oup : tf.nn.sigmoid_cross_entropy_with_logits(inp, oup)

def get_cse_loss(orig_data, gen_pred):
    return tf.losses.binary_crossentropy(orig_data, gen_pred)

def get_mse_loss(orig_data, gen_pred):
    return tf.losses.mean_squared_error(orig_data, gen_pred)

def train_step(orig_in, orig_out, val_in, val_out, epochs):
    for i in range(epochs):
        with tf.GradientTape() as tape:
            print("[DEBUG] ----- EPOCH " + str(i) + "/" + str(epochs))
            # print("[DEBUG] Predicting...")
            gen_out = model(orig_in) # make prediction
            # print("[DEBUG] Validating...")
            gen_val_out = model(val_in)

            # print("[DEBUG] Reshaping training data...")
            gen_out = tf.reshape(gen_out, (gen_out.shape[0],88))
            gen_out = match_gen_output_shape(gen_out, orig_out)
            # print("[DEBUG] Reshaping validation data...")
            gen_val_out = np.reshape(gen_val_out, (gen_val_out.shape[0],88))
            gen_val_out = np.array(match_gen_output_shape(gen_val_out, val_out))

            # print("[DEBUG] flattening training data...")
            # orig_out = tf.reshape(orig_out, [-1])
            # gen_out = tf.reshape(gen_out, [-1])
            # print("[DEBUG] flattening validation data...")
            # val_out = np.reshape(val_out, [-1])
            # gen_val_out = np.reshape(gen_val_out, [-1])

            # print("[DEBUG] Calculating loss...")
            train_bin_loss = get_cse_loss(orig_out, gen_out)
            train_cus_loss = get_custom_loss(orig_out, gen_out)
            # print(train_cus_loss)
            val_bin_loss = get_cse_loss(val_out, gen_val_out)
            val_cus_loss = get_custom_loss(val_out, gen_val_out)

            loss_over_time.append(np.mean(train_bin_loss.numpy().tolist()))
            loss_over_time1.append(np.mean(train_cus_loss.numpy().tolist()))
            loss_over_time2.append(np.mean(val_bin_loss.numpy().tolist()))
            loss_over_time3.append(np.mean(val_cus_loss.numpy().tolist()))

            # print("[DEBUG] Calculating Gradient...")
            grad = tape.gradient(train_bin_loss, model.trainable_variables)

            # print("[DEBUG] Applying Gradient...\n")
            opt.apply_gradients(zip(grad, model.trainable_variables))

            avg_loss = np.mean(train_bin_loss)
            avg_loss1 = np.mean(val_bin_loss)
            print(np.reshape(np.array(list(map(np.float16,gen_out[0]))), (8,11)))
            print('---')
            print(np.reshape(np.array(list(map(np.float16,orig_out[0]))), (8,11)))
            print("[DEBUG] model train loss: ", avg_loss, '; dimension: ' + str(train_bin_loss.shape))
            print("[DEBUG] validation loss: ", avg_loss1, '; dimension: ' + str(val_bin_loss.shape) + "\n")

        print('[DEBUG] Epochs finished')

absolute_path = '/home/lemonorange/catRemix/'
save_path = os.path.join(absolute_path, 'LSTMmodels')

quick_save_models = []

att=0
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
input_length = len(os.listdir(input_path))
for input_file_name in tqdm(os.listdir(input_path)):
    if(input_file_name.split('.')[1] == 'txt'): continue # ignore all txt files
    if(len(input_file_name.split('.')[0].split('_')) != 1): continue # exclude all files with precussions
    meta = input_file_name.replace('-', '.').split('.') # meta: name, ins, section, type
    print(input_file_name)
    if(int(meta[1]) != 2): continue # use only piano
    # input_file_name = '777-0-0.rawWav' # For overfitting test only
    if(lost_data > 50): print("[WARNING] Over 50 datasets have been lost or corrupted. Are you sure this is normal?")
    input_file_path = os.path.join(input_path, input_file_name) # generate input file path from name
    val_file_path = os.path.join(input_path, os.listdir(input_path)[-1]) # generate validation input file path (this is static, why am I repeating this operation?)
    meta_val = os.listdir(input_path)[-1].replace('-', '.').split('.') # meta_val: name, ins, section, type

    output_path = os.path.join(absolute_path, 'LSTMreadyData', 'output') # generate output absolute path
    output_file_name = meta[0].split('_')[0] + '_' + meta[2] + '.rawOut' # generate output file name
    output_file_path = os.path.join(output_path, output_file_name) # generate output file path

    # validation output file & path
    val_output_file_name = meta_val[0].split('_')[0] + '_' + meta_val[2] + '.rawOut'
    val_output_file_path = os.path.join(output_path, val_output_file_name)

    print('[DEBUG] Attempting to read in input_file [' + input_file_name + ']')
    print('--------------------',input_file_path)
    try:
        input_data = read_input(input_file_path)
    except:
        print('[WARNING] File [' + input_file_name + '] failed to read. Moving on')
        lost_data += 1
        continue
    val_input_data = read_input(val_file_path)
    val_input_data = val_input_data[:sample_size_per_sample]

    # print('[DEBUG] Attempting to trim [' + input_file_name + ']')
    if (input_data.shape[0] > sample_size_per_sample): # trim
        input_data = input_data[:sample_size_per_sample]
    elif (input_data.shape[0] < sample_size_per_sample):
        lost_data += 1
        continue

    print('[DEBUG] Attempting to read in output_file [' + output_file_name + ']')
    try:
        output_data = read_output(output_file_path)
    except:
        print('[WARNING] File [' + input_file_name + '] failed to read. Moving on')
        continue

    val_output_data = read_output(val_output_file_path)

    # print('[DEBUG] Reshaping input...')
    input_data = tf.reshape(input_data, (int(input_data.shape[0]/compression_size), 1, compression_size))
    val_input_data = tf.reshape(val_input_data, (int(val_input_data.shape[0]/compression_size), 1, compression_size))

    # print('[DEBUG] Training network...')
    train_step(input_data, output_data, val_input_data, val_output_data, 15)
    # loss = train_step(input_data, output_data)

    processed_data += 1

    print('\n\n[DEBUG] Currently ' + str(processed_data) + ' files processed. Lost ' + str(lost_data) + ' files.')
    lap_time = systemClock.time()
    # print('[DEBUG] Program has been running non-stop for ' + str(int(lap_time - start_time)/60) + ' minutes now. You should probably go to sleep.')

    if(processed_data >= 10*save_time):
        att+=1
        save_name = 'att' + str(att) + '.h5'
        # save current archetechture as h5 in the appointed directory
        model.save(os.path.join(save_path, save_name))
        quick_save_models.append(model)
        save_time += 1
        print('[DEBUG] Network Attempt just saved! Glad you came so far ;)')
end_time = systemClock.time()
save_name = 'final_att.h5'
model.save(os.path.join(save_path, save_name))

print('[DEBUG] Program finished with ' + str(processed_data) + ' files processed. Lost ' + str(lost_data) + ' files. Program runtime: ' + str(int(lap_time - start_time)/(60**2)) + ' hours')
print('[DEBUG] Final loss: ' + str(loss_over_time[-1]))
fout = open(os.path.join(save_path, 'loss_time_graph.txt'), 'w')
fout.write(str(loss_over_time) + '\n')
fout.write(str(loss_over_time1) + '\n')
fout.write(str(loss_over_time2) + '\n')
fout.write(str(loss_over_time3) + '\n')
print('[DEBUG] Program is now finished and will automatically shutdown. Goodbye.')
