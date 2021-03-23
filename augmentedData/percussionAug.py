import os
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavFile
from IPython.display import Audio
from numpy.fft import fft, ifft
import time as systemClock
from tqdm import tqdm
from scipy.io.wavfile import write
# %matplotlib inline

absolute_path = os.path.join('.', 'wav')

sample_drum, data_drum = wavFile.read(os.path.join(absolute_path, '..', '..', 'drum.wav')) # read in augmenters
data_drum = data_drum[:,0]
sample_drum2, data_drum2 = wavFile.read(os.path.join(absolute_path, '..', '..', 'drum2.wav'))
data_drum2 = data_drum2[:,0]

dir_amt = len(os.listdir(absolute_path))

start_time = systemClock.time()
for ins in os.listdir(absolute_path):
    ins_path = os.path.join(absolute_path, ins)
    file_amt = str(len(os.listdir(ins_path)))
    file_index = 1
    for track in os.listdir(ins_path): # loop thru every track and create augmentations
        if(track.split('.')[1] != 'wav') continue
        original_track_path = os.path.join(ins_path, track)
        sample_rate_orig, data_orig = wavFile.read(original_track_path) # read in original wav files
        data_orig = data_orig[:,0] # fix meta
        track_time = systemClock.time()
        # part 1
        print("[DEBUG] Augmenting file \"" + track + "\", part 1/2 | Progress: " + str(file_index) + "/" + file_amt + ' files; time_elapsed: ' + str(int((track_time-start_time)*1000)/1000) + 's')
        combined = []
        new_aug_1_name = track.split('.')[0] + '_0.wav'
        for i in tqdm(range(len(data_orig))):
            combined.append(data_drum[i%len(data_drum)] + data_orig[i])
        print("[DEBUG] Augment file \"" + track + "\" [OK]")
        # part 2
        track_time = systemClock.time()
        print("[DEBUG] Augmenting file \"" + track + "\", part 2/2 | Progress: " + str(file_index) + "/" + file_amt + ' files; time_elapsed: ' + str(int((track_time-start_time)*1000)/1000) + 's')
        combined2 = []
        new_aug_2_name = track.split('.')[0] + '_1.wav'
        for i in tqdm(range(len(data_orig))):
            combined2.append(data_drum2[i%len(data_drum2)] + data_orig[i])
        print("[DEBUG] Augment file \"" + track + "\" [OK]")
        print("[DEBUG] Writing files...")
        # write files
        write(os.path.join(ins_path, new_aug_1_name), sample_rate_orig, np.array(combined).astype(np.int16))
        write(os.path.join(ins_path, new_aug_2_name), sample_rate_orig, np.array(combined2).astype(np.int16))
        print("[DEBUG] SUCCESS: file writing completed with no errors\n")
        file_index += 1
