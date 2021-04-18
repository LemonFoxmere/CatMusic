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

split_time = 5 # in seconds

absolute_path = '/home/lemonorange/catRemix/LSTMreadyData/'
# read in rawMid files
print("[DEBUG] generating raw output data with sample timeframe of " + str(split_time) + " seconds")
sourceMid_path = os.path.join(absolute_path, '..', 'gen_raw_midi')
for midFile in tqdm(os.listdir(sourceMid_path)):
    if(midFile.split('.')[1] != 'rawMid'): continue
    fin = open(os.path.join(sourceMid_path, midFile))
    tempo = int(fin.readline())
    # calculate the splitting points
    clock_period = ((tempo/1000000)/24)/16
    reps = int(split_time / clock_period)
    counterThingIdek = 0
    file_output_stream = str(tempo)+'\n'
    for line in fin:
        if(line == 'eof'): break # if at the end of line, no matter what it has collected, it is not long enough to last exactly 5 seconds, and precision is key here. discard all rest of unfinished data and move on
        start_time = int(line.split(',')[0]) - reps*counterThingIdek
        end_time = int(line.split(',')[1]) - reps*counterThingIdek
        data = line.split(',')[2]
        if(start_time <= reps and end_time >= reps):
            for i in range(start_time, reps+1):
                file_output_stream += str(i) + ','  + data
            new_output_name = midFile.split('.')[0] + '_' + str(counterThingIdek) + '.rawOut'
            fout = open(os.path.join(absolute_path, 'output', new_output_name), 'w')
            fout.write(file_output_stream)
            fout.close()
            file_output_stream = str(tempo)+'\n'
            counterThingIdek += 1
            for i in range(0, int(line.split(',')[1])-reps*counterThingIdek):
                file_output_stream += str(i) + ','  + data
        else:
            for i in range(start_time, end_time):
                file_output_stream += str(i) + ','  + data
# for midFile in os.listdir(sourceMid_path).remove('readme.txt')[0]
print("[DEBUG] SUCCESS: raw output generation completed with no errors")

print("[DEBUG] generating raw input data with sample timeframe of " + str(split_time) + " seconds")
sourceWavPath = os.path.join(absolute_path, '..', 'augmentedData', 'wav')
instruments = os.listdir(sourceWavPath)
ins_ID = 0
# loop through every instrument
for ins in instruments:
    for file in tqdm(os.listdir(os.path.join(sourceWavPath, ins))): # list all files
        print('[DEBUG] attempting to read: ' + file)
        if(file.split('.')[1] != 'wav'): continue
        path = (os.path.join(sourceWavPath, ins, file)) # create file path
        Fs, data = wavFile.read(path) # read in file
        singleChannel = [] # prepare for data compression if necessary
        print('[DEBUG] compressing data...')
        for i in tqdm(data): # compress all data into 1 channel
            if(type(i) != np.int16):
                singleChannel.append(i[0])
                continue
            singleChannel.append(i)
        singleChannel = np.array(singleChannel) # faster processing
        split_frame = split_time * Fs
        current_data = ""
        part = 0
        print('\n[DEBUG] splitting data...')
        for i in tqdm(range(data.shape[0])):
            current_data += str(singleChannel[i]) + ' '
            if i == split_frame * (part+1):
                # interval reached,
                fout = open(os.path.join(absolute_path, 'input', file.split('.')[0] + '-' + str(ins_ID) + '-' + str(part) + '.rawWav'), 'w') # we can re-assure all sample rate are 44.1khz as they are all synthesized
                fout.write(current_data)
                current_data = str(singleChannel[i]) + ' '
                part += 1
    print("[DEBUG] All files within the \"" + ins + "\" is finished\n\n")
    ins_ID += 1
print("[DEBUG] SUCCESS: raw input generation completed with no errors")
print("[DEBUG] Program is now finished, and will now automatically shutoff. Goodbye.")
