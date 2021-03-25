# NOTE: SOME LIBRARIES ARE INCLUDED FOR FUTURE DEVELOPMENT AND ARE FOR NOW UNNECCESSARY
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavFile
from IPython.display import Audio
from numpy.fft import fft, ifft
import time as systemClock
# %matplotlib inline

print("[DEBUG] reading in files...")
all_start = systemClock.time()
start_time = systemClock.time()
ALL_FILE_PATHS = os.listdir('midcsv')
# ALL_FILE_PATHS = list(map(lambda x : os.path.join('midcsv', x), ALL_FILE_PATHS))
ALL_FILE_PATHS = list(map(lambda x : os.path.join('midcsv', x), ALL_FILE_PATHS))
ALL_FILES = []
for path in ALL_FILE_PATHS:
    ALL_FILES.append(open(path))
# file readin complete
end_time = systemClock.time()
print("[DEBUG] SUCCESS: read file successfully. Completed in " + str(int((end_time-start_time)*1000000)/1000) + ' ms')

print("[DEBUG] creating compressed rawmidi files... Total: " + str(len(ALL_FILES)) + ' files')
start_time = systemClock.time()
# iteration one: create all necessary compressed raw midi files
for fin in ALL_FILES:
    tempo = 0 # extract tempo rate. Refer to devnote for converting tempo to absolute time unit
    line = fin.readline().split() # initial read line
    # read in meta data and move to read position
    while(line[1] == '0,' or 'End_track' in line[2]):
        if('Tempo' in line[2]): # extract tempo
            tempo = int(line[3])
        line = fin.readline().split()

    # audio extraction and compression starts here
    output = ''
    time = 0 # timestamp for tracking and syncing
    noteOn = False
    stm = [] # short term memory storage
    while(line[2].strip(',') != 'End_track'):
        newTime = int(line[1].strip(','))
        noteOn = line[2].strip(',') == 'Note_on_c' # update it to on or off
        # this if statement checks whether or not
        if(len(stm) == 0 and noteOn): # this means that a new line midi line has started, append a silence
            output += str(time) + ',' + str(newTime) + ',' + '{0}\n'
            time = newTime # update the last time pointer

        if(noteOn): # if the note is pressed down
            if(len(stm) != 0): # if the note is on for this one, and the last section is not silent, it means another frequency band has started
                output += str(time) + ',' + str(newTime) + ',' + str(set(stm)).replace(' ', '').replace(',','x') + '\n'
                time = newTime # update last time pointer
            # add the current Frequency to stm
            stm.append(int(line[4].strip(','))) # add whatever frequency to stm
        else: # if it is noteOff, that means that one of the portion has ended, and we add what we have
            # first add the desired note sections to the output
            output += str(time) + ',' + str(newTime) + ',' + str(set(stm)).replace(' ', '').replace(',','x') + '\n'
            # second remove ended frequency from stm
            stm.remove(int(line[4].strip(',')))
            # third update last time pointer
            time = newTime
        line = fin.readline().split()
        # end while
    # end extraction
    output_file_name = os.path.join("gen_raw_midi", fin.name.split(os.sep)[-1].split('.')[0][3:] + '.rawMid')
    fout = open(output_file_name, 'w')
    fout.write(str(tempo)+'\n')
    fout.write(output)
    fout.write("eof")
    fout.close()
    fin.seek(0) # reset pointer for next iteration
end_time = systemClock.time()
print("[DEBUG] SUCCESS: files compression successful. Completed in " + str(int((end_time-start_time)*1000000)/1000) + ' ms')

print("[DEBUG] augmenting midi files... Total: " + str(len(ALL_FILES)) + ' files')
start_time = systemClock.time()
# iteration two: create different intrument variety
for fin in ALL_FILES:
    fin.seek(0)
    # 2 intruments for now
    output_file_name_sb1 = os.path.join("augmentedData", "midcsv", "csv", "synthBass1", fin.name.split(os.sep)[-1].split('.')[0] + '.csv')
    output_file_name_pia = os.path.join("augmentedData", "midcsv", "csv", "piano", fin.name.split(os.sep)[-1].split('.')[0] + '.csv')
    output_file_name_vio = os.path.join("augmentedData", "midcsv", "csv", "violin", fin.name.split(os.sep)[-1].split('.')[0] + '.csv')
    foutSB1 = open(output_file_name_sb1, 'w')
    foutPIA = open(output_file_name_pia, 'w')
    foutVIA = open(output_file_name_vio, 'w')
    for line in fin:
        modifyer = line.split()
        if('Program_c' in line.split()[2].replace(',','')):
            modifyer[4] = '38' # synth bass 1
            foutSB1.write(''.join(modifyer) + '\n')
            modifyer[4] = '0' # synth bass 1
            foutPIA.write(''.join(modifyer) + '\n')
            modifyer[4] = '40' # synth bass 1
            foutVIA.write(''.join(modifyer) + '\n')
            continue
        foutSB1.write(line)
        foutPIA.write(line)
        foutVIA.write(line)
    foutSB1.close()
    foutPIA.close()
    foutVIA.close()
end_time = systemClock.time()
all_end = systemClock.time()
print("[DEBUG] SUCCESS: augmentation successful. Completed in " + str(int((end_time-start_time)*1000000)/1000) + ' ms')
print("[DEBUG] Program completed in " + str(int((all_end-all_start)*1000000)/1000) + ' ms with no errors. Program will now exit.')
