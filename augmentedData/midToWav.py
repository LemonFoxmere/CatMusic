# NOTE: the files outputted with this script will match up with the generated raw midi files for the ease of compiling raw data for the LSTM network and future augmentation.
# WARNING: IF YOU ARE PLANNING TO RUN THIS ON YOUR COMPUTER, YOU MUST BE RUNNING UBTUNU 20.04 WITH TiMidity++ INSTALLED.
# WARNING: THIS CODE IS A WIP, AND MAY NOT WORK SYSTEM WIDE YET. IF YOU HAVE ANY COMTRIBUTION IDEAS, PLEASE CONTACT reallemonorange@gmail.com FOR MORE INFO
import os
import time

absolute_path = '.'
csv_path = os.path.join(absolute_path, 'midcsv', 'midi')

print('[DEBUG] conversion started. WARNING: LARGE AMOUNTS OF DATA WILL TAKE TIME TO CONVERT')
start_time = time.time()
for ins in os.listdir(csv_path): # loop thru every music and instrument catagory
    wav_path = os.path.join(absolute_path, 'wav', ins)
    for csvMid in os.listdir(os.path.join(csv_path, ins)):
        music_ID = csvMid.split('.')[0][3:]
        mid_path = os.path.join(csv_path, ins, csvMid) # extract the correct midi file
        output_wav_path = os.path.join(wav_path, music_ID) + '.wav'
        command = 'timidity ' + mid_path + ' -Ow -o ' + output_wav_path
        os.system(command)
end_time = time.time()
print('[DEBUG] SUCCESS: conversion finished in ~' + str(int((end_time-start_time)*1000000)/1000) + ' ms')
