import time
import os

absolute_path = '.'
ins_names = os.listdir(absolute_path + os.sep + 'csv') # get instrument names

print('[DEBUG] conversion started with ' + str(len(ins_names)) + ' sub-directories')
start_time = time.time()
for ins in ins_names:
    csv_path = os.path.join(absolute_path,'csv',ins)
    mid_path = os.path.join(absolute_path,'midi',ins)
    files = os.listdir(csv_path)
    for file in files:
        command = 'csvmidi ' + os.path.join(csv_path, file) + ' ' + os.path.join(mid_path, (file.split('.')[0] + '.mid'))
        os.system(command)
end_time = time.time()
print('[DEBUG] SUCCESS: conversion finished in ~' + str(int((end_time-start_time)*1000000)/1000) + ' ms')
