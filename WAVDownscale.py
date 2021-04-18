import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavFile
from IPython.display import Audio
from numpy.fft import fft, ifft
# %matplotlib inline

# file path may differ
Fs, data = wavFile.read('/home/lemonorange/catRemix/augmentedData/wav/synthBass1/772.wav')
data = data[:,0]

data.tolist()

print("sampling Frequency is", Fs)
downScale = []
sample = []
repetition = 5 #the amount of downScale
plc = 0

# asdf = range(-200,200)
# x=[]
# for i in asdf:
#     x.append(i/100)
# hpFilter = lambda x : 1-(math.tanh(x)**2)
# y = []
# for i in x:
#     y.append(hpFilter(i))
# plt.plot(x, y, 'r')

for datapoint in data:
    # pass through high pass filter
    sample.append(datapoint)
    plc += 1
    if(plc == repetition):
        # average out sample
        avg = np.mean(sample)
        for i in range(repetition):
            downScale.append(avg)
        sample = []
        plc = 0

downScale = np.array(downScale, dtype="float32")

Audio(data, rate=Fs)
plt.figure()
plt.plot(downScale)

fout = open("downScaledTest.rawaud", 'w')
audioStr = ''
lastPhrase = None;
for data in downScale:
    if(lastPhrase != data):
        if(lastPhrase != None):
            audioStr += str(lastPhrase) + ' '
        lastPhrase = data
fout.write(str(Fs) + '/' + str(repetition) + '/' +  str(downScale.shape[0]) + '\n')
fout.write(audioStr)
fout.close()
