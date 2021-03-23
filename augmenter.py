import os
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavFile
from IPython.display import Audio
from numpy.fft import fft, ifft
from tqdm import tqdm
from scipy.io.wavfile import write
%matplotlib inline

FsAug, dataAug = wavFile.read('drum.wav')
dataAug = dataAug[:,0]
print("Augmenter sampling frequency is", FsAug)

FsOrig, dataOrig = wavFile.read('bwv773.wav')
dataOrig = dataOrig[:,0]
print("Original sampling frequency is", FsOrig)

# DEPRECATED DO NOT USE
# print("Matching Sampling Rates...")
# if FsOrig != FsAug:
#     downScale = []
#     sample = []
#     repetition = FsOrig/FsAug #the amount of downScale
#     plc = 0
#     plced = 1
#     for datapoint in dataOrig:
#         # pass through high pass filter
#         sample.append(datapoint)
#         plc += 1
#         if(plc >= repetition * plced):
#             # average out sample
#             avg = np.mean(sample)
#             downScale.append(avg)
#             sample = []
#             plced += 1
#     print("Matching [OK]")
# else:
#     print("Sampling Rates Already Matched")

# now that they are the same sampling frequency we add them together
print("Augmenting Data...")
combined = []
for i in tqdm(range(len(dataOrig))):
    combined.append(dataAug[i%len(dataAug)]*1 + dataOrig[i]*1)
print("Augment [OK]")

Audio(combined, rate=FsAug)

write("testAug.wav", FsAug, np.array(combined).astype(np.int16))
