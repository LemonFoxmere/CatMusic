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

# DATA STRUCTURE:
# rawWav_section : rawMid_section
# all rawWav are 44.1khz, and all rawMid are 500,000 tempo
absolute_path = '/home/lemonorange/catRemix/'

dataSet = {}
# read in all datasets
