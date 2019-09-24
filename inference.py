import numpy as np
import time
from pydub import AudioSegment
import random
import sys
import io
import os
import glob
import pyaudio
import wave
import os
import librosa
import IPython
from td_utils import *
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
# To generate wav file from np array.
from scipy.io.wavfile import write
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam

inf_model = Sequential()

inf_model.add(Dense(256, input_shape=(40,)))
inf_model.add(Activation('relu'))
inf_model.add(Dropout(0.5))

inf_model.add(Dense(256))
inf_model.add(Activation('relu'))
inf_model.add(Dropout(0.5))

# try_model.add(Dense(256))
# try_model.add(Activation('relu'))
# try_model.add(Dropout(0.5))

inf_model.add(Dense(256))
inf_model.add(Activation('relu'))
inf_model.add(Dropout(0.5))

inf_model.add(Dense(12))
inf_model.add(Activation('softmax'))

inf_model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

inf_model = load_model("my_model.h5")
inf_model.summary()

def load_sound_files(file_paths):

    X,sr = librosa.load(file_paths, sr=11025)
    #, res_type='kaiser_fast'
    mfccs=np.mean(librosa.feature.mfcc(y=X,sr=sr,n_mfcc=40).T,axis=0)

    return mfccs

CHUNK = 4096
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 20
WAVE_OUTPUT_FILENAME = "output"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index = 2,
                frames_per_buffer=CHUNK)

print("* recording")

for j in range(0,RECORD_SECONDS):
    frames = []
    for i in range(int(RATE / CHUNK * j), int(RATE / CHUNK * (j+4))):
        data = stream.read(CHUNK, exception_on_overflow = False)
        frames.append(data)
    wf = wave.open("output_{}.wav".format(j), 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    inf = []
    inf.append(load_sound_files("output_{}.wav".format(j)))
    inf_pred = inf_model.predict_classes(np.array(inf))
    print(inf_pred)
    if inf_pred[0] == 11:
        print("Chimp!")
        
print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()
