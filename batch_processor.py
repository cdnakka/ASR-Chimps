import numpy as np
import os
from pathlib import Path
import tensorflow as tf
import librosa
from keras.models import load_model, Sequential
from keras.layers import Dense, Activation, Dropout


def load_sound_files(file_paths):
    x, sr = librosa.load(file_paths, sr=11025)
    mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=40)
    return mfccs


class Predict:

    def __init__(self, path, threshold):
        self.path = path
        self.thresh = threshold

    def fetch_model(self):

        inf_model = Sequential()

        inf_model.add(Dense(256, input_shape=(40,)))
        inf_model.add(Activation('relu'))
        inf_model.add(Dropout(0.5))

        inf_model.add(Dense(256))
        inf_model.add(Activation('relu'))
        inf_model.add(Dropout(0.5))

        inf_model.add(Dense(256))
        inf_model.add(Activation('relu'))
        inf_model.add(Dropout(0.5))

        inf_model.add(Dense(12))
        inf_model.add(Activation('softmax'))

        inf_model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

        model_path = Path(self.path) / "Model" / "AUG_model.h5"
        inf_model = load_model(model_path)

        return inf_model

    def fetch_sounds(self):
        sound_paths = []
        for file in os.listdir(self.path):
            if file.endswith(".wav"):
                sound_paths.append(os.path.join(curdir, file))
        return sound_paths

    def predict_classes(self):
        model = self.fetch_model()
        all_preds = {}
        for i_file in self.fetch_sounds():
            ind_preds = []
            sf = load_sound_files(i_file)
            n = sf.shape[1]
            if n > 87:
                for i_sec in range(int(n/87)):
                    sf_chunk = sf[:, i_sec*87: (i_sec+1) * 87]
                    sf_chunk = np.array(sf_chunk).reshape(1, 40, 87, 1)
                    chunk_pred = model.predict_proba(np.array(sf_chunk))
                    ind_preds.append([i_sec, model.predict_classes(np.array(sf_chunk))[0], max(chunk_pred[0])])

            all_preds[i_file] = ind_preds

        return all_preds

    def generate_log(self):
        cleaned_preds = {}
        preds = self.predict_classes()
        for key in preds.keys():
            values = []
            for i_secs in range(len(preds[key])):
                if preds[key][i_secs][1] in [10, 11] and preds[key][i_secs][2] > self.thresh:
                    values.append(np.array(preds[key][i_secs]).tolist())

            cleaned_preds[key] = values

        return cleaned_preds


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    curdir = os.getcwd()
    i_t = 0.5
    a = Predict(curdir, i_t)
    out_dict = a.generate_log()

    for k, v in out_dict.items():
        text_file = open("log.txt", "w")
        text_file.write(k + "->" + str(v) + "\n")
        text_file.close()
