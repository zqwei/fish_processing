'''
Copyright 2018 Wei, Ziqiang, Janelia Research Campus

weiz@janelia.hhmi.org

Acknowledgement for example code
https://gist.github.com/philipperemy/b8a7b7be344e447e7ee6625fe2fdd765
'''
import numpy as np
import deprecation
from tensorflow.keras.layers import RepeatVector, Bidirectional, TimeDistributed
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
# import keras
# print(keras.__version__)


def prepare_sequences_center(x_train, y_train, window_length,peak_wid=0):
    full_seq = x_train.flatten()
    target_seq = y_train.flatten()
    windows = []
    outliers = []
    for window_start in range(0, len(full_seq) - window_length + 1):
        window_end = window_start + window_length
        window_range = range(window_start, window_end)
        window_mid = int(round(np.median(window_range)))
        window = list(full_seq[window_range])
        contain_outlier = target_seq[window_mid-peak_wid:window_mid+peak_wid+1].sum()>0
        outliers.append(contain_outlier)
        windows.append(window)
    return np.expand_dims(np.array(windows), axis=2), np.array(outliers).astype(np.bool)


def create_lstm_model(hidden_dim, window_length, m=1):
    model = Sequential()
    model.add(Bidirectional(LSTM(hidden_dim, return_sequences=True, use_bias=True), input_shape=(window_length, m)))
    model.add(Dropout(rate=0.2))
    model.add(Bidirectional(LSTM(hidden_dim//2, return_sequences=False, use_bias=True)))
    model.add(Dense(1, activation='sigmoid', use_bias=True)) # bias_initializer='zeros'
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


@deprecation.deprecated(details="Use create_lstm_model function instead")
def _create_lstm_model(hidden_dim, window_length, m=1):
    model = Sequential()
    model.add(Bidirectional(LSTM(hidden_dim, return_sequences=True), input_shape=(window_length, m)))
    model.add(Dropout(rate=0.2))
    model.add(Bidirectional(LSTM(hidden_dim, return_sequences=False)))
    model.add(Dense(1, activation='sigmoid'))
    # model.compile(loss='binary_crossentropy', optimizer='adam')
    model.compile(loss='kullback_leibler_divergence', optimizer='adam')
    return model


@deprecation.deprecated(details="Use prepare_sequences_center function instead")
def prepare_sequences(x_train, y_train, window_length):
    full_seq = x_train.flatten()
    target_seq = y_train.flatten()
    windows = []
    outliers = []
    for window_start in range(0, len(full_seq) - window_length + 1):
        window_end = window_start + window_length
        window_range = range(window_start, window_end)
        window = list(full_seq[window_range])
        contain_outlier = target_seq[window_range].sum()
        outliers.append(contain_outlier)
        windows.append(window)
    return np.expand_dims(np.array(windows), axis=2), np.array(outliers).astype(np.bool)