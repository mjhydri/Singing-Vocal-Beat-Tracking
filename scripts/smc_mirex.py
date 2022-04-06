import numpy as np
import librosa
import jams
import os
import csv
from cqt import CQT
import pickle


class SMC_MIREX:
    def __init__(self, hop_length, sample_rate=22050, base_dir=None, splits=None, train_test_ratio="all_for_train",
                 shuffle=None, data_proc=[CQT()]):
        self.data_proc = data_proc  # what feature to extract
        self.sample_rate = sample_rate  # sample rate
        self.hop_length = hop_length  # hop
        self.shuffle = shuffle
        self.train_test_ratio = train_test_ratio
