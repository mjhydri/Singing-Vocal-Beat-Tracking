from random import shuffle as sh
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict
import pickle
from wav_lm import WAV_LM
from distil_hubert import DISTILHUBERT
from log_spect import LOG_SPECT
import librosa
import torch
import os


class DATA_HANDLER(Dataset):
    def __init__(self, data, data_proc,  device='cpu', seq_len=None, root_dir=None, data_dir=None, random_sample=True):
        if 'wavlm' in data_proc:
            self.processor = WAV_LM(device=device,
                                    pretrained_dir=os.path.join(root_dir, 'scripts', 'pretrained_models', 'wavlm', 'WavLM-Base+.pt'))
        if 'distilhubert' in data_proc:
            self.processor = DISTILHUBERT(device=device)

        if 'log_spec' in data_proc:
            self.processor = LOG_SPECT(sample_rate=16000, win_length=1024, hop_size=320, n_bands=[24], device=device)

        self.data_proc = data_proc
        self.seq_len = seq_len
        self.data = data
        self.base_dir = data_dir
        self.root_dir = root_dir
        self.rng = np.random.RandomState(0)
        self.random_sample = random_sample

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        dataset = self.data[i].split('#')[0]
        with open(self.base_dir + "/data/" + self.data[i], 'rb') as f:
            data = pickle.load(f)
        out = {}
        if self.seq_len is not None:
            if len(data['data']) < self.seq_len * 320:
                data['data'] = np.pad(data['data'], (self.seq_len * 320 - len(data['data']), 0), 'constant')
                data['gt'] = np.pad(data['gt'], (self.seq_len - len(data['gt']), 0), 'constant')

            # to make sure the sample contains sufficient annotations
            for j in range(10):
                if self.random_sample:
                    frame_start = self.rng.randint(0, int(data['data'].shape[-1]/320) - self.seq_len + 1)
                else:
                    frame_start = 0
                frame_end = frame_start + self.seq_len
                times = data["times"][data["times"] > frame_start/50]
                times = times[times < frame_end / 50]
                if len(times) > 3:
                    break
            frame_start_s = frame_start * 320
            frame_end_s = frame_end * 320

            out["embeddings"] = self.processor.process_audio(data['data'][..., frame_start_s: frame_end_s])
            out["gt"] = torch.from_numpy(data["gt"][..., frame_start: frame_start + np.shape(out["embeddings"])[1]])
            if 'log_spec' in self.data_proc:
                out["embeddings"]=out["embeddings"][...,:self.seq_len -1]
                out["gt"] = torch.from_numpy(data["gt"][..., frame_start: frame_start+np.shape(out["embeddings"])[1]])[...,:self.seq_len -1]
        else:
            out["embeddings"] = self.processor.process_audio(data['data'])
            out["gt"] = torch.from_numpy(data["gt"])
            if 'log_spec' in self.data_proc:
                lenn = min(np.shape(out["embeddings"])[-1], len(out["gt"]))
                out["embeddings"]=out["embeddings"][...,:lenn]
                out["gt"] = torch.from_numpy(data["gt"])[...,:lenn]


        return out
