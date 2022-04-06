import os
import librosa
import torchaudio
from typing import List, Tuple, Optional
from wav_lm import WAV_LM
import numpy as np
import pickle
import scipy as sp
import torchaudio
import torch
import random
import math
import soundfile as sf
from random import shuffle as sh

# from torchaudio_augmentations import *
# from audio_augmentations import *
########################################################

def data_splitter(datasets, ratio=0.8, shuffle='None'):
    splits = {}
    train=[]
    test=[]
    list_all = []
    for dataset in datasets:
        basedir = 'C:\datasets/'
        audio_dir = basedir + dataset + '/vocal/'
        annot_dir = basedir + dataset + '/vocal_annotations/'
        for entry in os.scandir(annot_dir):
            if os.path.isfile(audio_dir + entry.name + '.wav'):
                list_all.append(dataset+'#'+entry.name)
                if dataset == 'GTZAN':
                    test.append(dataset+'#'+entry.name)
                else:
                    train.append(dataset+'#'+entry.name)
    if shuffle:
        sh(train)
    splits['val'] = train.copy()[round(ratio*len(train)):]
    splits['train'] = train[:round(ratio*len(train))]
    splits['test'] = test
    with open(basedir + 'vocal_data/' +'list_all', 'wb') as f:
        pickle.dump(list_all, f)
        f.close()
    with open(basedir + 'vocal_data/' + 'splits', 'wb') as f:
        pickle.dump(splits, f)
        f.close()

# list=data_splitter(['Ballroom','Hainsworth','musdb18','rock_corpus','RWC_pop','RWC_royalty_free','URSing','GTZAN'],shuffle=True)

##################################################################

def data_preparer(dataset=None,chunk_length_seconds=8,splits=None):
    # model = WAV_LM(r'C:\research\vocal_beat\wavlm\WavLM-Base+.pt')
    basedir = 'C:\datasets/'
    audio_dir = basedir + dataset + '/vocal/'
    annot_dir = basedir + dataset + '/vocal_annotations/'
    chunk_length_f = chunk_length_seconds * 50  # sequence lengths seconds * fps = (frames)
    chunk_length_s = chunk_length_seconds * 16000  # sequence lengths (samples)
    data = {}
    counter = 0
    over_sampled_training_list = []
    for entry in os.scandir(annot_dir):
        if os.path.isfile(audio_dir + entry.name + '.wav'):
            counter += 1
            print(f'number:{counter}   dataset:{dataset}   song{entry.name}')
            audio, _ = librosa.load(audio_dir + entry.name + '.wav', sr=16000)
            ##  preparing ground truth frames
            lenn = int(len(audio) * 50 / 16000)-1   # len audio (frames)
            file = open(annot_dir + entry.name, 'rb')
            times = pickle.load(file)
            beat_frames = librosa.time_to_frames(times, sr=16000, hop_length=320)
            beat_frames = beat_frames[beat_frames < lenn]
            gt = np.zeros((lenn))
            gt[beat_frames] = 1
            # gt_gaussian = sp.ndimage.filters.gaussian_filter(gt, 1, mode='constant')
            if chunk_length_seconds * 16000 > len(audio):
                audio = np.pad(audio, (chunk_length_seconds * 16000 - len(audio), 0), 'constant')
                gt = np.pad(gt, (chunk_length_f-len(gt),0), 'constant')
                times += (chunk_length_seconds - (len(audio)/16000))
            data['name'] = entry.name
            data['dataset'] = dataset
            # data['embeddings'] = model.process_audio(audio)
            # data['data'] = audio
            data['gt'] = gt
            data['times'] = times
            with open(basedir + 'vocal_data/gt/' +dataset+ '#' +entry.name, 'wb') as f:
                pickle.dump(data, f)
                f.close()
            sf.write(basedir + 'vocal_data/audio/'+dataset+ '#' +entry.name + '.wav', audio, 16000)
    #         if dataset+'#'+entry.name in splits['train']:
    #             for i in range(math.ceil(lenn/chunk_length_f)):
    #                 over_sampled_training_list.append(dataset+'#'+entry.name)
    #
    # sh(over_sampled_training_list)
    # splits['train'] = over_sampled_training_list
    # with open(basedir + 'vocal_data/splits', 'wb') as f:
    #     pickle.dump(splits, f)
    #     f.close()


file = open(r'C:\datasets\vocal_data\splits', 'rb')
splits = pickle.load(file)
datasets = ['Ballroom', 'Hainsworth', 'musdb18', 'rock_corpus', 'RWC_pop', 'RWC_royalty_free', 'URSing', 'GTZAN']
for dataset in datasets:
    data_preparer(dataset=dataset, splits=splits)








# model = WAV_LM(r'C:\research\vocal_beat\wavlm\WavLM-Base+.pt')


def data_preparer2(dataset=None ,chunk_length_seconds=8,splits=None):
    basedir = 'C:\datasets/'
    audio_dir = basedir + dataset + '/vocal/'
    annot_dir = basedir + dataset + '/vocal_annotations/'
    chunk_length_f = chunk_length_seconds * 50  # sequence lengths seconds * fps = (frames)
    chunk_length_s = chunk_length_seconds * 16000  # sequence lengths (samples)
    ground_truth = {}
    counter = 0
    for entry in os.scandir(annot_dir):
        if os.path.isfile(audio_dir + entry.name + '.wav'):
            counter += 1
            print('counter')
            audio, _ = librosa.load(audio_dir + entry.name + '.wav', sr=16000)
            ##  preparing ground truth frames
            lenn = int(len(audio) * 50 / 16000)-1   # len audio (frames)
            file = open(annot_dir + entry.name, 'rb')
            times = pickle.load(file)
            beat_frames = librosa.time_to_frames(times, sr=16000, hop_length=320)
            beat_frames = beat_frames[beat_frames < lenn]
            gt = np.zeros((lenn))
            gt[beat_frames] = 1
            # gt_gaussian = sp.ndimage.filters.gaussian_filter(gt, 1, mode='constant')
            ground_truth['name'] = entry.name
            ground_truth['gt'] = gt
            ground_truth['times'] = times
            if entry.name in splits['val']:
                with open(basedir + 'vocal_data/val/gt/' + entry.name, 'wb') as f:
                    pickle.dump(ground_truth, f)
                    f.close()
                sf.write(basedir + 'vocal_data/val/data/' + entry.name + '.wav', audio, 16000)
            elif entry.name in splits['test']:
                with open(basedir + 'vocal_data/test/gt/' + entry.name, 'wb') as f:
                    pickle.dump(ground_truth, f)
                    f.close()
                sf.write(basedir + 'vocal_data/test/data/' + entry.name + '.wav', audio, 16000)
            else:
                ##  preparing audio chunks for training
                for i in range(math.ceil(lenn/chunk_length_f)):
                    ground_truth['name'] = entry.name + '#' + str(i)
                    ground_truth['times'] = []  # for training bet times are useless
                    if i*chunk_length_s*2 > len(audio):
                        if i*chunk_length_s*2 - len(audio) > chunk_length_seconds/2*16000:  # if the length of the last chunk is more than half of the full chunk, pad the beganing and keep it
                            # savind audio chunk
                            chunk = np.pad(audio[i*chunk_length_s:], (i*chunk_length_s*2 - len(audio), 0), 'constant')
                            sf.write(basedir + 'vocal_data/train/data/'+entry.name + '#' + str(i) + '.wav', chunk, 16000)
                            # saving ground truth chunk
                            ground_truth['gt'] = np.pad(gt[i*chunk_length_f:], (chunk_length_f - len(gt[i*chunk_length_f:]), 0), 'constant')
                            with open(basedir + 'vocal_data/train/gt/'+ entry.name + '#' + str(i), 'wb') as f:
                                pickle.dump(ground_truth, f)
                                f.close()
                        break
                    else:
                        chunk = audio[i*chunk_length_s:i*chunk_length_s+chunk_length_s]
                        sf.write(basedir + 'vocal_data/train/data/'+entry.name + '#' + str(i) + '.wav', chunk, 16000)
                        ground_truth['gt'] = gt[i*chunk_length_f:i*chunk_length_f+chunk_length_f]
                        with open(basedir + 'vocal_data/train/gt/' + entry.name + '#' + str(i), 'wb') as f:
                            pickle.dump(ground_truth, f)
                            f.close()










print('done!')
########################feature extractor in case features are extracted offline
# counter = 0
# lens={}
# for entry in os.scandir(annot_dir):
#     if os.path.isfile(audio_dir+entry.name+'.wav'):
#         counter += 1
#         out = {}
#         audio, _ = librosa.load(audio_dir+entry.name+'.wav', sr=16000)
#         # if len(audio) < 128000:
#         #     continue
#         embeddings = model.process_audio(audio)
#         file = open(annot_dir+entry.name, 'rb')
#         beats = pickle.load(file)
#         beat_frames = librosa.time_to_frames(beats, sr=16000, hop_length=320)
#         beat_frames = beat_frames[beat_frames < embeddings.shape[1]]
#         gt = np.zeros((1, embeddings.shape[1]))
#         gt[0, beat_frames] = 1
#         # gt_gaussian = sp.ndimage.filters.gaussian_filter(gt, 1, mode='constant')
#         out['embeddings'] = embeddings
#         out['gt'] = gt
#         out['times'] = beats
#         lens[entry.name]= embeddings.shape[1]
#         with open(basedir + 'vocal_data/' + dataset +'/embedings/'+ entry.name, 'wb') as f:
#             pickle.dump(out, f)
#             f.close()
#         if counter % 5==0 or counter >100:
#             with open(basedir + 'vocal_data/' + dataset + '/lists/lens', 'wb') as f:
#                 pickle.dump(lens, f)
#                 f.close()
#







# class data_loader(torch.utils.data.Dataset):
#     def __init__(self, flist: List[str], sample_rate: int):
#         super().__init__()
#         self.flist = flist
#         self.sample_rate = sample_rate
#
#     def __getitem__(self, index):
#         speed = 0.5 + 1.5 * random.randn()
#         effects = [
#         ['gain', '-n', '-10'],  # apply 10 db attenuation
#         ['remix', '-'],  # merge all the channels
#         ['speed', f'{speed:.5f}'],  # duration is now 0.5 ~ 2.0 seconds.
#         ['rate', f'{self.sample_rate}'],
#         ['pad', '0', '1.5'],  # add 1.5 seconds silence at the end
#         ['trim', '0', '2'],  # get the first 2 seconds
#         ]
#         waveform, _ = torchaudio.sox_effects.apply_effects_file(self.flist[index], effects)
#         return waveform
#
#     def __len__(self):
#         return len(self.flist)
#
# file_list=2
# dataset = data_loader(file_list, sample_rate=8000)
#
# class augment():
#     def init(self,):
#         transforms = [
#             # RandomResizedCrop(n_samples=num_samples),
#             RandomApply([PolarityInversion()], p=0.5),
#             RandomApply([Noise(min_snr=0.001, max_snr=0.005)], p=0.3),
#             RandomApply([Gain()], p=0.2),
#             HighLowPass(sample_rate=16000),  # this augmentation will always be applied in this aumgentation chain!
#             RandomApply([Delay(sample_rate=16000)], p=0.5),
#             RandomApply([PitchShift(n_samples=num_samples, sample_rate=16000)], p=0.4),
#             RandomApply([Reverb(sample_rate=16000)], p=0.3)]
#         self.transforms = Compose(transforms=transforms)
#
#     def process(self, audio):
#         return self.transforms(audio)

# audio, sr = torchaudio.load("C:\datasets\GTZAN/audio/blues/blues.00001.wav",)


##########
# new_splits_train = []
# new_splits_val = []
# new_splits_test = []
# for i in range(len(splits['train'])):
#     with open(data_dir + "/gt/" + splits['train'][i], 'rb') as f:
#         gt = pickle.load(f)
#     if len(gt['times']) > 2:
#         new_splits_train.append(splits['train'][i])
# for i in range(len(splits['val'])):
#     with open(data_dir + "/gt/" + splits['val'][i], 'rb') as f:
#         gt = pickle.load(f)
#     if len(gt['times']) > 2:
#         new_splits_val.append(splits['val'][i])
# for i in range(len(splits['test'])):
#     with open(data_dir + "/gt/" + splits['test'][i], 'rb') as f:
#         gt = pickle.load(f)
#     if len(gt['times']) > 2:
#         new_splits_test.append(splits['test'][i])

#########



# params = list(net.parameters())
# params.extend(list(loss.parameters()))
# opt = torch.optim.Adam(params,lr=1e-3,weight_decay=5e-4)