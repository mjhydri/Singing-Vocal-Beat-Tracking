import jams
import numpy as np
import musdb
import sounddevice as sd
from madmom.features import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
from scipy.io.wavfile import write
from BeatNet.BeatNet import BeatNet
import librosa
import pickle
from mido import MidiFile
import matplotlib.pyplot as plt
import os
import playsound
from madmom.features.beats import detect_beats, RNNBeatProcessor, DBNBeatTrackingProcessor
import pandas as pd
import fnmatch
import scipy.io

dataset = 'ballroom'

def rms(signal, window=22050 * 5, hop=441):
    rms = []
    hop = hop
    window = window
    for i in range(int(len(signal - window) / hop)):  # calculate RMS for each 0.5 seconds hope
        rms.append(np.sqrt(np.mean(signal[hop * i:hop * i + window] ** 2)))
    return np.array(rms)


def chunks(signal, threshold):
    chunks = []
    start = 0
    end = 0
    start = np.argwhere(signal >= threshold)[0]
    hook = np.argwhere(signal < threshold)[0]


def midi_to_note(name=''):
    directory = rf'C:\datasets\URSing/{name}'
    mid = MidiFile(directory, clip=True)
    source_time = 0
    annotations = []
    for msg in mid.tracks[2]:
        if msg.type == "note_on" or msg.type == "note_off":
            source_time += msg.time
            if msg.type == "note_on" and msg.note <= 70:
                annotations.append([round(source_time / 120, 3), 1])
            if msg.type == "note_on" and msg.note > 70:
                annotations.append([round(source_time / 120, 3), 2])
    annotations = np.array(annotations)
    with open(rf'C:\datasets\URSing/annotations/{name.replace(".mid", "")}', 'wb') as fp:
        pickle.dump(annotations, fp)
        fp.close()
    return annotations


def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)


## creating revision list
# with os.scandir("C:\datasets\musdb18/beat_tracking_derivation/audio_with_annot") as it:
#     list_of_revise = []
#     for entry in it:
#         if entry.name.endswith(".wav") and entry.is_file() and entry.name[0]=='#':
#             print(entry.name, entry.path)
#             list_of_revise.append(entry.name[1:])
# with open(f'C:\datasets\musdb18/beat_tracking_derivation/list_of_revise', 'wb') as fp:
#     pickle.dump(list_of_revise, fp)


## converting midi files into time annotation
# with os.scandir("C:\datasets\musdb18/beat_tracking_derivation") as it:
#    for entry in it:
#        if entry.name.endswith(".mid") and entry.is_file():
#            a = midi_to_note(entry.name)

# with os.scandir(r"C:\datasets\URSing") as it:
#    for entry in it:
#        if entry.name.endswith(".mid") and entry.is_file():
#            a = midi_to_note(entry.name)


# list_of_revise = []
# # Early Label Extractor
# estimator_type = 'beatnet'
# if estimator_type == 'beatnet':
#     estimator = BeatNet(1, mode='offline', inference_model='DBN', plot=[], thread=False)
# elif estimator_type == 'madmom':
#     estimator = DBNDownBeatTrackingProcessor(fps=100, beats_per_bar=[4])
#
# if dataset == 'musdb':
#     mus = musdb.DB(root="C:\datasets\musdb18/")
#     for track in mus:
#         mixture = track.stems[0].astype(np.float32)
#         mixture = np.mean(mixture, axis=1)
#         if track.name + '.wav' in list_of_revise:
#             mixture = track.stems[0].astype(np.float32)
#             mixture = np.mean(mixture, axis=1)
#             write(f'C:\datasets\musdb18/beat_tracking_derivation/audio_to_revise/{track.name}.wav', 44100, mixture)
#         else:
#             if estimator_type == 'beatnet':
#                 Output = estimator(mixture)
#             elif estimator_type == 'madmom':
#                 act = RNNDownBeatProcessor()(mixture)
#                 Output = estimator(act)
#                 with open(f'C:\datasets\musdb18/beat_tracking_derivation/annotations/{track.name}', 'wb') as fp:
#                     pickle.dump(Output, fp)
#                     fp.close()

# if dataset == 'ursing':
#     counter = -1
#     dir = 'C:\datasets/ursing'
#     for entry in os.scandir(dir + '/data/'):
#         if entry.is_dir():
#             counter += 1
#             # if counter < 44:
#             #     continue
#             dir2 = dir + f"/data/{entry.name}/"
#             for entry2 in os.scandir(dir2):
#                 if entry2.name == 'Mix.wav':
#                     mixture, _ = librosa.load(dir + f'/data/{entry.name}/Mix.wav', sr=22050)
#                     if estimator_type == 'beatnet':
#                         Output = estimator.process(mixture)
#                     elif estimator_type == 'madmom':
#                         act = RNNDownBeatProcessor()(mixture)
#                         Output = estimator(act)
#                     with open(dir + f'/annotations/{entry.name}', 'wb') as fp:
#                         pickle.dump(Output, fp)
#                         fp.close()

##adding sound
# if dataset == 'musdb':
#    mus = musdb.DB(root="C:\datasets\musdb18/")
# with open(f'C:\datasets\musdb18/beat_tracking_derivation/list_of_revise', 'rb') as jj:
#    list_of_revise = pickle.load(jj)
# for track in mus:
#   if track.name+'.wav' in list_of_revise:
#       mixture = track.stems[0].astype(np.float32)
#       mixture = np.mean(mixture, axis=1)
#       mixture = librosa.resample(mixture, 44100, 22050)
#       file = open("C:\datasets\musdb18/beat_tracking_derivation/annotations/"+track.name, 'rb')
#       annot = pickle.load(file)
#       file.close()
#       a = librosa.clicks(times=annot[:, 0], frames=None, sr=22050, hop_length=441, click_freq=480.0,
#                                       click_duration=0.1, click=None, length=len(mixture))
#       #                # b = librosa.clicks(times=Output[Output[:, 1] != 1], frames=None, sr=22050, hop_length=441,
#       #                                        # click_freq=960.0, click_duration=0.1, click=None, length=len(mixture))
#       mixture = mixture + a
#       write(f'C:\datasets\musdb18/beat_tracking_derivation/audio_with_annot/{track.name}.wav', 22050, mixture)
#       continue






# if dataset == 'ursing':
#     counter = -1
#     dir = 'C:\datasets/ursing'
#     for entry in os.scandir(dir + '/data/'):
#         if entry.is_dir():
#             counter += 1
#             # if counter < 44:
#             #     continue
#             dir2 = dir + f"/data/{entry.name}/"
#             for entry2 in os.scandir(dir2):
#                 if entry2.name == 'Mix.wav':
#                     mixture, _ = librosa.load(dir + f'/data/{entry.name}/Mix.wav', sr=22050)
#                     file = open(rf"C:\datasets\urSing/annotations/{entry.name}", 'rb')
#                     annot = pickle.load(file)
#                     file.close()
#                     a = librosa.clicks(times=annot[:, 0], frames=None, sr=22050, hop_length=441, click_freq=480.0,
#                                        click_duration=0.1, click=None, length=len(mixture))
#                     #                # b = librosa.clicks(times=Output[Output[:, 1] != 1], frames=None, sr=22050, hop_length=441,
#                     #                                        # click_freq=960.0, click_duration=0.1, click=None, length=len(mixture))
#                     mixture = mixture + a
#                     write(rf'C:\datasets\URSing/audio_with_annot/{entry.name}.wav', 22050,
#                           mixture)


def segment(vocal, annot, name, dir):
    vocal_rms = rms(vocal)
    vocal_boundaries = np.where(np.diff(np.sign(vocal_rms - 0.01)))[0]
    if vocal_rms[0] > 0.01:
        vocal_boundaries = np.append(0, vocal_boundaries)
    if len(vocal_boundaries) % 2 != 0:
        vocal_boundaries = np.append(vocal_boundaries, len(vocal_rms))
    vocal_diffs = np.diff(vocal_boundaries)
    to_be_removed = np.where(vocal_diffs < 400)[0]
    to_be_removed = np.append(to_be_removed, to_be_removed + 1)
    to_be_removed = consecutive(np.unique(
        to_be_removed))  # to keep the last threshold crossing if the number of consecutive to be removed is odd
    for i in range(len(to_be_removed)):
        if len(to_be_removed[i]) % 2 != 0:
            to_be_removed[i] = np.delete(to_be_removed[i], -1)
    to_be_removed = np.concatenate(to_be_removed).flat
    vocal_boundaries = np.delete(vocal_boundaries, to_be_removed, None)
    # if vocal_rms[0] > 0.005 and 0 not in vocal_boundaries:
    #     vocal_boundaries = np.append(0, vocal_boundaries)
    if len(vocal_boundaries) % 2 != 0:
        vocal_boundaries = np.append(vocal_boundaries, len(vocal_rms))
    vocal_boundaries *= 441
    for i in range(len(vocal_boundaries) // 2):
        vocal_start = vocal_boundaries[i * 2]
        vocal_stop = vocal_boundaries[i * 2 + 1]
        chunk = vocal[vocal_start:vocal_stop]
        # revsing the start and end of each chunk
        RMS = rms(chunk, window=2205)
        # if len(RMS) == 0:
        #     continue
        if (RMS > 0.01).any():
            vocal_start = vocal_start + np.where(RMS > 0.01)[0][0] * 441
        if vocal_stop + 22050 < len(vocal):
            vocal_stop += 22050
        chunk = vocal[vocal_start:vocal_stop]
        write(dir + f'/vocal/{name}_{i}.wav', 22050, chunk)
        label_start = vocal_start / 22050
        label_end = vocal_stop / 22050
        label = annot[label_start < annot]
        label = label[label < label_end]
        label -= vocal_start / 22050
        a = librosa.clicks(times=label, frames=None, sr=22050, hop_length=441, click_freq=480.0,
                           click_duration=0.1, click=None, length=len(chunk))
        chunk += a
        with open(dir + f'/vocal_annotations/{name}_{i}', 'wb') as fp:
            pickle.dump(label, fp)
            fp.close()
        write(dir + f'/vocal_with_annot/{name}_{i}.wav', 22050, chunk)


## Vocal Sound and Label Segmentation
if dataset == 'musdb':
    dir = 'C:\datasets\musdb18/beat_tracking_derivation'
    mus = musdb.DB(root=dir)
    counter = -1
    with open(dir + '/list_of_used', 'rb') as fp:
        list_of_used = pickle.load(fp)
    for track in mus:
        if track.name in list_of_used:
            counter += 1
            # if counter < 9:
            #    continue
            file = open(dir + "/annotations/" + track.name, 'rb')
            annot = pickle.load(file)
            file.close()
            annot = annot[:, 0]
            vocal = track.stems[4].astype(np.float32)
            vocal = np.mean(vocal, axis=1)
            vocal = librosa.resample(vocal, 44100, 22050)
            segment(vocal, annot, track.name, dir)

if dataset == 'rock_corpus':
    counter = -1
    dir = 'C:\datasets\ROCK_CORPUS'
    for entry in os.scandir(dir + "/audio\separated\mdx_extra_q/"):
        if entry.is_dir():
            dir2 = dir + f"/audio\separated\mdx_extra_q/{entry.name}/"
            for entry2 in os.scandir(dir2):
                if entry2.name == 'vocals.mp3':
                    counter += 1
                    if counter < 39:
                        break
                    vocal, _ = librosa.load(dir2 + entry2.name, sr=22050)
                    beats = []
                    downs = []
                    label_path = os.path.join(dir, 'annotations', 'timing_data', f'{entry.name}.tim')
                    # self.generate_time_signatures(track_name)
                    signature_path = os.path.join(dir, 'annotations', 'signature_data', f'{entry.name}')
                    # print(track_name)
                    labels = pd.read_csv(label_path, header=None)  # extracting annotations
                    with open(signature_path, 'rb') as fp:
                        signatures = pickle.load(fp)
                    if len(signatures) < len(labels) + 1:
                        signatures.extend([signatures[-1]] * (len(labels) + 1 - len(signatures)))
                    for i, L in enumerate(labels[0]):
                        down = float(L.split("\t")[0])
                        downs.append(down)
                        if signatures[i] == '1208':
                            signatures[i] = '608'
                            if L.split("\t")[0] != labels[0].iloc[-1].split("\t")[
                                0]:  # to add the middle of two adjacent downbeats as a new downbeat
                                downs.append((down + float(labels[0][i + 1].split("\t")[0])) / 2)
                            else:
                                downs.append(down + (down - float(labels[0][i - 1].split("\t")[0])) / 2)
                        if L.split("\t")[0] != labels[0].iloc[-1].split("\t")[
                            0]:  # to generate beat positions based on downbeats and time signature
                            beat_step = (float(labels[0][i + 1].split("\t")[0]) - down) / int(
                                signatures[i].split("0")[0])
                        else:
                            beat_step = (down - float(labels[0][i - 1].split("\t")[0])) / int(
                                signatures[i].split("0")[0]) - 1
                        for j in range((int(signatures[i].split("0")[0]) - 1)):
                            beats.append(down + beat_step * (j + 1))
                    beats = beats + downs
                    segment(vocal, np.array(beats), entry.name, dir)

if dataset == 'beatles':
    counter = -1
    dir = 'C:\datasets/beatles'
    for entry in os.scandir(dir + "/audio\separated\mdx_extra_q/"):
        if entry.is_dir():
            dir2 = dir + f"/audio\separated\mdx_extra_q/{entry.name}/"
            for entry2 in os.scandir(dir2):
                if entry2.name == 'vocals.mp3':
                    counter += 1
                    # if counter<158:
                    #     continue
                    for file in os.listdir(os.path.join(dir, 'annotations')):
                        aaa = entry.name.replace(" - ", "_").replace(" ", "_").replace("'", "").replace(
                            "!", "").replace(".", "").replace(",", "") + ".beats"
                        if file[-len(aaa):] == aaa:
                            anot_name = file
                    vocal, _ = librosa.load(dir2 + entry2.name, sr=22050)
                    label_path = os.path.join(dir, 'annotations', anot_name)
                    labels = pd.read_csv(label_path, header=None)  # extracting annotations
                    labels = labels.values.flatten()
                    beats = []
                    downs = []
                    for i in range(len(labels)):
                        if int(labels[i].split("\t")[1]) == 1:
                            downs.append(float(labels[i].split("\t")[0]))
                        else:
                            beats.append(float(labels[i].split("\t")[0]))
                    beats = beats + downs
                    segment(vocal, np.array(beats), entry.name, dir)

if dataset == 'gtzan':
    counter = -1
    dir = 'C:\datasets/gtzan'
    for entry in os.scandir(dir + "/audio\separated/"):
        if entry.is_dir():
            counter += 1
            # if counter < 44:
            #     continue
            dir2 = dir + f"/audio\separated\/{entry.name}/"
            for entry2 in os.scandir(dir2):
                if entry2.name == 'vocals.mp3':
                    vocal, _ = librosa.load(dir2 + entry2.name, sr=22050)
                    jam_path = os.path.join(dir, 'annotations', 'beat_labels', 'jams', f'{entry.name}.wav.jams')
                    jams_data = jams.load(jam_path).annotations  # extracting annotations
                    beat_data, down_data = jams_data[0], jams_data[1]
                    assert beat_data.sandbox.annotation_type == 'beat'
                    assert down_data.sandbox.annotation_type == 'downbeat'
                    beats = [beat.time for beat in beat_data.data]
                    downs = [beat.time for beat in down_data.data]
                    beats = beats + downs
                    segment(vocal, np.array(beats), entry.name, dir)

if dataset == 'hainsworth':
    counter = -1
    dir = 'C:\datasets/hainsworth'
    for entry in os.scandir(dir + "/separated/"):
        if entry.is_dir():
            counter += 1
            # if counter < 44:
            #     continue
            dir2 = dir + f"\separated\/{entry.name}/"
            for entry2 in os.scandir(dir2):
                if entry2.name == 'vocals.mp3':
                    vocal, _ = librosa.load(dir2 + entry2.name, sr=22050)
                    mat = scipy.io.loadmat(dir + f'/mats/{entry.name}_info.mat')
                    beats = np.concatenate(np.array(mat['datarec'][10][0])).flat
                    segment(vocal, np.array(beats) / 44100, entry.name, dir)

if dataset == 'rwc_pop':
    counter = -1
    dir = 'C:\datasets/rwc'
    for entry in os.scandir(dir + "/rwc pop/separated/"):
        if entry.is_dir():
            counter += 1
            # if counter < 44:
            #     continue
            dir2 = dir + f"/rwc pop\separated\{entry.name}/"
            for entry2 in os.scandir(dir2):
                if entry2.name == 'vocals.mp3':
                    vocal, _ = librosa.load(dir2 + entry2.name, sr=22050)
                    file1 = open(
                        dir + f'/Annotations\AIST.RWC-MDB-P-2001.BEAT/AIST.RWC-MDB-P-2001.BEAT/RM-P{entry.name[5:]}.BEAT.txt',
                        'r')
                    beats = []
                    Lines = file1.readlines()
                    for i in range(len(Lines)):
                        beats.append(int(Lines[i].split('\t')[1]) / 100)
                    segment(vocal, np.array(beats), entry.name, dir)

if dataset == 'rwc_royalty_free':
    counter = -1
    dir = 'C:\datasets/rwc'
    for entry in os.scandir(dir + "/rwc royalty free/separated/"):
        if entry.is_dir():
            counter += 1
            # if counter < 44:
            #     continue
            dir2 = dir + f"/rwc royalty free\separated\{entry.name}/"
            for entry2 in os.scandir(dir2):
                if entry2.name == 'vocals.mp3':
                    vocal, _ = librosa.load(dir2 + entry2.name, sr=22050)
                    file1 = open(
                        dir + f'/Annotations\AIST.RWC-MDB-R-2001.BEAT\AIST.RWC-MDB-R-2001.BEAT/RM-R0{entry.name[5:]}.BEAT.txt',
                        'r')
                    beats = []
                    Lines = file1.readlines()
                    for i in range(len(Lines)):
                        beats.append(int(Lines[i].split('\t')[1]) / 100)
                    segment(vocal, np.array(beats), entry.name, dir)


if dataset == 'ursing':
    counter = -1
    dir = 'C:\datasets/ursing'
    for entry in os.scandir(dir + '/data/'):
        if entry.is_dir():
            counter += 1
            # if counter < 44:
            #     continue
            dir2 = dir + f"/data/{entry.name}/"
            for entry2 in os.scandir(dir2):
                if entry2.name == 'Vocal.wav':
                    mixture, _ = librosa.load(dir + f'/data/{entry.name}/Vocal.wav', sr=22050)
                    file = open(dir + "/annotations/" + entry.name, 'rb')
                    annot = pickle.load(file)
                    file.close()
                    beats = annot[:, 0]


if dataset == 'ballroom':
    counter = -1
    dir = 'C:\datasets/ballroom'
    for entry in os.scandir(dir + "/audio\separated/"):
        if entry.is_dir():
            dir2 = dir + f"/audio\separated/{entry.name}/"
            for entry2 in os.scandir(dir2):
                if entry2.name == 'vocals.mp3':
                    counter += 1
                    # if counter < 39:
                    #     break
                    vocal, _ = librosa.load(dir2 + entry2.name, sr=22050)
                    label_path = os.path.join(dir, 'annotations', f'{entry.name}.beats')
                    labels = pd.read_csv(label_path, header=None)  # extracting annotations
                    labels = labels.values.flatten()
                    beats = []
                    downs = []
                    for i in range(len(labels)):
                        if int(labels[i].split(" ")[1]) == 1:
                            downs.append(float(labels[i].split(" ")[0]))
                        else:
                            beats.append(float(labels[i].split(" ")[0]))
                    beats = beats + downs
                    segment(vocal, np.array(beats), entry.name, dir)



print('hg')

# continue
# sd.play(mixture, 22050)
# playsound.playsound('storm.mp3',True)

# a=mus[0].stems


# write(f'C:\datasets\musdb18/beat_tracking_derivation/audio_with_annot/{track.name}.wav', 44100, vocal)

# mixture = librosa.resample(mixture, 44100, 22050)
