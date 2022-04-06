import numpy as np
import librosa
import os
import pandas as pd
from cqt import CQT
import pickle
from collections import defaultdict
from data_handler import DATA_EXTRACTOR
import fnmatch

class ROCK_CORPUS(DATA_EXTRACTOR):  # In order to user get_names it uses data handler as parent
    def __init__(self, base_dir=None, splits=None, data_proc=[CQT()]):
        self.name = "ROCK_CORPUS"
        self.data_proc = data_proc  # what feature to extract
        self.sample_rate = data_proc[0].get_sample_rate()  # sample rate
        self.hop_length = data_proc[0].get_hop_length()  # hop
        if base_dir is None:
            base_dir = os.path.join("D:/datasets/rock_corpus")
        else:
            base_dir = os.path.join(base_dir, "rock_corpus")
        self.base_dir = base_dir
        self.tracks = defaultdict(list)
        if splits is None:
            splits = self.available_splits()
        self.splits = splits
        for spl in self.splits:
            self.tracks[spl].append(self.get_tracks(spl))
        tracks_list = defaultdict(list)
        # c=-1
        for spl in splits:
            for track in self.tracks[spl][0]:  # covers all tracks in splits:
                # c+=1
                # if c<12:
                #     continue
                data = {}
                feats = []
                beats =[]
                downs = []
                genre = spl
                track_name = track.split('#')[2]
                wav_path = os.path.join(self.base_dir, 'audio', genre, f'{track_name}.wav')
                label_path = os.path.join(self.base_dir, 'annotations', 'timing_data', f'{track_name}.tim')
                # self.generate_time_signatures(track_name)
                signature_path = os.path.join(self.base_dir, 'annotations', 'signature_data', f'{track_name}')
                # print(track_name)
                labels = pd.read_csv(label_path, header=None)  # extracting annotations
                with open(signature_path, 'rb') as fp:
                    signatures = pickle.load(fp)
                if len(signatures) < len(labels)+1:
                    signatures.extend([signatures[-1]] * (len(labels)+1 - len(signatures)))
                for i , L in enumerate(labels[0]):
                    down = float(L.split("\t")[0])
                    downs.append(down)
                    if signatures[i] == '1208':
                        signatures[i] = '608'
                        if L.split("\t")[0] != labels[0].iloc[-1].split("\t")[0]:   # to add the middle of two adjacent downbeats as a new downbeat
                            downs.append((down + float(labels[0][i+1].split("\t")[0]))/2)
                        else:
                            downs.append(down + (down - float(labels[0][i - 1].split("\t")[0])) / 2)
                    if L.split("\t")[0] != labels[0].iloc[-1].split("\t")[0]:    # to generate beat positions based on downbeats and time signature
                        beat_step = (float(labels[0][i + 1].split("\t")[0]) - down) / int(signatures[i].split("0")[0])
                    else:
                        beat_step = (down-float(labels[0][i - 1].split("\t")[0])) / int(signatures[i].split("0")[0])-1
                    for j in range((int(signatures[i].split("0")[0])-1)):
                        beats.append(down+beat_step*(j+1))
                wav, _ = librosa.load(wav_path, sr=self.sample_rate)  # reading file
                for i in range(len(data_proc)):
                    if self.data_proc[i].sample_rate != self.sample_rate:
                        wav, _ = librosa.load(wav_path, sr=self.data_proc[i].sample_rate)  # reading file
                    feats.append(self.data_proc[i].process_audio(wav).T)  # extracting features
                feats = [feats[i][:len(min(feats, key=len))] for i in
                         range(len(feats))]  # makes all feature lengths (frame lengths) the same
                feats = np.hstack(feats).T
                downs = np.asarray(downs)
                beats = np.asarray(beats)
                frame_idcs = np.arange(feats.shape[-1])
                # Obtain the time of the first sample of each frame
                times = librosa.frames_to_time(frames=frame_idcs,
                                               sr=self.sample_rate,
                                               hop_length=self.hop_length)

                gt = np.zeros((3, feats.shape[-1]))

                beat_frames = librosa.time_to_frames(beats, sr=self.sample_rate, hop_length=self.hop_length)
                down_frames = librosa.time_to_frames(downs, sr=self.sample_rate, hop_length=self.hop_length)

                # TODO - better way to account for GT beats which go beyond audio
                beat_frames = beat_frames[beat_frames < feats.shape[-1]]
                down_frames = down_frames[down_frames < feats.shape[-1]]

                gt[0, beat_frames] = 1
                gt[1, down_frames] = 1
                gt[0, down_frames] = 0

                gt[2, np.sum(gt, axis=0) == 0] = 1

                assert np.sum(gt) == feats.shape[-1]
                if sum(gt[0]) > 4 and sum(gt[1]) > 4:  # to check if there are labels
                    tracks_list[spl].append(track)
                else:
                    continue
                # if self.seq_len is not None:
                #     frame_start = self.rng.randint(0, feats.shape[-1] - self.seq_len)
                #     frame_end = frame_start + self.seq_len
                #     feats = feats[..., frame_start: frame_end]
                #     times = times[frame_start: frame_end]
                #     gt = gt[..., frame_start: frame_end]
                # else:
                #     # sample_start = frame_start * self.hop_length
                # sample_end = frame_end * self.hop_length
                # Determine the time in seconds of the boundary samples
                # sec_start = sample_start / self.sample_rate
                # sec_stop = sample_end / self.sample_rate

                # beats = beats[beats > sec_start]
                # beats = beats[beats < sec_stop]

                # downs = downs[downs > sec_start]
                # downs = downs[downs < sec_stop]
                self.feature_names = self.get_names()
                # data['beats'] = beats
                # data['downs'] = downs

                data['feats'] = feats
                data['times'] = times
                data['ground_truth'] = gt
                # data['features_setting'] = data_proc
                # data["feature_names"] = self.feature_names

                self.dir = self.base_dir + "/extracted/" + self.feature_names + "/sample_rate=" + str(
                    data_proc[0].sample_rate) + "-hop=" + str(data_proc[0].get_hop_length()) + "/"
                if not os.path.exists(self.dir):
                    os.makedirs(self.dir)
                with open(self.dir + track, 'wb') as f:
                    pickle.dump(data, f)
        # with open(self.base_dir + "/tracks_list", 'wb') as fa:  # writes tracks list
        #     pickle.dump(tracks_list, fa)


    def get_tracks(self, split):
        tracks = []
        split_path = os.path.join(self.base_dir, 'audio', split)
        for track in os.listdir(split_path):
            if track.endswith(".wav"):
                tracks.append("ROCK_CORPUS#" + split + "#" + os.path.splitext(track)[0])
        # tracks = [os.path.splitext(track)[0] for track in tracks]

        tracks = sorted(tracks)

        return tracks

    @staticmethod
    def available_splits():
        """
        Obtain a list of possible splits.

        Returns
        ----------
        splits : list of strings
          Player codes listed at beginning of file names
        """

        splits = ["all"]

        return splits

    def feature_names(self):  # gives feature_names
        feature_names = ""
        for i in range(len(self.data_proc)):
            feature_names = feature_names + self.data_proc[0].features_name() + "+"

        return feature_names[:-1]

    @staticmethod
    def name():
        return "ROCK_CORPUS"

    def generate_time_signatures(self, track_name):
        meters_file_path = os.path.join(self.base_dir, 'annotations', "tdc_tsigs_all.txt")
        aaa = open(meters_file_path, 'r')
        lines = aaa.readlines()
        track_times = []
        for i, L in enumerate(lines):
            if fnmatch.fnmatch(L, '*' + track_name.replace(" ", "_").replace("'", "").replace("!", "").replace(".",
                 "").replace(",", "") + "_tdc.txt\n"):
                while '%' not in lines[i + 1]:
                    track_times.append(lines[i + 1].split(' ')[1].replace("\n", ""))
                    i += 1
                break
        if not track_times:
            raise ValueError("the file named %s annotation is not available" % track_name)


        with open(os.path.join(self.base_dir, 'annotations', 'signature_data', track_name), 'wb') as fp:
            pickle.dump(track_times, fp)



