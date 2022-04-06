import numpy as np
import librosa
import os
import pandas as pd
from cqt import CQT
import pickle
from collections import defaultdict
from data_handler import DATA_EXTRACTOR
import fnmatch



class BEATLES(DATA_EXTRACTOR):  # In order to user get_names it uses data handler as parent
    def __init__(self, base_dir=None, splits=None, data_proc=[CQT()]):
        self.name = "BEATLES"
        self.data_proc = data_proc  # what feature to extract
        self.sample_rate = data_proc[0].get_sample_rate()  # sample rate
        self.hop_length = data_proc[0].get_hop_length()  # hop
        if base_dir is None:
            base_dir = os.path.join("D:/datasets/beatles")
        else:
            base_dir = os.path.join(base_dir, "beatles")
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
                # if c<164:
                #     continue
                data = {}
                feats = []
                genre = spl
                anot_name = ""
                track_name = track.split('#')[2]
                if track_name == '29 Revolution 9':
                    continue
                for file in os.listdir(os.path.join(self.base_dir, 'annotations')):
                    if fnmatch.fnmatch(file, '*'+track_name.replace(" - ", "_").replace(" ", "_").replace("'", "").replace("!", "").replace(".", "").replace(",", "")+".beats"):
                        anot_name = file
                        break
                if anot_name=="":
                        raise ValueError("the file named %s annotation is not available" %track_name)
                wav_path = os.path.join(self.base_dir, 'audio', genre, f'{track_name}.flac')
                label_path = os.path.join(self.base_dir, 'annotations', f'{anot_name}')
                wav, _ = librosa.load(wav_path, sr=self.sample_rate)  # reading file
                for i in range(len(data_proc)):
                    if self.data_proc[i].sample_rate != self.sample_rate:
                        wav, _ = librosa.load(wav_path, sr=self.data_proc[i].sample_rate)  # reading file
                    feats.append(self.data_proc[i].process_audio(wav).T)  # extracting features
                feats = [feats[i][:len(min(feats, key=len))] for i in
                         range(len(feats))]  # makes all feature lengths (frame lengths) the same
                feats = np.hstack(feats).T
                labels = pd.read_csv(label_path, header=None)  # extracting annotations
                labels = labels.values.flatten()
                beats = []
                downs = []
                for i in range(len(labels)):
                    if int(labels[i].split("\t")[1]) == 1:
                        downs.append(float(labels[i].split("\t")[0]))
                    else:
                        beats.append(float(labels[i].split("\t")[0]))
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
        with open(self.base_dir + "/tracks_list", 'wb') as fa:  # writes tracks list
            pickle.dump(tracks_list, fa)


    def get_tracks(self, split):
        tracks = []
        split_path = os.path.join(self.base_dir, 'audio', split)
        for track in os.listdir(split_path):
            if track.endswith(".flac"):
                tracks.append("BEATLES#" + split + "#" + os.path.splitext(track)[0])
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

        splits = ["A Hard Day's Night", 'Abbey Road', 'Beatles For Sale', 'Help!',
                  'Let It Be', 'Magical Mystery Tour', 'Please Please Me', 'Revolver', 'Rubber Soul', 'Sgt Pepper',
                  'The Beatles', 'With The Beatles', ]  # 'Yellow Submarine' : is not labeled

        return splits

    def feature_names(self):  # gives feature_names
        feature_names = ""
        for i in range(len(self.data_proc)):
            feature_names = feature_names + self.data_proc[0].features_name() + "+"

        return feature_names[:-1]

    @staticmethod
    def name():
        return "BEATLES"
