from librosa import cqt, convert
import numpy as np

from common import FeatureModule


class MCQT(FeatureModule):
    def __init__(self, sample_rate=22050, hop_size=441, bins_per_octave=96, n_bins=558, fmin=196,
                 average=True, keep_range=[], log=True, quartile=[]):
        self.sample_rate = sample_rate
        self.sample_rate = sample_rate
        self.hop_length = hop_size
        self.bins_per_octave = bins_per_octave
        self.n_bins = n_bins
        self.keep_range = keep_range
        self.fmin = fmin
        self.average = average
        self.log = log
        self.quartile = quartile  # zero all quartles below quartile_th section

    def process_audio(self, audio):
        S = np.abs(cqt(audio, sr=self.sample_rate, hop_length=self.hop_length, fmin=self.fmin, n_bins=self.n_bins,
                               bins_per_octave=self.bins_per_octave, tuning=0.0,
                               filter_scale=1, norm=1, sparsity=0.01, window='hann', scale=True, pad_mode='reflect',
                               res_type=None, dtype=None))
        if self.average:
            for index in range(len(S)):
                octaves = np.asarray([index + 96, index + 192, index + 288, index + 384, index + 480])
                octaves = octaves[octaves < 558]
                octaves = np.sum(S[octaves], axis=0)
                octaves = [S[index], octaves]
                S[index] = np.sum(octaves, axis=0)
        if self.keep_range:
            table = convert.cqt_frequencies(self.n_bins, fmin=self.fmin, bins_per_octave=self.bins_per_octave,
                                                    tuning=0.0)
            lower = min((np.argwhere(table >= self.keep_range[0])))[0]
            higher = max((np.argwhere(table <= self.keep_range[1])))[0]
            S = S[lower:higher + 1]
        if self.log:
            S = np.log(S+1)
        if self.quartile:
            S[S < (np.max(S) * self.quartile / 10)] = 0

        return S


# bbb=librosa.convert.cqt_frequencies(559, fmin=196, bins_per_octave=96, tuning=0.0)# number of bins to cover until 22025 (nyquist frequency)

# import numpy as np
# import matplotlib.pyplot as plt
# import librosa.display
#
# y1, sr1 = librosa.load(librosa.ex('trumpet'), sr=22050)
# mcqt = MCQT(keep_range=[392, 3520], quartile=3)
# S = mcqt.process_audio(y1)
# fig, ax = plt.subplots()
# img = librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='hz', x_axis='time', ax=ax)
# ax.set_title('Power spectrogram')
# fig.colorbar(img, ax=ax, format="%+2.0f dB")
# plt.show()
# pass
