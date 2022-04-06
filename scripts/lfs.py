from librosa import stft
import numpy as np

from common import FeatureModule


class LFS(FeatureModule):
    def __init__(self, sample_rate=500, win_length=32, hop_size=10, n_fft=2048, bins_to_keep=None, fmin=None,
                 fmax=200, decile_bypass=[]):
        self.sample_rate = sample_rate
        self.win_length = win_length
        self.hop_length = hop_size
        self.n_fft = n_fft
        self.bins_to_keep = bins_to_keep
        self.fmin = fmin
        self.fmax = fmax
        self.decile_bypass = decile_bypass

    def process_audio(self, audio):
        S = np.abs(stft(y=audio, win_length=self.win_length, hop_length=self.hop_length, n_fft=self.n_fft))
        if self.fmin:
            bins = round(self.fmin * self.n_fft / self.sample_rate)
            S = S[bins:]
        if self.fmax:
            bins = round(self.fmax * self.n_fft / self.sample_rate)
            S = S[:bins]
        if self.bins_to_keep:
            S = S[:self.bins_to_keep]
        if self.decile_bypass:
            S[S > (self.decile_bypass/10)*np.max(S)] = np.max(S)
        return S


# import librosa

# import matplotlib.pyplot as plt
# import librosa.display
# y1, sr1 = librosa.load(librosa.ex('trumpet'),sr=500)
# lfs = LFS(fmax=150, sample_rate=sr1, win_length=32, hop_size=10, n_fft=32)
# # #
# # # # y1, sr1 = librosa.load(librosa.ex('trumpet'),sr=22050)
# # # # lfs = LFS(fmax=150, sample_rate=sr1, win_length=1411, hop_size=441, n_fft=2048)
# # #
# S = lfs.process_audio(y1)
# fig, ax = plt.subplots()
# img = librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='hz', x_axis='time', ax=ax)
# ax.set_title('Power spectrogram')
# fig.colorbar(img, ax=ax, format="%+2.0f dB")
# plt.show()