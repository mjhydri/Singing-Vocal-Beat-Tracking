import librosa
import numpy as np
from madmom.audio import SpectrogramDifferenceProcessor

from common import FeatureModule


class MEL(FeatureModule):
    def __init__(self, sample_rate=22050, win_length=441, mel_n_fft=1024, hop_size=441, n_mels=128, fmin=0.0,
                 fmax=None,
                 diffs=True):
        self.sample_rate = sample_rate
        self.win_length = win_length
        self.hop_length = hop_size
        self.n_mels = n_mels
        self.mel_n_fft = mel_n_fft
        self.fmin = fmin
        self.fmax = fmax
        self.diffs = diffs

    def process_audio(self, audio):
        S = librosa.feature.melspectrogram(y=audio, sr=self.sample_rate, S=None, n_fft=self.mel_n_fft, hop_length=self.hop_length,
                                           win_length=self.win_length, window='hann', center=True, pad_mode='reflect', power=2.0,)

        # S = np.log(0000.1*S+1) / np.log(10000)
        S= np.log10(S+1)
        if self.fmin:
            bins = round(self.fmin * self.n_fft / self.sample_rate)
            S = S[bins:]
        if self.fmax:
            bins = round(self.fmax * self.n_fft / self.sample_rate)
            S = S[:bins]
        if self.diffs:
            S1 = S*S#*S*S
            S1 = np.column_stack((S1[:,0],np.diff(S1, axis=1)))
            S1 = np.diff(S1, axis=0)
            S1[S1 < 0] = 0
            S = np.vstack((S, S1))
        return S


# import librosa

# import matplotlib.pyplot as plt
# import librosa.display
#
# # y1, sr1 = librosa.load(librosa.ex('trumpet'),sr=500)
# # mel = MEL(sample_rate=sr1, win_length=32, mel_n_fft=2048, hop_size=10, n_mels=128, fmin=0.0, fmax=None,
# #                  diffs=True)
# # #
# y1, sr1 = librosa.load(librosa.ex('trumpet'), sr=22050)
# mel = MEL(sample_rate=sr1, win_length=1024, mel_n_fft=2048, hop_size=441, n_mels=128, fmin=0.0, fmax=None,
#           diffs=True)
# # #
# S = mel.process_audio(y1)
# fig, ax = plt.subplots()
# img = librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='hz', x_axis='time', ax=ax)
# ax.set_title('Power spectrogram')
# fig.colorbar(img, ax=ax, format="%+2.0f dB")
# plt.show()
