from librosa import stft, onset

from common import FeatureModule


class ODF(FeatureModule):
    def __init__(self, sample_rate=22050, win_length=1411, hop_size=441, channels=[0, 32, 64, 96, 128], feature=None):
        self.sample_rate = sample_rate
        self.win_length = win_length
        self.hop_length = hop_size
        self.channels = channels
        self.feature = feature

    def process_audio(self, audio):
        result = onset.onset_strength_multi(y=audio, sr=self.sample_rate, hop_length=self.hop_length,
                                            channels=self.channels, feature=self.feature)
        return result



# import librosa.display
# import numpy as np
# import matplotlib.pyplot as plt
# y, sr = librosa.load(librosa.util.example_audio_file(), duration=10.0)
# D = np.abs(librosa.stft(y))
# plt.figure()
# plt.subplot(2, 1, 1)
# librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),y_axis = 'log')
# plt.title('Power spectrogram')


# each bin = max frquency/ num_bins = 11025/1025 = 10.756097560975 =>
# boundaries = 0, (150*1025)/11025, (500*1025)/11025, 1025
# onset_subbands = librosa.onset.onset_strength_multi(y=y, channels = [0, 14, 47, 1025], feature=stft)

# onset_subbands = librosa.onset.onset_strength_multi(y=y, sr=sr, channels=[0, 32, 64, 96, 128])
#
# plt.subplot(2, 1, 2)
# librosa.display.specshow(onset_subbands, x_axis='time')
# plt.ylabel('Sub-bands')
# plt.title('Sub-band onset strength')
#
# plt.show()
# playsound(y, sr)
