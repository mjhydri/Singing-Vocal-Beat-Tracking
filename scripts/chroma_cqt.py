import librosa

from common import FeatureModule


class CHROMA(FeatureModule):
    def __init__(self, sample_rate=22050, hop_size=441, n_octaves=7, bins_per_octave=36, fmin=73):
        self.sample_rate = sample_rate
        self.hop_length = hop_size
        self.bins_per_octave = bins_per_octave
        self.n_octaves = n_octaves
        self.fmin = fmin

    def process_audio(self, audio):
        result = librosa.feature.chroma_cqt(y=audio, sr=self.sample_rate, C=None, hop_length=self.hop_length,
                                   fmin=self.fmin, norm=None, threshold=0.0, tuning=None, n_chroma=12,
                                   n_octaves=self.n_octaves, window=None,
                                   bins_per_octave=self.bins_per_octave, cqt_mode='full')
        return result


# y1, sr1 = librosa.load(librosa.ex('trumpet'))
# chroma= CHROMA(sample_rate=sr1)
# c= chroma.process_audio(y1)