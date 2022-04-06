from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.audio.spectrogram import (
    FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor,
    SpectrogramDifferenceProcessor)
from madmom.processors import ParallelProcessor, SequentialProcessor
import numpy as np
import torch.nn as nn
import torch

# feature extractor for Magnitude spectrogoram and its differences

class LOG_SPECT(nn.Module):
    def __init__(self, num_channels=1, sample_rate=22050, win_length=2048, hop_size=512, n_bands=[12], device=None):
        super(LOG_SPECT, self).__init__()
        self.device=device
        if self.device != 'cpu':
            self.device = torch.device(device)
        sig = SignalProcessor(num_channels=num_channels, win_length=win_length, sample_rate=sample_rate)
        self.sample_rate = sample_rate
        self.hop_length = hop_size
        self.num_channels = num_channels
        multi = ParallelProcessor([])
        frame_sizes = [win_length, 2* win_length, 4* win_length]  # [1024, 2048, 4096]
        num_bands = [6, 12, 24] #n_bands  #
        for frame_size, num_bands in zip(frame_sizes, num_bands):
            frames = FramedSignalProcessor(frame_size=frame_size, hop_size=hop_size)     #    origin='stream' num_frames=1    num_frames=1x origin='online, num_frames=1  origin='stream', num_frames=1
            stft = ShortTimeFourierTransformProcessor()  # caching FFT window
            filt = FilteredSpectrogramProcessor(
                num_bands=num_bands, fmin=30, fmax=17000, norm_filters=True)
            spec = LogarithmicSpectrogramProcessor(mul=1, add=1)
            diff = SpectrogramDifferenceProcessor(
                diff_ratio=0.5, positive_diffs=True, stack_diffs=np.hstack)
            # process each frame size with spec and diff sequentially
            multi.append(SequentialProcessor((frames, stft, filt, spec, diff)))
        # stack the features and processes everything sequentially
        self.pipe = SequentialProcessor((sig, multi, np.hstack))
        self.to(self.device)

    def process_audio(self, audio):
        feats = self.pipe(audio).T
        return torch.tensor(feats, device=self.device)
