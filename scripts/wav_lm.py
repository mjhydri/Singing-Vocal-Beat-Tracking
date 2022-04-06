import torch
from WavLM import WavLM, WavLMConfig
import numpy as np
import torch.nn as nn
from torch import Tensor

class WAV_LM(nn.Module):
    def __init__(self, device=None, pretrained_dir=None):
        super(WAV_LM, self).__init__()
        self.device = device
        checkpoint = torch.load(pretrained_dir)
        cfg = WavLMConfig(checkpoint['cfg'])
        with torch.no_grad():
            if self.device!='cpu':
                self.device = torch.device(device)
            self.model = WavLM(cfg)
            self.model.load_state_dict(checkpoint['model'])
            self.to(self.device)
            #self.model.change_device(device)
            self.model.eval()

    def process_audio(self, audio):
        with torch.no_grad():
            audio = torch.from_numpy(np.expand_dims(audio, axis=0))#.to(self.device)
            if self.device!='cpu':
                audio = torch.tensor(audio, device=self.device)
            rep, layer_results = self.model.extract_features(audio, output_layer=self.model.cfg.encoder_layers, ret_layer_results=True)[0]
            layer_reps = torch.cat([x.transpose(0, 1) for x, _ in layer_results], dim=0)
            return layer_reps.T.detach()
