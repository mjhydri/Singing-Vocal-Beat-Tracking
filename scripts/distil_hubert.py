import torch
# import s3prl.hub as hub
import torch.nn as nn


class DISTILHUBERT(nn.Module):
    def __init__(self, device=None):
        super(DISTILHUBERT, self).__init__()
        self.device = device
        with torch.no_grad():
            from s3prl.hub import distilhubert
            self.model = distilhubert()
            if self.device != 'cpu':
                self.device = torch.device(device)
            self.to(self.device)
            self.model.eval()

    def process_audio(self, audio):
        with torch.no_grad():
            audio = torch.from_numpy(audio)#.to(self.device)
            if self.device != 'cpu':
                audio = torch.tensor(audio, device=self.device)
            results = self.model([audio])
            representation = results["paper"]
            return representation.T.detach()
