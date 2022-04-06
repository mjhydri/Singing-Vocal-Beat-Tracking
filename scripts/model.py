import torch.nn as nn
import torch
from fast_transformers.builders import TransformerEncoderBuilder
from timeit import default_timer as timer

class MODEL(nn.Module):  #model for Wav_lm
    def __init__(self, wavlm_type='base+', n_layers=2, n_heads=8, query_dimensions=64, value_dimensions=64, feed_forward_dimensions=1024, device=None):
        super(MODEL, self).__init__()
        # Create the builder for our transformers
        assert wavlm_type == 'base+' or wavlm_type == 'large'
        if wavlm_type == 'base+':
            self.weights = torch.nn.parameter.Parameter(data=torch.ones(13, dtype=torch.float32), requires_grad=True)
        elif wavlm_type == 'large':
            self.weights = torch.nn.parameter.Parameter(data=torch.ones(25, dtype=torch.float32), requires_grad=True)
        builder = TransformerEncoderBuilder.from_kwargs(
            n_layers=n_layers,
            n_heads=n_heads,
            query_dimensions=query_dimensions,
            value_dimensions=value_dimensions,
            feed_forward_dimensions=feed_forward_dimensions
        )

        # Build a transformer with linear attention
        builder.attention_type = "linear"
        self.linear_transformer = builder.get()
        self.linear = nn.Linear(in_features=n_heads*value_dimensions, out_features=1)
        self.sigmoid = nn.Sigmoid()
        self.device = device
        self.to(self.device)

    def forward(self, embeddings):
        weights = self.sigmoid(self.weights)
        embeddings = torch.matmul(embeddings, weights)
        embeddings = self.linear_transformer(embeddings)
        preds = self.linear(embeddings)
        return preds.squeeze()

    def final_pred(self, input):
        return self.sigmoid(input)


class MODEL2(nn.Module):  #model for distilHubert
    def __init__(self, distil_type='base', n_layers=2, n_heads=8, query_dimensions=64, value_dimensions=64, feed_forward_dimensions=1024, device=None):
        super(MODEL2, self).__init__()
        # Create the builder for our transformers
        assert distil_type == 'base' or distil_type == 'normal'
        builder = TransformerEncoderBuilder.from_kwargs(
            n_layers=n_layers,
            n_heads=n_heads,
            query_dimensions=query_dimensions,
            value_dimensions=value_dimensions,
            feed_forward_dimensions=feed_forward_dimensions
        )

        # Build a transformer with linear attention
        builder.attention_type = "linear"
        self.linear_transformer = builder.get()
        self.linear = nn.Linear(in_features=n_heads*value_dimensions, out_features=1)
        self.sigmoid = nn.Sigmoid()
        self.device = device
        self.to(self.device)

    def forward(self, embeddings):
        embeddings = self.linear_transformer(torch.squeeze(embeddings, -1))
        preds = self.linear(embeddings)
        return preds.squeeze()

    def final_pred(self, input):
        return self.sigmoid(input)



class MODEL3(nn.Module):  #bmodel for log_spectrogram
    def __init__(self, n_layers=2, n_heads=8, query_dimensions=64, value_dimensions=64, feed_forward_dimensions=1024, device=None):
        super(MODEL3, self).__init__()
        # Create the builder for our transformers
        builder = TransformerEncoderBuilder.from_kwargs(
            n_layers=n_layers,
            n_heads=n_heads,
            query_dimensions=query_dimensions,
            value_dimensions=value_dimensions,
            feed_forward_dimensions=feed_forward_dimensions
        )

        # Build a transformer with linear attention
        builder.attention_type = "linear"
        self.linear_transformer = builder.get()
        self.linear0 = nn.Linear(in_features=586, out_features=n_heads * value_dimensions)
        self.linear = nn.Linear(in_features=n_heads*value_dimensions, out_features=1)
        self.sigmoid = nn.Sigmoid()
        self.device = device
        self.to(self.device)

    def forward(self, embeddings):
        embeddings = self.linear0(embeddings)
        embeddings = self.linear_transformer(embeddings)
        preds = self.linear(embeddings)
        return preds.squeeze()

    def final_pred(self, input):
        return self.sigmoid(input)