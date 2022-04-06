# import torch
# from fast_transformers.builders import TransformerEncoderBuilder
# from timeit import default_timer as timer
# # Create the builder for our transformers
# builder = TransformerEncoderBuilder.from_kwargs(
#     n_layers=2,
#     n_heads=8,
#     query_dimensions=64,
#     value_dimensions=64,
#     feed_forward_dimensions=1024
# )
#
# # Build a transformer with softmax attention
# builder.attention_type = "full"
# softmax_model = builder.get()
#
# # Build a transformer with linear attention
# builder.attention_type = "linear"
# linear_model = builder.get()
#
# # Construct the dummy input
# X = torch.rand(10, 400, 8*64)
#
# # Prepare everythin for CUDA
# # X = X.cuda()
# # softmax_model.cuda()
# softmax_model.eval()
# # linear_model.cuda()
# linear_model.eval()
#
# # Warmup the GPU
# # with torch.no_grad():
# #     softmax_model(X)
# #     linear_model(X)
# # torch.cuda.synchronize()
#
# # Measure the execution time
# # softmax_start = torch.cuda.Event(enable_timing=True)
# # softmax_end = torch.cuda.Event(enable_timing=True)
# # linear_start = torch.cuda.Event(enable_timing=True)
# # linear_end = torch.cuda.Event(enable_timing=True)
#
# with torch.no_grad():
#     start = timer()
#     y = softmax_model(X)
#     end = timer()
#     # torch.cuda.synchronize()
#     print("Softmax: ", end - start, "s")
#     # Softmax: 144 ms (on a GTX1080Ti)
#
# with torch.no_grad():
#     start = timer()
#     y = linear_model(X)
#     end = timer()
#     # torch.cuda.synchronize()
#     print("Linear: ", end - start, "s")
#     # Linear: 68 ms (on a GTX1080Ti)
#
# print('input: torch.rand(10, 400, 8*64) ')
# print('Music duration: 3m')
# print('Number of frames: 9000')

# from s3prl.upstream.hubert import distilhubert
# import torch
# from pretrained_models.distiller.expert import UpstreamExpert
# a=UpstreamExpert()
# wavs = [torch.randn(16000) for _ in range(4)]
# b=a()

# import s3prl.hub as hub
# print(dir(hub))
# print('hi')
import torch
import s3prl.hub as hub
from s3prl.hub import distilhubert
wavs = [torch.randn(16000) for _ in range(4)]
pretrained_model = distilhubert()
results = pretrained_model(wavs)

# The representation used in the paper
representation = results["paper"]

# All hidden states
hidden_states = results["hidden_states"]