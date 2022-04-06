import torch
import fairseq
import librosa
import numpy as np
import os
import pickle
# from transformers import (
#     Wav2Vec2ForCTC,
#     Wav2Vec2Processor,
#     AutoTokenizer,
#     AutoModelWithLMHead)
#
# from transformers import Wav2Vec2CTCTokenizer
# tokenizer = Wav2Vec2CTCTokenizer("C:/research/vocal_beat\wav2vec_large_multilingual/facebookwav2vec2-large-xlsr-53/toy_vocab.txt", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
# cp_path = 'C:/research/vocal_beat\scripts\models/wav2vec_large.pt'
# base_model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
#
# def make_data(vocal,annotations, dir):
#     model = base_model[0].copy()
#     model.eval()
#     vocal = np.expand_dims(vocal, axis=0)
#     vocal = torch.from_numpy(vocal)
#     z = model.feature_extractor(vocal)
#     c = model.feature_aggregator(z)
#     gt = np.zeros((2, c.shape[-1]))
#     beat_frames = librosa.time_to_frames(annotations, sr=16000, hop_length=320)
#     gt[0, beat_frames] = 1
#
# # wav_input_16khz = torch.randn(1,5000)
# dir = r"C:\datasets\URSing"
# with os.scandir(dir+'vocal_annotations') as it:
#    for entry in it:
#         file = open(rf"C:\datasets\urSing/annotations/{entry.name}", 'rb')
#         annot = pickle.load(file)
# vocal, _ = librosa.load("C:\datasets\GTZAN/audio/blues/blues.00000.wav", sr=16000)
# vocal = np.expand_dims(vocal, axis=0)
# wav_input_16khz = torch.from_numpy(vocal)
# wav_input_16khz = torch.randn(1, 16000*2)
# z = model.feature_extractor(wav_input_16khz)
# c = model.feature_aggregator(z)
# print('hi')

#################################################

#
# import soundfile as sf
# import torch
# from datasets import load_dataset
# from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
#
# # load pretrained model
# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
# model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
# librispeech_samples_ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
# # load audio
# # audio_input, sample_rate = sf.read(librispeech_samples_ds[0]["file"])
# audio_input, sample_rate = librosa.load("C:\datasets\GTZAN/audio/blues/blues.00001.wav", sr=16000)
# # pad input values and return pt tensor
# input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values
#
# # INFERENCE
#
# # retrieve logits & take argmax
# logits = model(input_values).logits
# predicted_ids = torch.argmax(logits, dim=-1)
#
# # transcribe
# transcription = processor.decode(predicted_ids[0])
#
# # FINE-TUNE
#
# target_transcription = "A MAN SAID TO THE UNIVERSE I EXIST"
#
# # encode labels
# with processor.as_target_processor():
#   labels = processor(target_transcription, return_tensors="pt").input_ids
#
# # compute loss by passing labels
# loss = model(input_values, labels=labels).loss
# loss.backward()


####################################################

# import torch
# from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForPreTraining, PreTrainedModel
# from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices
# from transformers import Wav2Vec2Processor

# import torch
# from fairseq.models.wav2vec import Wav2VecModel

# voidful/wav2vec2-xlsr-multilingual-56"
#"facebook/wav2vec2-large-xlsr-53"

# feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53", feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
# feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("voidful/wav2vec2-xlsr-multilingual-56")
# feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
# model = Wav2Vec2ForPreTraining.from_pretrained("voidful/wav2vec2-xlsr-multilingual-56",do_normalize=True, return_attention_mask=True)
# model = Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-large-xlsr-53")
# PreTrainedModel.save_pretrained(model,save_directory="C:\datasets\GTZAN/")
# aaa=Wav2Vec2Processor(feature_extractor)
# processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
# processor = Wav2Vec2Processor.from_pretrained("voidful/wav2vec2-xlsr-multilingual-56")
# audio_input, sample_rate = librosa.load("C:\datasets\GTZAN/audio/blues/blues.00001.wav", sr=16000)
# input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values
# batch_size, raw_sequence_length = input_values.shape
# sequence_length = model._get_feat_extract_output_lengths(raw_sequence_length)
# mask_time_indices = _compute_mask_indices((batch_size, sequence_length), mask_prob=0.2, mask_length=2)
# mask_time_indices = torch.tensor(mask_time_indices, device=input_values.device, dtype=torch.long)
# with torch.no_grad():
#   outputs = model(input_values, mask_time_indices=mask_time_indices)
#   # compute cosine similarity between predicted (=projected_states) and target (=projected_quantized_states)
#   cosine_sim = torch.cosine_similarity(outputs.projected_states, outputs.projected_quantized_states, dim=-1)
#   # to show that cosine similarity is much higher than random
#   print(cosine_sim[mask_time_indices.to(torch.bool)].mean() > 0.5)
####################################
# model = Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-large-xlsr-53")
# audio_input, sample_rate = librosa.load("C:\datasets\GTZAN/audio/blues/blues.00001.wav", sr=16000)
# with torch.no_grad():
#     aaa=model.forward(torch.tensor(audio_input).unsqueeze(0))


#################################### distilhubert example #############

# import torch
# import librosa
#from s3prl.hub import distilhubert
# import torch
# model=torch.hub.load('s3prl/s3prl','distilhubert')
# audio_input, sample_rate = librosa.load("C:\datasets\GTZAN/audio/blues/blues.00001.wav", sr=16000)
# # wavs = [torch.randn(16000) for _ in range(4)]
# wavs = audio_input
# pretrained_model = distilhubert()
# results = pretrained_model(wavs)
#
# # The representation used in the paper
# representation = results["paper"]
#
# # All hidden states
# hidden_states = results["hidden_states"]

############################################## Wavlm example ##########

import torch
from WavLM import WavLM, WavLMConfig

# load the pre-trained checkpoints
checkpoint = torch.load(r'C:\research\vocal_beat\wavlm\WavLM-Large.pt')
cfg = WavLMConfig(checkpoint['cfg'])
model = WavLM(cfg)
model.load_state_dict(checkpoint['model'])
model.eval()

# extract the representation of last layer
# wav_input_16khz = torch.randn(1,16000)
# rep = model.extract_features(wav_input_16khz)[0]

# extract the representation of each layer
wav_input_16khz = torch.randn(1, 16000)
rep, layer_results = model.extract_features(wav_input_16khz, output_layer=model.cfg.encoder_layers, ret_layer_results=True)[0]
layer_reps = torch.cat([x.transpose(0, 1) for x, _ in layer_results], dim=0)
weigths = torch.nn.parameter.Parameter(data=torch.ones(25, dtype=torch.float32), requires_grad=True)
embedings = torch.matmul(layer_reps.T, weigths)



print('asd')

