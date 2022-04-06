# this is the batch test module using a pytorch batchloader instance to load the data and obtain the performance over the large datasets 

import torch
import numpy as np
from madmom.evaluation import BeatEvaluation
from madmom.features import DBNBeatTrackingProcessor

from particle_filter import particle_filter
# import timeit

class TEST():
    """
    Implements the validation or evaluation loop for a model and dataset partition.
    Optionally save predictions and log results.
    Parameters
    ----------
    model : Beat/Downbeat extraction Model
      NN Model to validate or evalaute  
    dataset : Beat/Downbeat test Dataset
      Dataset (partition) to use for validation or evaluation. Note that the dataset should be the extracted features
    estim_dir : str or None (optional)
      Path to the directory to save predictions
    results_dir : str or None (optional)
      Path to the directory to log results
    Returns
    ----------
    results : dict
      Dictionary containing all relevant results for both beat and downbeat 
    """
    def __init__(self,model, dataset, val_estimator, estim_dir=None, results_dir=None):
        if val_estimator == "DBN":
            self.estimate = DBNBeatTrackingProcessor(fps=50)
            self.model = model
            self.dataset = dataset
            self.estim_dir = estim_dir
            self.resutls_dir = results_dir

    def process(self):
        # Make sure the model is in evaluation mode
        self.model.eval()

        # Create lists to hold the results
        beat_results = []

        # Turn off gradient computation
        with torch.no_grad():
            # Loop through the validation track ids
            for track in self.dataset:
                # calculating ground truth times
                beats_g = track['gt'][0].detach().numpy()
                beats_g = np.argwhere(beats_g == 1) * 0.02
                beats_g = np.squeeze(beats_g)
                if len(beats_g) < 3:
                    continue
                # start = timeit.default_timer()
                # Extract the activations of the track
                embeddings = track['embeddings'].transpose(1, 2).to(self.model.device)
                preds = self.model(embeddings)
                preds = self.model.final_pred(preds)
                preds = preds.detach().numpy() #.cpu() before .numpy() if tensor
                # preds = np.transpose(preds[:2, :])
                # stop = timeit.default_timer()
                # print(stop - start)
                beats= self.estimate.process_offline(preds)
                # Evaluate the predictions
                beat_evaluation = BeatEvaluation(beats, beats_g)
                beat_results.append(beat_evaluation)

        return beat_results




# class VALIDATE():
#     """
#     Implements the validation or evaluation loop for a model and dataset partition.
#     Optionally save predictions and log results.
#
#     Parameters
#     ----------
#     model : Beat/Downbeat extraction Model
#       Model to validate or evalaute
#     dataset : Beat/Downbeat test Dataset
#       Dataset (partition) to use for validation or evaluation
#     estim_dir : str or None (optional)
#       Path to the directory to save predictions
#     results_dir : str or None (optional)
#       Path to the directory to log results
#
#     Returns
#     ----------
#     results : dict
#       Dictionary containing all relevant results averaged across all tracks
#     """
#     def __init__(self,model, dataset, val_estimator, estim_dir=None, results_dir=None):
#         if val_estimator == "DBN":
#             self.estimate = DBNBeatTrackingProcessor(fps=50)
#             self.model = model
#             self.dataset = dataset
#             self.estim_dir = estim_dir
#             self.resutls_dir = results_dir
#     def process(self):
#         # Make sure the model is in evaluation mode
#         self.model.eval()
#
#         # Create a lists to hold the results
#         beat_results = []
#         downbeat_results = []
#
#         # Turn off gradient computation
#         with torch.no_grad():
#             # Loop through the validation track ids
#             for track in self.dataset:
#                 # calculating ground truth times
#                 beats_g = track['ground_truth'][0][0].detach().numpy()
#                 beats_g = np.argwhere(beats_g == 1) * 0.02
#                 beats_g = np.sort(beats_g)
#
#                 # Transcribe the track
#                 embeddings = track['embeddings'].transpose(1, 2).to(self.model.device)
#                 preds = self.model(embeddings)
#                 preds = self.model.final_pred(preds)
#                 preds = preds.detach().numpy()
#                 preds = np.transpose(preds[:2, :])
#                 # if self.val_estimator == "PF":
#                 #     # PF instance
#                 #     particle = particle_filter(beats_per_bar=[round(len(beats_g)/len(downs_g))], fps=50)
#                 #     beats, downs = particle.process(preds)
#                 #     beats = np.sort(np.append(beats, downs))
#                 if self.val_estimator == "DBN":
#                     DBNDownBeatTracking = DBNDownBeatTrackingProcessor(beats_per_bar=[round(len(beats_g)/len(downs_g))], fps=50, observation_lambda=16)
#                     data = DBNDownBeatTracking(preds)
#                     beats = data[:, 0]
#                 # Evaluate the predictions
#                 beat_evaluation = BeatEvaluation(beats, beats_g, skip=5)
#                 beat_results.append(beat_evaluation)
#
#
#         # Average the results from all tracks
#         # results = average_results(results)
#
#         return beat_results


def get_fmeasure(beat_results):
    beat_f = []
    for i in range(len(beat_results)):
        beat_f.append(beat_results[i].fmeasure)
    if len(beat_f) == 0:
        beat_f = 0
    else:
        beat_f = sum(beat_f) / len(beat_f)
    return beat_f
    # print(f'beat_f: {beat_f}') #+ f' iteration: {global_iter + 1}')
    # print(f'down_f: {down_f}') #+ f' iteration: {global_iter + 1}')