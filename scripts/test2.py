# this is the batch test module using a pytorch batchloader instance to load the data and obtain the performance over the large datasets
import pandas as pd
import torch
import numpy as np
from madmom.evaluation import BeatEvaluation
from madmom.features import DBNDownBeatTrackingProcessor
#from particle_filtering_1D_3 import particle_filter_1D
# from deterministic_1D import deterministic_1D
from Deterministic_1D_Published import deterministic_1D
# import timeit

def test(model, dataset, val_estimator, estim_dir=None, results_dir=None):
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

    # Make sure the model is in evaluation mode
    model.eval()

    # Create lists to hold the results
    # beat_results = []
    # downbeat_results = []
    results={}
    counter=0
    # Turn off gradient computation
    with torch.no_grad():
        particle = particle_filter_1D(beats_per_bar=[], fps=50, plot=False)
        inference = deterministic_1D(beats_per_bar=[], fps=50, plot=False)
        # Loop through the validation track ids
        for track in dataset:
            counter += 1
            # calculating ground truth times
            downs_g = track['ground_truth'][0][1].detach().numpy()
            downs_g = np.argwhere(downs_g == 1) * 0.02
            beats_g = track['ground_truth'][0][0].detach().numpy()
            beats_g = np.argwhere(beats_g == 1) * 0.02
            beats_g = np.sort(np.append(downs_g, beats_g))

            # start = timeit.default_timer()
            # Extract the activations of the track
            feats = track['feats'].transpose(1, 2)#.to(model.device)
            preds = model(feats)[0]
            preds = model.final_pred(preds)
            preds = preds.detach().numpy()
            preds = np.transpose(preds[:2, :])
            # stop = timeit.default_timer()
            # print(stop - start)
            if val_estimator == "PF":
                # Particle Filter instance
                data, x = particle.process(preds)
                # beats = np.sort(np.append(beats, downs))
            elif val_estimator == "DT":
                data = inference.process(preds)
            elif val_estimator == "DBN":
                DBNDownBeatTracking = DBNDownBeatTrackingProcessor(beats_per_bar=[round(len(beats_g) / len(downs_g))],
                                                                   fps=50, observation_lambda=16)
                data = DBNDownBeatTracking(preds)
            downs = data[:, 0][data[:, 1] == 1]
            beats = data[:, 0]
            # Evaluate the predictions
            beat_evaluation = BeatEvaluation(beats, beats_g, skip=5)
            down_evaluation = BeatEvaluation(downs, downs_g, skip=5)
            # beat_results.append(beat_evaluation)
            # downbeat_results.append(down_evaluation)
            results[track['track'][0]] = [beat_evaluation, down_evaluation]
            if counter > 100:
                (pd.DataFrame.from_dict(data=results, orient='index')
                 .to_csv('C:/research/1D_state_space\model80_460_GTZAN/model80_step460_deterministic_53.csv', header=False))


        # Average the results from all tracks
        # results = average_results(results)

    # return beat_results, downbeat_results


            # Add the results to the dictionary
            # results = append_results(results, track_results)

    # Average the results from all tracks
    # results = average_results(results)

    # return beat_results, downbeat_results



def get_fmeasure(beat_results,downbeat_results):
    beat_f = []
    down_f = []
    for i in range(len(beat_results)):
        beat_f.append(beat_results[i].fmeasure)
    for i in range(len(downbeat_results)):
        down_f.append(downbeat_results[i].fmeasure)
    if len(beat_f) == 0:
        beat_f = 0
    else:
        beat_f = sum(beat_f) / len(beat_f)
    if len(down_f) == 0:
        down_f = 0
    else:
        down_f = sum(down_f) / len(down_f)
    return beat_f, down_f
    # print(f'beat_f: {beat_f}') #+ f' iteration: {global_iter + 1}')
    # print(f'down_f: {down_f}') #+ f' iteration: {global_iter + 1}')