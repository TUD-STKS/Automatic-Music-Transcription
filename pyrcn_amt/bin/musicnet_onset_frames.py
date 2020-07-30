import numpy as np
import argparse
import os
import warnings
from shutil import copyfile
from sklearn.model_selection import ParameterGrid
from sklearn.base import clone
from joblib import dump, load, Parallel, delayed

from pyrcn.echo_state_network import ESNRegressor
from pyrcn_amt.datasets import musicnet
from pyrcn_amt.feature_extraction.audio_features import parse_feature_settings, create_processors, load_sound_file, extract_features
from pyrcn_amt.feature_extraction.discretize_labels import discretize_notes
from pyrcn_amt.evaluation import loss_functions
from pyrcn_amt.config.parse_config_file import parse_config_file
from pyrcn_amt.evaluation.multipitch_scoring import determine_threshold
from pyrcn_amt.post_processing.binarize_output import thresholding
from pyrcn_amt.evaluation.multipitch_scoring import get_mir_eval_rows


def train_esn(base_esn, params, feature_settings, pre_processor, scaler, training_set):
    print(params)
    esn = clone(base_esn)
    esn.set_params(**params)
    esn.set_params(teacher_scaling=0.1)
    for fids in training_set:
        s = load_sound_file(file_name=fids[0], feature_settings=feature_settings)
        U = extract_features(s=s, pre_processor=pre_processor, scaler=scaler)
        pitch_labels = musicnet.get_pitch_labels(fids[1])
        y_true = discretize_notes(pitch_labels,  fps=feature_settings['fps'], num_pitches=128, target_widening=True, length=U.shape[0])
        esn.partial_fit(X=U, y=y_true, update_output_weights=False)
    esn.finalize()
    return esn


def opt_function(base_esn, params, feature_settings, pre_processor, scaler, training_set, test_set, loss_function):
    esn = train_esn(base_esn, params, feature_settings, pre_processor, scaler, training_set)

    #  Validation
    train_loss = []
    for fids in training_set:
        s = load_sound_file(file_name=fids[0], feature_settings=feature_settings)
        U = extract_features(s=s, pre_processor=pre_processor, scaler=scaler)
        pitch_labels = musicnet.get_pitch_labels(fids[1])
        y_true = discretize_notes(pitch_labels,  fps=feature_settings['fps'], num_pitches=128, target_widening=True, length=U.shape[0])
        y_pred = esn.predict(X=U, keep_reservoir_state=False)
        if isinstance(loss_function, list):
            train_loss.append([loss(y_true, y_pred) for loss in loss_function])
        else:
            train_loss.append([loss_function(y_true, y_pred)])

    val_loss = []
    for fids in test_set:
        s = load_sound_file(file_name=fids[0], feature_settings=feature_settings)
        U = extract_features(s=s, pre_processor=pre_processor, scaler=scaler)
        pitch_labels = musicnet.get_pitch_labels(fids[1])
        y_true = discretize_notes(pitch_labels,  fps=feature_settings['fps'], num_pitches=128, target_widening=True, length=U.shape[0])
        y_pred = esn.predict(X=U, keep_reservoir_state=False)
        if isinstance(loss_function, list):
            val_loss.append([loss(y_true, y_pred) for loss in loss_function])
        else:
            val_loss.append([loss_function(y_true, y_pred)])

    return [np.mean(train_loss, axis=0), np.mean(val_loss, axis=0)]


def score_function(base_esn, params, feature_settings, pre_processor, scaler, training_set, test_set):
    try:
        f_name = r"C:\Users\Steiner\Documents\Python\multipitch-tracking\experiments\experiment_0\models\esn_500_False.joblib"
        esn = load(f_name)
    except FileNotFoundError:
        esn = train_esn(base_esn, params, feature_settings, pre_processor, scaler, training_set)

    # Training set
    Y_pred_train = []
    Pitch_times_train = []
    for fids in training_set:
        s = load_sound_file(file_name=fids[0], feature_settings=feature_settings)
        U = extract_features(s=s, pre_processor=pre_processor, scaler=scaler)
        pitch_labels = discretize_notes(musicnet.get_pitch_labels(fids[1]), target_widening=False, length=U.shape[0])
        Pitch_times_train.append(pitch_labels)
        y_pred = esn.predict(X=U, keep_reservoir_state=False)
        Y_pred_train.append(y_pred)
    train_scores = determine_threshold(Y_true=Pitch_times_train, Y_pred=Y_pred_train, threshold=np.linspace(start=0.1, stop=0.4, num=16))

    # Test set
    Y_pred_test = []
    Pitch_times_test = []
    for fids in test_set:
        s = load_sound_file(file_name=fids[0], feature_settings=feature_settings)
        U = extract_features(s=s, pre_processor=pre_processor, scaler=scaler)
        pitch_labels = discretize_notes(musicnet.get_pitch_labels(fids[1]), target_widening=False, length=U.shape[0])
        Pitch_times_test.append(pitch_labels)
        y_pred = esn.predict(X=U, keep_reservoir_state=False)
        Y_pred_test.append(y_pred)
    test_scores = determine_threshold(Y_true=Pitch_times_test, Y_pred=Y_pred_test, threshold=np.linspace(start=0.1, stop=0.4, num=16))

    return train_scores, test_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate Echo State Network')
    parser.add_argument('-inf',  type=str)

    in_file = r"Z:\Projekt-Musik-Datenbank\musicNET\train_data\1727.wav"
    out_file = r"C:\Users\Steiner\Documents\Python\Automatic-Music-Transcription\1727.f0"
    args = parser.parse_args()
    test_multipitch_tracking(args.inf, in_file, out_file)
