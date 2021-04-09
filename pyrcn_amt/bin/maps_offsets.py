import numpy as np
import argparse
import os
import warnings
from shutil import copyfile
from sklearn.model_selection import ParameterGrid
from sklearn.base import clone
from joblib import dump, load, Parallel, delayed

from pyrcn.echo_state_network import ESNRegressor
from pyrcn_amt.datasets import maps_dataset
from pyrcn_amt.feature_extraction.audio_features import parse_feature_settings, create_processors, load_sound_file, extract_features
from pyrcn_amt.feature_extraction.discretize_labels import discretize_offset_labels
from pyrcn_amt.evaluation import loss_functions
from pyrcn_amt.config.parse_configuration import parse_config_file
from pyrcn_amt.evaluation.onset_scoring import determine_peak_picking_threshold
from pyrcn_amt.post_processing.binarize_output import peak_picking


def train_maps_offsets(config_file):
    io_params, esn_params, fit_params, feature_settings, loss_fn, n_jobs = parse_config_file(config_file)
    base_esn = ESNRegressor()
    base_esn.set_params(**esn_params)
    feature_settings = parse_feature_settings(feature_settings)
    pre_processor, scaler = create_processors(feature_settings=feature_settings)

    in_folder = io_params['in_folder']
    out_folder = io_params['out_folder']

    # Make Paths
    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)
    if not os.path.isdir(os.path.join(out_folder, 'train')):
        os.mkdir(os.path.join(out_folder, 'train'))
    if not os.path.isdir(os.path.join(out_folder, 'validation')):
        os.mkdir(os.path.join(out_folder, 'validation'))
    if not os.path.isdir(os.path.join(out_folder, 'test')):
        os.mkdir(os.path.join(out_folder, 'test'))
    if not os.path.isdir(os.path.join(out_folder, 'models')):
        os.mkdir(os.path.join(out_folder, 'models'))

    #   Optimizer
    if loss_fn == "bce":
        loss_function = loss_functions.bce
    elif loss_fn == "cosine_distance":
        loss_function = loss_functions.cosine_distance
    elif loss_fn == "correlation":
        loss_function = loss_functions.correlation
    elif loss_fn == "mse":
        loss_function = loss_functions.mse
    elif loss_fn == "all":
        loss_function = [loss_functions.cosine_distance, loss_functions.correlation, loss_functions.mse]
    else:
        warnings.warn("No valid loss function specified. Using mean_squared_error from sklearn!", UserWarning)
        loss_function = loss_functions.mean_squared_error

    # replicate config file and store results there
    copyfile(config_file, os.path.join(out_folder, 'config.yaml'))

    losses = []
    for k in range(4):
        (training_set, validation_set), _ = maps_dataset.load_dataset(dataset_path=in_folder, fold_id=k, validation=True, configuration=2)
        tmp_losses = Parallel(n_jobs=n_jobs)(delayed(opt_function)(base_esn, params, feature_settings, pre_processor, scaler, training_set, validation_set, loss_function, out_folder) for params in ParameterGrid(fit_params))
        losses.append(tmp_losses)
    dump(losses, filename=os.path.join(out_folder, 'losses.lst'))


def validate_maps_offsets(config_file):
    io_params, esn_params, fit_params, feature_settings, loss_fn, n_jobs = parse_config_file(config_file)
    base_esn = ESNRegressor()
    base_esn.set_params(**esn_params)
    feature_settings = parse_feature_settings(feature_settings)
    pre_processor, scaler = create_processors(feature_settings=feature_settings)

    in_folder = io_params['in_folder']
    out_folder = io_params['out_folder']

    # Make Paths
    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)
    if not os.path.isdir(os.path.join(out_folder, 'train')):
        os.mkdir(os.path.join(out_folder, 'train'))
    if not os.path.isdir(os.path.join(out_folder, 'validation')):
        os.mkdir(os.path.join(out_folder, 'validation'))
    if not os.path.isdir(os.path.join(out_folder, 'test')):
        os.mkdir(os.path.join(out_folder, 'test'))
    if not os.path.isdir(os.path.join(out_folder, 'models')):
        os.mkdir(os.path.join(out_folder, 'models'))

    # replicate config file and store results there
    copyfile(config_file, os.path.join(out_folder, 'config.yaml'))

    scores = []
    for k in range(4):
        training_set, test_set = maps_dataset.load_dataset(dataset_path=in_folder, fold_id=k, validation=False, configuration=2)
        tmp_scores = Parallel(n_jobs=n_jobs)(delayed(score_function)(base_esn, params, feature_settings, pre_processor, scaler, training_set, test_set, out_folder) for params in ParameterGrid(fit_params))
        scores.append(tmp_scores)
    dump(scores, filename=os.path.join(out_folder, 'scores.lst'))


def test_maps_offsets(config_file, in_file, out_file):
    io_params, esn_params, fit_params, feature_settings, loss_fn, n_jobs = parse_config_file(config_file)
    base_esn = ESNRegressor()
    base_esn.set_params(**esn_params)
    feature_settings = parse_feature_settings(feature_settings)
    pre_processor, scaler = create_processors(feature_settings=feature_settings)

    in_folder = io_params['in_folder']
    out_folder = io_params['out_folder']

    # Make Paths
    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)
    if not os.path.isdir(os.path.join(out_folder, 'train')):
        os.mkdir(os.path.join(out_folder, 'train'))
    if not os.path.isdir(os.path.join(out_folder, 'validation')):
        os.mkdir(os.path.join(out_folder, 'validation'))
    if not os.path.isdir(os.path.join(out_folder, 'test')):
        os.mkdir(os.path.join(out_folder, 'test'))
    if not os.path.isdir(os.path.join(out_folder, 'models')):
        os.mkdir(os.path.join(out_folder, 'models'))

    # replicate config file and store results there
    copyfile(config_file, os.path.join(out_folder, 'config.yaml'))
    try:
        f_name = r"C:\Users\Steiner\Documents\Python\Automatic-Music-Transcription\pyrcn_amt\experiments\experiment_8\models\esn_500_False.joblib"
        esn = load(f_name)
    except FileNotFoundError:
        training_set, test_set = maps_dataset.load_dataset(dataset_path=in_folder, fold_id=0, validation=False)
        Parallel(n_jobs=n_jobs)(delayed(train_esn)(base_esn, params, feature_settings, pre_processor, scaler, training_set + test_set, out_folder) for params in ParameterGrid(fit_params))

    s = load_sound_file(file_name=in_file, feature_settings=feature_settings)
    U = extract_features(s=s, pre_processor=pre_processor, scaler=scaler)
    y_pred = esn.predict(X=U, keep_reservoir_state=False)
    offset_times_res = peak_picking(y_pred, 0.4)
    with open(out_file, 'w') as f:
        for offset_time in offset_times_res:
            f.write('{0}'.format(offset_time))
            f.write('\n')


def train_esn(base_esn, params, feature_settings, pre_processor, scaler, training_set, out_folder):
    print(params)
    esn = clone(base_esn)
    esn.set_params(**params)
    for fids in training_set:
        s = load_sound_file(file_name=fids[0], feature_settings=feature_settings)
        U = extract_features(s=s, pre_processor=pre_processor, scaler=scaler)
        offset_labels = maps_dataset.get_offset_labels(fids[1])
        y_true = discretize_offset_labels(offset_labels,  fps=feature_settings['fps'], target_widening=True, length=U.shape[0])
        esn.partial_fit(X=U, y=y_true, update_output_weights=False)
    esn.finalize()
    serialize = True
    if serialize:
        dump(esn, os.path.join(out_folder, "models", "esn_" + str(params['reservoir_size']) + '_' + str(params['bi_directional']) + '.joblib'))
    return esn


def opt_function(base_esn, params, feature_settings, pre_processor, scaler, training_set, test_set, loss_function, out_folder):
    esn = train_esn(base_esn, params, feature_settings, pre_processor, scaler, training_set, out_folder)

    #  Validation
    train_loss = []
    for fids in training_set:
        s = load_sound_file(file_name=fids[0], feature_settings=feature_settings)
        U = extract_features(s=s, pre_processor=pre_processor, scaler=scaler)
        offset_labels = maps_dataset.get_offset_labels(fids[1])
        y_true = discretize_offset_labels(offset_labels,  fps=feature_settings['fps'], target_widening=False, length=U.shape[0])
        y_pred = esn.predict(X=U, keep_reservoir_state=False)
        if isinstance(loss_function, list):
            train_loss.append([loss(y_true, y_pred) for loss in loss_function])
        else:
            train_loss.append([loss_function(y_true, y_pred)])

    val_loss = []
    for fids in test_set:
        s = load_sound_file(file_name=fids[0], feature_settings=feature_settings)
        U = extract_features(s=s, pre_processor=pre_processor, scaler=scaler)
        offset_labels = maps_dataset.get_offset_labels(fids[1])
        y_true = discretize_offset_labels(offset_labels,  fps=feature_settings['fps'], target_widening=False, length=U.shape[0])
        y_pred = esn.predict(X=U, keep_reservoir_state=False)
        if isinstance(loss_function, list):
            val_loss.append([loss(y_true, y_pred) for loss in loss_function])
        else:
            val_loss.append([loss_function(y_true, y_pred)])

    return [np.mean(train_loss, axis=0), np.mean(val_loss, axis=0)]


def score_function(base_esn, params, feature_settings, pre_processor, scaler, training_set, test_set, out_folder):
    try:
        f_name = os.path.join(out_folder, "models", "esn_" + str(params["reservoir_size"]) + "_" + str(params['bi_directional']) + ".joblib")
        esn = load(f_name)
    except FileNotFoundError:
        esn = train_esn(base_esn, params, feature_settings, pre_processor, scaler, training_set, out_folder)

    # Training set
    Y_pred_train = []
    Offset_times_train = []
    for fids in training_set:
        s = load_sound_file(file_name=fids[0], feature_settings=feature_settings)
        U = extract_features(s=s, pre_processor=pre_processor, scaler=scaler)
        offset_labels = maps_dataset.get_offset_labels(fids[1])
        Offset_times_train.append(offset_labels)
        y_pred = esn.predict(X=U, keep_reservoir_state=False)
        Y_pred_train.append(y_pred)
    train_scores = determine_peak_picking_threshold(Onset_times_ref=Offset_times_train, odf=Y_pred_train, threshold=np.linspace(start=0.2, stop=0.5, num=16))

    # Test set
    Y_pred_test = []
    Offset_times_test = []
    for fids in test_set:
        s = load_sound_file(file_name=fids[0], feature_settings=feature_settings)
        U = extract_features(s=s, pre_processor=pre_processor, scaler=scaler)
        offset_labels = maps_dataset.get_offset_labels(fids[1])
        Offset_times_test.append(offset_labels)
        y_pred = esn.predict(X=U, keep_reservoir_state=False)
        Y_pred_test.append(y_pred)
    test_scores = determine_peak_picking_threshold(Onset_times_ref=Offset_times_test, odf=Y_pred_test, threshold=np.linspace(start=0.2, stop=0.5, num=16))

    return train_scores, test_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate Echo State Network')
    parser.add_argument('-inf',  type=str)

    in_file = r"Z:\Projekt-Musik-Datenbank\MultiPitch-Tracking\MAPS_SptkBGAm\MUS\MAPS_MUS-liz_et2_SptkBGAm.wav"
    out_file = r"C:\Users\Steiner\Documents\Python\Automatic-Music-Transcription\MAPS_MUS-liz_et2_SptkBGAm.offsets"
    args = parser.parse_args()
    validate_maps_offsets(args.inf)
