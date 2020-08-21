import numpy as np
import argparse
import os
import warnings
from shutil import copyfile
from sklearn.model_selection import ParameterGrid
from sklearn.base import clone
from joblib import dump, load, Parallel, delayed
from sklearn.preprocessing import MultiLabelBinarizer

from pyrcn.echo_state_network import ESNRegressor
from pyrcn_amt.datasets import music_prediction_datasets
from pyrcn_amt.evaluation import loss_functions
from pyrcn_amt.config.parse_config_file import parse_config_file
from pyrcn_amt.evaluation.multipitch_scoring import determine_prediction_threshold

from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = 'jet'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


def validate_music_prediction(config_file):
    io_params, esn_params, fit_params, feature_settings, loss_fn, n_jobs = parse_config_file(config_file)
    base_esn = ESNRegressor()
    base_esn.set_params(**esn_params)

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
    copyfile(config_file, os.path.join(out_folder, 'config.ini'))

    (training_set, validation_set), _ = music_prediction_datasets.load_piano_midi_dataset(dataset_path=in_folder, validation=True)
    training_set = music_prediction_datasets.convert_to_short_sequences(training_set)
    mlb = MultiLabelBinarizer(classes=range(128))
    training_set = [mlb.fit_transform(training_set[k]) for k in range(len(training_set))]
    validation_set = [mlb.fit_transform(validation_set[k]) for k in range(len(validation_set))]

    scores = Parallel(n_jobs=n_jobs)(delayed(score_function)(base_esn, params, training_set, validation_set, out_folder) for params in ParameterGrid(fit_params))
    dump(scores, filename=os.path.join(out_folder, 'scores.lst'))


def test_music_prediction(config_file):
    io_params, esn_params, fit_params, feature_settings, loss_fn, n_jobs = parse_config_file(config_file)
    base_esn = ESNRegressor()
    base_esn.set_params(**esn_params)

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
    copyfile(config_file, os.path.join(out_folder, 'config.ini'))

    training_set, test_set = music_prediction_datasets.load_piano_midi_dataset(dataset_path=in_folder, validation=False)

    mlb = MultiLabelBinarizer(classes=range(128))
    training_set = [mlb.fit_transform(training_set[k]) for k in range(len(training_set))]
    test_set = [mlb.fit_transform(test_set[k]) for k in range(len(test_set))]

    scores = Parallel(n_jobs=n_jobs)(delayed(score_function)(base_esn, params, training_set, test_set, out_folder) for params in ParameterGrid(fit_params))
    dump(scores, filename=os.path.join(out_folder, 'scores.lst'))


def train_music_prediction(config_file):
    io_params, esn_params, fit_params, feature_settings, loss_fn, n_jobs = parse_config_file(config_file)
    base_esn = ESNRegressor()
    base_esn.set_params(**esn_params)

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
    copyfile(config_file, os.path.join(out_folder, 'config.ini'))

    (training_set, validation_set), _ = music_prediction_datasets.load_piano_midi_dataset(dataset_path=in_folder, validation=True)
    mlb = MultiLabelBinarizer(classes=range(128))
    mlb.fit_transform(y=training_set[0])
    training_set = [mlb.fit_transform(training_set[k]) for k in range(len(training_set))]
    validation_set = [mlb.fit_transform(validation_set[k]) for k in range(len(validation_set))]

    losses = Parallel(n_jobs=n_jobs)(delayed(opt_function)(base_esn, params, training_set, validation_set, loss_function, out_folder) for params in ParameterGrid(fit_params))
    dump(losses, filename=os.path.join(out_folder, 'losses.lst'))


def opt_function(base_esn, params, training_set, validation_set, loss_function, out_folder):
    esn = train_esn(base_esn, params, training_set, out_folder)

    #  Validation
    train_loss = []
    for X in training_set:
        y_pred = esn.predict(X=X[:-1, :], keep_reservoir_state=False)
        if isinstance(loss_function, list):
            train_loss.append([loss(X[1:, :], y_pred) for loss in loss_function])
        else:
            train_loss.append([loss_function(X[1:, :], y_pred)])

    val_loss = []
    for X in validation_set:
        y_pred = esn.predict(X=X[:-1, :], keep_reservoir_state=False)
        if isinstance(loss_function, list):
            val_loss.append([loss(X[1:, :], y_pred) for loss in loss_function])
        else:
            val_loss.append([loss_function(X[1:, :], y_pred)])

    return [np.mean(train_loss, axis=0), np.mean(val_loss, axis=0)]


def score_function(base_esn, params, training_set, test_set, out_folder):
    try:
        f_name = os.path.join(out_folder, "models", "esn_" + str(params["reservoir_size"]) + "_" + str(params['bi_directional']) + ".joblib")
        esn = load(f_name)
    except FileNotFoundError:
        esn = train_esn(base_esn, params, training_set, out_folder)

    # Training set
    Y_pred_train = []
    Pitch_times_train = []
    for X in training_set:
        y_pred = esn.predict(X=X[:-1, :], keep_reservoir_state=False)[:, 21:109]
        Pitch_times_train.append(X[1:, 21:109])
        Y_pred_train.append(y_pred)
    train_scores = determine_prediction_threshold(Y_true=Pitch_times_train, Y_pred=Y_pred_train, threshold=np.linspace(start=0.1, stop=1.0, num=10))

    # Test set
    Y_pred_test = []
    Pitch_times_test = []
    for X in test_set:
        y_pred = esn.predict(X=X[:-1, :], keep_reservoir_state=False)[:, 21:109]
        Pitch_times_test.append(X[1:, 21:109])
        Y_pred_test.append(y_pred)
    test_scores = determine_prediction_threshold(Y_true=Pitch_times_test, Y_pred=Y_pred_test, threshold=np.linspace(start=0.1, stop=1.0, num=10))
    X = test_set[0]
    k = 0
    while X.shape[0] <= 500:
        y_pred = esn.predict(X=X[:-1, :])
        y_pred = np.asarray(y_pred > 0.2, dtype=int)
        X = np.vstack((X, y_pred[-1, :]))
        k = k + 1
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(12, 6))
    ax0.imshow(Pitch_times_test[0].T, origin='lower', aspect='auto', cmap=plt.cm.binary)
    ax0.set_xlabel('n')
    ax0.set_ylabel('Y_true[n]')
    ax0.grid()
    ax1.imshow(Y_pred_test[0].T, origin='lower', aspect='auto', cmap=plt.cm.binary)
    ax1.set_xlabel('n')
    ax1.set_ylabel('Y_pred[n]')
    ax1.grid()
    ax2.imshow(np.power(Pitch_times_test[0].T - Y_pred_test[0].T, 2), origin='lower', aspect='auto')
    ax2.set_xlabel('n')
    ax2.set_ylabel('Y_pred[n]')
    ax2.grid()
    plt.tight_layout()
    plt.show()

    return train_scores, test_scores


def train_esn(base_esn, params, training_set, out_folder):
    print(params)
    esn = clone(base_esn)
    esn.set_params(**params)
    for X in training_set:
        esn.partial_fit(X=X[:-1, :], y=X[1:, :], update_output_weights=False)
    esn.finalize()
    serialize = False
    if serialize:
        dump(esn, os.path.join(out_folder, "models", "esn_" + str(params['reservoir_size']) + '_' + str(params['bi_directional']) + '.joblib'))
    return esn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate Echo State Network')
    parser.add_argument('-inf',  type=str)

    args = parser.parse_args()
    validate_music_prediction(args.inf)
