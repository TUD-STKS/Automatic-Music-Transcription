import numpy as np
import scipy
from scipy.spatial.distance import cosine
import argparse
import os
import librosa

from shutil import copyfile
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, log_loss, zero_one_loss
from joblib import dump, load, Parallel, delayed

from pyrcn.echo_state_network import ESNClassifier
from pyrcn.base import InputToNode, NodeToNode
from pyrcn.linear_model import IncrementalRegression, FastIncrementalRegression

from pyrcn_amt.datasets.timit_dataset import TIMITCorpus
from pyrcn_amt.config.parse_configuration import parse_configuration
from pyrcn_amt.feature_extraction.feature_extractor import FeatureExtractor


def create_feature_extraction_pipeline(sr, frame_size, fps_hz):
    audio_loading = Pipeline([("load_audio", FeatureExtractor(librosa.load, sr=sr, mono=True)),
                              ("pre_emphasis", FeatureExtractor(librosa.effects.preemphasis, coef=0.97)),
                              ("normalize", FeatureExtractor(librosa.util.normalize, norm=np.inf))
                               ])

    mfcc = ("mfcc",
            FeatureExtractor(librosa.feature.mfcc, sr=sr, n_fft=512, win_length=frame_size, hop_length=int(sr/fps_hz),
                             window=scipy.signal.windows.hamming, n_mels=24, n_mfcc=13)
            )
    delta = Pipeline(
        steps=[
            mfcc,
            ("delta", FeatureExtractor(librosa.feature.delta, width=3, order=1))
        ])
    delta_delta = Pipeline(
        steps=[
            mfcc,
            ("delta_delta", FeatureExtractor(librosa.feature.delta, width=3, order=2))
        ]
    )

    feature_extractor = FeatureUnion(
        transformer_list=[mfcc,
                          ("delta", delta),
                          ("delta_delta", delta_delta)
                          ])

    feature_extraction_pipeline = Pipeline([("audio_loading", audio_loading),
                                            ("feature_extractor", feature_extractor)])
    return feature_extraction_pipeline


def create_base_esn(input_to_node_settings, node_to_node_settings, regression_settings):
    base_input_to_node = InputToNode(random_state=eval(input_to_node_settings.pop("random_state")))
    remove = []
    for key, value in input_to_node_settings.items():
        if isinstance(eval(value), str):
            base_input_to_node.set_params(**{key: eval(value)})
            remove.append(key)
        if not hasattr(eval(value), "__iter__"):
            base_input_to_node.set_params(**{key: eval(value)})
            remove.append(key)
        else:
            input_to_node_settings[key] = eval(value)
    for key in remove:
        del input_to_node_settings[key]
    base_node_to_node = NodeToNode(random_state=eval(node_to_node_settings.pop("random_state")))
    remove = []
    for key, value in node_to_node_settings.items():
        if isinstance(eval(value), str):
            base_node_to_node.set_params(**{key: eval(value)})
            remove.append(key)
        if not hasattr(eval(value), "__iter__"):
            base_node_to_node.set_params(**{key: eval(value)})
            remove.append(key)
        else:
            node_to_node_settings[key] = eval(value)
    for key in remove:
        del node_to_node_settings[key]
    base_regressor = eval(regression_settings['regressor'] + '()')
    del regression_settings['regressor']
    if not hasattr(eval(regression_settings['alpha']), "__iter__"):
        base_regressor.set_params(**{'alpha': eval(regression_settings['alpha'])})
        del regression_settings['alpha']
    else:
        regression_settings['alpha'] = eval(regression_settings['alpha'])
    base_esn = ESNClassifier(input_to_nodes=[('default', base_input_to_node)],
                             nodes_to_nodes=[('default', base_node_to_node)],
                             regressor=base_regressor)
    fit_params = {**input_to_node_settings, **node_to_node_settings, **regression_settings}
    return base_esn, fit_params


def train_phoneme_recognition(config_file):
    experiment_settings, input_to_node_settings, node_to_node_settings, regression_settings = parse_configuration(
        config_file=config_file)

    # Make Paths
    if not os.path.isdir(experiment_settings['out_folder']):
        os.mkdir(experiment_settings['out_folder'])
    if not os.path.isdir(os.path.join(experiment_settings['out_folder'], 'models')):
        os.mkdir(os.path.join(experiment_settings['out_folder'], 'models'))

    # replicate config file and store results there
    copyfile(config_file, os.path.join(experiment_settings['out_folder'], 'config.ini'))

    try:
        feature_extraction_pipeline = load(
            os.path.join(experiment_settings['out_folder'], 'models', 'feature_extraction_pipeline.joblib'))
    except FileNotFoundError:
        feature_extraction_pipeline = create_feature_extraction_pipeline(sr=16000, frame_size=440, fps_hz=100)
        dump(feature_extraction_pipeline,
             os.path.join(experiment_settings['out_folder'], 'models', 'feature_extraction_pipeline.joblib'))

    corpus = TIMITCorpus(transcription_dir=experiment_settings['in_folder'], audio_dir=experiment_settings['in_folder'])

    try:
        scaler = load(os.path.join(experiment_settings['out_folder'], 'models', 'scaler.joblib'))
    except FileNotFoundError:
        training_utterances = corpus.get_utterances(utttype="train")
        scaler = StandardScaler()
        for utterance in training_utterances:
            if not utterance.split("\\")[-1].startswith("sa"):
                X = feature_extraction_pipeline.transform(corpus.get_audiofilename(utterance=utterance))
                scaler.partial_fit(X=X, y=None)
        dump(scaler, os.path.join(experiment_settings['out_folder'], 'models', 'scaler.joblib'))

    feature_extraction_pipeline.steps.append(("standard_scaler", scaler))

    base_esn, fit_params = create_base_esn(input_to_node_settings, node_to_node_settings, regression_settings)

    training_files = corpus.get_utterances("train")
    test_files = corpus.get_utterances("test")
    losses = Parallel(n_jobs=1)(delayed(opt_function)(base_esn, params, feature_extraction_pipeline, corpus,
                                                      training_files, test_files, experiment_settings)
                                for params in ParameterGrid(fit_params))
    dump(losses, filename=os.path.join(experiment_settings["out_folder"], 'losses.lst'))


def validate_phoneme_recognition(config_file):
    io_params, esn_params, fit_params, feature_settings, loss_fn, n_jobs = parse_config_file(config_file)
    base_esn = ESNClassifier()
    base_esn.set_params(**esn_params)
    feature_settings = parse_feature_settings(feature_settings)

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

    training_set, test_set = timit_dataset.load_dataset(dataset_path=in_folder, validation=False)
    scores = Parallel(n_jobs=n_jobs)(delayed(score_function)(base_esn, params, feature_settings, training_set, test_set, out_folder) for params in ParameterGrid(fit_params))
    dump(scores, filename=os.path.join(out_folder, 'scores.lst'))


def test_phoneme_recognition(config_file, in_file, out_file):
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
    copyfile(config_file, os.path.join(out_folder, 'config.ini'))
    try:
        f_name = r"C:\Users\Steiner\Documents\Python\Automatic-Music-Transcription\pyrcn_amt\experiments\experiment_0\models\esn_200000_True.joblib"
        esn = load(f_name)
    except FileNotFoundError:
        training_set, test_set = timit_dataset.load_dataset(dataset_path=in_folder, validation=False)
        Parallel(n_jobs=n_jobs)(delayed(train_esn)(base_esn, params, feature_settings, pre_processor, scaler, training_set + test_set, out_folder) for params in ParameterGrid(fit_params))
        esn = load(f_name)

    s = load_sound_file(file_name=in_file, feature_settings=feature_settings)
    U = mfcc(signal=s, samplerate=feature_settings['fs'],
             winlen=feature_settings['frame_size'] / feature_settings['fs'],
             winstep=1 / feature_settings['fps'], nfft=512, appendEnergy=True)
    delta_U = delta(U, 2)
    U = np.hstack((U, delta_U))
    # delta_delta_U = delta(delta_U, 2)
    # U = np.hstack((U, delta_delta_U))
    y_pred = esn.predict(X=U, keep_reservoir_state=False)
    onset_times_res = peak_picking(y_pred, 0.4)
    with open(out_file, 'w') as f:
        for onset_time in onset_times_res:
            f.write('{0}'.format(onset_time))
            f.write('\n')


def train_esn(base_esn, params, feature_extraction_pipeline, corpus, training_utterances, experiment_settings):
    print(params)
    esn = base_esn
    if "input_scaling" in params:
        esn.input_to_nodes[0][1].set_params(**{"input_scaling": params["input_scaling"]})
        del params["input_scaling"]
    if "hidden_layer_size" in params:
        esn.input_to_nodes[0][1].set_params(**{"hidden_layer_size": params["hidden_layer_size"]})
        esn.nodes_to_nodes[0][1].set_params(**{"hidden_layer_size": params["hidden_layer_size"]})
        del params["hidden_layer_size"]
    if "alpha" in params:
        esn.regressor.set_params(**{"alpha": params["alpha"]})
        del params["alpha"]
    if params:
        esn.nodes_to_nodes[0][1].set_params(**params)
    for fids in training_utterances[:-1]:
        if not fids[:-1].endswith("sa"):
            U = feature_extraction_pipeline.transform(corpus.get_audiofilename(fids))
            y = corpus.get_phoneme_labels(corpus.get_labelfilename(fids), fps=100, fs=16000, n_frames=U.shape[0])
            esn.partial_fit(X=U, y=y, update_output_weights=False, classes=np.arange(39))
    U = feature_extraction_pipeline.transform(corpus.get_audiofilename(training_utterances[-1]))
    y = corpus.get_phoneme_labels(corpus.get_labelfilename(training_utterances[-1]), fps=100, fs=16000, n_frames=U.shape[0])
    esn.partial_fit(X=U, y=y, update_output_weights=True)
    serialize = False
    if serialize:
        dump(esn, os.path.join(experiment_settings["out_folder"], "models", "esn_" + str(params['reservoir_size']) + '_'
                               + str(params['bi_directional']) + '.joblib'))
    return esn


def opt_function(base_esn, params, feature_extraction_pipeline, corpus, training_files, test_files,
                 experiment_settings):
    esn = train_esn(base_esn, params, feature_extraction_pipeline, corpus, training_files, experiment_settings)

    #  Validation
    train_loss = []
    for fids in training_files[:-1]:
        if not fids[:-1].endswith("sa"):
            U = feature_extraction_pipeline.transform(corpus.get_audiofilename(fids))
            y_true = corpus.get_phoneme_labels(corpus.get_labelfilename(fids), fps=100, fs=16000, n_frames=U.shape[0])
            y_true_proba = np.zeros(shape=(y_true.shape[0], 39))
            for k, y in enumerate(y_true):
                y_true_proba[k, y] = 1
            y_pred = esn.predict(X=U)
            y_pred_proba = esn.predict_proba(X=U)
            train_loss.append([cosine(y_true_proba.ravel(), y_pred_proba.ravel()),
                               mean_squared_error(y_true_proba, y_pred_proba),
                               log_loss(y_true, y_pred_proba, labels=np.arange(39)),
                               zero_one_loss(y_true, y_pred)])

    test_loss = []
    for fids in test_files[:-1]:
        if not fids[:-1].endswith("sa"):
            U = feature_extraction_pipeline.transform(corpus.get_audiofilename(fids))
            y_true = corpus.get_phoneme_labels(corpus.get_labelfilename(fids), fps=100, fs=16000, n_frames=U.shape[0])
            y_true_proba = np.zeros(shape=(y_true.shape[0], 39))
            for k, y in enumerate(y_true):
                y_true_proba[k, y] = 1
            y_pred = esn.predict(X=U)
            y_pred_proba = esn.predict_proba(X=U)
            test_loss.append([cosine(y_true_proba.ravel(), y_pred_proba.ravel()),
                              mean_squared_error(y_true_proba, y_pred_proba),
                              log_loss(y_true, y_pred_proba, labels=np.arange(39)),
                              zero_one_loss(y_true, y_pred)])

    return [np.mean(train_loss, axis=0), np.mean(test_loss, axis=0)]


def score_function(base_input_to_node, base_node_to_node, base_regressor, params, feature_pipeline, corpus,
                   experiment_settings):
    try:
        f_name = os.path.join(experiment_settings["out_folder"], "models", "esn_" + str(params["reservoir_size"]) + "_"
                              + str(params['bi_directional']) + ".joblib")
        esn = load(f_name)
    except FileNotFoundError:
        esn = train_esn(base_input_to_node, base_node_to_node, base_regressor, corpus, feature_pipeline, experiment_settings, params, False)

    train_loss = []
    for utterance in corpus.get_utterances("train"):
        if not utterance.split("\\")[-1].startswith("sa"):
            X = feature_pipeline.transform(corpus.get_audiofilename(utterance=utterance))
            start_times, stop_times, labels = corpus.get_labels(utterance)
            y = np.zeros(shape=(X.shape[0],), dtype=int)
            for start_time, stop_time, label in zip(start_times, stop_times, labels):
                y[int(start_time):int(stop_time + 1)] = corpus.phone2int[label]
            y_pred = esn.predict(X=X)
            train_loss.append(zero_one_loss(y, y_pred))

    test_loss = []
    for utterance in corpus.get_utterances("test"):
        if not utterance.split("\\")[-1].startswith("sa"):
            X = feature_pipeline.transform(corpus.get_audiofilename(utterance=utterance))
            start_times, stop_times, labels = corpus.get_labels(utterance)
            y = np.zeros(shape=(X.shape[0],), dtype=int)
            for start_time, stop_time, label in zip(start_times, stop_times, labels):
                y[int(start_time):int(stop_time + 1)] = corpus.phone2int[label]
            y_pred = esn.predict(X=X)
            test_loss.append(zero_one_loss(y, y_pred))

    return train_loss, test_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate Echo State Network')
    parser.add_argument('-inf',  type=str)
    in_file = r"Z:\Projekt-Musik-Datenbank\OnsetDetektion\onsets_audio\ah_development_percussion_bongo1.flac"
    out_file = r"C:\Users\Steiner\Documents\Python\Automatic-Music-Transcription\ah_development_percussion_bongo1.onsets"
    args = parser.parse_args()
    train_phoneme_recognition(args.inf)
