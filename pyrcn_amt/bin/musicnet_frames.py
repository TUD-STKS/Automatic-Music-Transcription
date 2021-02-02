import numpy as np
from scipy.spatial.distance import cosine
import argparse
import os
import librosa
from madmom.processors import SequentialProcessor, ParallelProcessor
from madmom.audio import SignalProcessor, FramedSignalProcessor
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.audio.filters import LogarithmicFilterbank
from madmom.audio.spectrogram import FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor, \
    SpectrogramDifferenceProcessor

from shutil import copyfile
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, log_loss
from joblib import dump, load, Parallel, delayed

from pyrcn.echo_state_network import ESNRegressor
from pyrcn.base import InputToNode, NodeToNode
from pyrcn.linear_model import IncrementalRegression, FastIncrementalRegression

from pyrcn_amt.datasets.musicnet import MusicNET
from pyrcn_amt.config.parse_configuration import parse_configuration
from pyrcn_amt.feature_extraction.feature_extractor import FeatureExtractor
from pyrcn_amt.evaluation.multipitch_scoring import determine_threshold
from pyrcn_amt.post_processing.binarize_output import thresholding
from pyrcn_amt.evaluation.multipitch_scoring import get_mir_eval_rows


def create_feature_extraction_pipeline(sr=44100, frame_sizes=[1024, 2048, 4096], fps_hz=100.):
    audio_loading = Pipeline([("load_audio", FeatureExtractor(librosa.load, sr=sr, mono=True)),
                              ("normalize", FeatureExtractor(librosa.util.normalize, norm=np.inf))])

    sig = SignalProcessor(num_channels=1, sample_rate=sr)
    multi = ParallelProcessor([])
    for frame_size in frame_sizes:
        frames = FramedSignalProcessor(frame_size=frame_size, fps=fps_hz)
        stft = ShortTimeFourierTransformProcessor()  # caching FFT window
        filt = FilteredSpectrogramProcessor(filterbank=LogarithmicFilterbank, num_bands=12, fmin=30, fmax=17000,
                                            norm_filters=True, unique_filters=True)
        spec = LogarithmicSpectrogramProcessor(log=np.log10, mul=5, add=1)
        diff = SpectrogramDifferenceProcessor(diff_ratio=0.5, positive_diffs=True, stack_diffs=np.hstack)
        # process each frame size with spec and diff sequentially
        multi.append(SequentialProcessor([frames, stft, filt, spec, diff]))
    feature_extractor = FeatureExtractor(SequentialProcessor([sig, multi, np.hstack]))

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
    base_esn = ESNRegressor(input_to_nodes=[('default', base_input_to_node)],
                            nodes_to_nodes=[('default', base_node_to_node)],
                            regressor=base_regressor)
    fit_params = {**input_to_node_settings, **node_to_node_settings, **regression_settings}
    return base_esn, fit_params


def train_musicnet_frames(config_file):
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
        feature_extraction_pipeline = \
            create_feature_extraction_pipeline(sr=44100, frame_sizes=[1024, 2048, 4096], fps_hz=100)
        dump(feature_extraction_pipeline,
             os.path.join(experiment_settings['out_folder'], 'models', 'feature_extraction_pipeline.joblib'))

    base_esn, fit_params = create_base_esn(input_to_node_settings, node_to_node_settings, regression_settings)

    corpus = MusicNET(audio_dir=r"Z:\Projekt-Musik-Datenbank\musicNET",
                      label_dir=r"Z:\Projekt-Musik-Datenbank\musicNET")
    training_files = corpus.get_utterances(fold="train")
    test_files = corpus.get_utterances(fold="test")

    losses = Parallel(n_jobs=1)(delayed(opt_function)(base_esn, params, feature_extraction_pipeline, corpus,
                                                      training_files, test_files, experiment_settings)
                                for params in ParameterGrid(fit_params))
    dump(losses, filename=os.path.join(experiment_settings["out_folder"], 'losses.lst'))


def validate_musicnet_frames(config_file):
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
        feature_extraction_pipeline = \
            create_feature_extraction_pipeline(sr=44100, frame_sizes=[1024, 2048, 4096], fps_hz=100)
        dump(feature_extraction_pipeline,
             os.path.join(experiment_settings['out_folder'], 'models', 'feature_extraction_pipeline.joblib'))

    base_esn, fit_params = create_base_esn(input_to_node_settings, node_to_node_settings, regression_settings)

    corpus = MusicNET(audio_dir=r"Z:\Projekt-Musik-Datenbank\musicNET",
                      label_dir=r"Z:\Projekt-Musik-Datenbank\musicNET")
    training_files = corpus.get_utterances(fold="train")
    test_files = corpus.get_utterances(fold="test")

    scores = Parallel(n_jobs=1)(delayed(score_function)(base_esn, params, feature_extraction_pipeline, corpus,
                                                        training_files, test_files, experiment_settings)
                                for params in ParameterGrid(fit_params))
    dump(scores, filename=os.path.join(experiment_settings["out_folder"], 'scores.lst'))


def test_musicnet_frames(config_file, in_file, out_file):
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
        feature_extraction_pipeline = \
            create_feature_extraction_pipeline(sr=44100, frame_sizes=[1024, 2048, 4096], fps_hz=100)
        dump(feature_extraction_pipeline,
             os.path.join(experiment_settings['out_folder'], 'models', 'feature_extraction_pipeline.joblib'))

    base_esn, fit_params = create_base_esn(input_to_node_settings, node_to_node_settings, regression_settings)
    f_name = os.path.join(experiment_settings["out_folder"], "models", "esn_200000_True.joblib")
    try:
        esn = load(f_name)
    except FileNotFoundError:
        corpus = MusicNET(audio_dir=r"Z:\Projekt-Musik-Datenbank\musicNET",
                          label_dir=r"Z:\Projekt-Musik-Datenbank\musicNET")
        training_files = corpus.get_utterances(fold="train")
        Parallel(n_jobs=-1)(delayed(train_esn)(base_esn, params, feature_extraction_pipeline, corpus, training_files,
                                               experiment_settings) for params in ParameterGrid(fit_params))
        esn = load(f_name)

    U = feature_extraction_pipeline.transform(in_file)
    y_pred = esn.predict(X=U)
    y_pred_bin = thresholding(y_pred, 0.3)
    est_time, est_freqs = get_mir_eval_rows(y=y_pred_bin)
    with open(out_file, 'w') as f:
        for t in range(len(est_time)):
            f.write('{0}\t'.format(est_time[t]))
            notes = est_freqs[t]
            for note in notes:
                f.write('{0}\t'.format(note))
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
        U = feature_extraction_pipeline.transform(corpus.get_audiofilename(fids)).T
        y = corpus.get_note_labels(corpus.get_labelfilename(fids), fps=100, fs=44100, n_frames=U.shape[0])
        esn.partial_fit(X=U, y=y, update_output_weights=False)
    U = feature_extraction_pipeline.transform(corpus.get_audiofilename(training_utterances[-1])).T
    y = corpus.get_labels(corpus.get_labelfilename(training_utterances[-1]), fps=100, fs=44100, n_frames=U.shape[0])
    esn.partial_fit(X=U, y=y, update_output_weights=True)
    serialize = False
    if serialize:
        dump(esn, os.path.join(experiment_settings["out_folder"], "models", "esn_" + str(params['reservoir_size']) + '_'
                               + str(params['bi_directional']) + '.joblib'))
    return esn


def opt_function(base_esn, params, feature_extraction_pipeline, corpus, training_utterances, test_utterances,
                 experiment_settings):
    esn = train_esn(base_esn, params, feature_extraction_pipeline, corpus, training_utterances, experiment_settings)

    #  Validation
    train_loss = []
    for fids in training_utterances:
        U = feature_extraction_pipeline.transform(corpus.get_audiofilename(fids)).T
        y_true = corpus.get_note_labels(corpus.get_labelfilename(fids), fps=100, fs=44100, n_frames=U.shape[0])
        y_true[y_true < 1] = 0
        y_pred = esn.predict(X=U)
        train_loss.append([cosine(y_true, y_pred), mean_squared_error(y_true, y_pred), log_loss(y_true, y_pred)])

    test_loss = []
    for fids in test_utterances:
        U = feature_extraction_pipeline.transform(corpus.get_audiofilename(fids)).T
        y_true = corpus.get_note_labels(corpus.get_labelfilename(fids), fps=100, fs=44100, n_frames=U.shape[0])
        y_true[y_true < 1] = 0
        y_pred = esn.predict(X=U)
        test_loss.append([cosine(y_true, y_pred), mean_squared_error(y_true, y_pred), log_loss(y_true, y_pred)])

    return [np.mean(train_loss, axis=0), np.mean(test_loss, axis=0)]


def score_function(base_esn, params, feature_extraction_pipeline, corpus, training_utterances, test_utterances,
                   experiment_settings):
    try:
        f_name = os.path.join(experiment_settings["out_folder"], "models", "esn_" + str(params["reservoir_size"]) +
                              "_" + str(params['bi_directional']) + ".joblib")
        esn = load(f_name)
    except FileNotFoundError:
        esn = train_esn(base_esn, params, feature_extraction_pipeline, corpus, training_utterances, experiment_settings)

    # Training set
    Y_pred_train = []
    Pitch_times_train = []
    for fids in training_utterances:
        U = feature_extraction_pipeline.transform(corpus.get_audiofilename(fids)).T
        pitch_labels = corpus.get_note_labels(corpus.get_labelfilename(fids), fps=100, fs=44100, n_frames=U.shape[0])
        Pitch_times_train.append(pitch_labels)
        y_pred = esn.predict(X=U)
        Y_pred_train.append(y_pred)
    train_scores = determine_threshold(Y_true=Pitch_times_train, Y_pred=Y_pred_train,
                                       threshold=np.linspace(start=0.1, stop=0.4, num=16))

    # Test set
    Y_pred_test = []
    Pitch_times_test = []
    for fids in test_utterances:
        U = feature_extraction_pipeline.transform(corpus.get_audiofilename(fids)).T
        pitch_labels = corpus.get_note_labels(corpus.get_labelfilename(fids), fps=100, fs=44100, n_frames=U.shape[0])
        Pitch_times_test.append(pitch_labels)
        y_pred = esn.predict(X=U)
        Y_pred_test.append(y_pred)
    test_scores = determine_threshold(Y_true=Pitch_times_test, Y_pred=Y_pred_test,
                                      threshold=np.linspace(start=0.1, stop=0.4, num=16))
    print(train_scores)
    print(test_scores)
    return train_scores, test_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate Echo State Network')
    parser.add_argument('-inf',  type=str)

    in_file = r"Z:\Projekt-Musik-Datenbank\musicNET\train_data\1727.wav"
    out_file = r"C:\Users\Steiner\Documents\Python\Automatic-Music-Transcription\1727.f0"
    args = parser.parse_args()
    train_musicnet_frames(args.inf)
