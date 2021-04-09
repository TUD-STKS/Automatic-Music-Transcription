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
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from joblib import dump, load, Parallel, delayed

from pyrcn.echo_state_network import ESNRegressor
from pyrcn.base import InputToNode, PredefinedWeightsInputToNode, NodeToNode
from pyrcn.linear_model import IncrementalRegression

import yaml

from pyrcn_amt.datasets.boeck_onset_dataset import BoeckOnsetCorpus
from pyrcn_amt.feature_extraction.feature_extractor import FeatureExtractor
from pyrcn_amt.post_processing.binarize_output import peak_picking
from pyrcn_amt.evaluation.onset_scoring import determine_peak_picking_threshold


def create_feature_extraction_pipeline(sr, frame_sizes, fps_hz):
    audio_loading = Pipeline([("load_audio", FeatureExtractor(librosa.load, sr=sr, mono=True)),
                              ("normalize", FeatureExtractor(librosa.util.normalize, norm=np.inf))])

    sig = SignalProcessor(num_channels=1, sample_rate=sr)
    multi = ParallelProcessor([])
    for frame_size in frame_sizes:
        frames = FramedSignalProcessor(frame_size=frame_size, fps=fps_hz)
        stft = ShortTimeFourierTransformProcessor()  # caching FFT window
        filt = FilteredSpectrogramProcessor(filterbank=LogarithmicFilterbank, num_bands=12, fmin=30, fmax=17000,
                                            norm_filters=True, unique_filters=True)
        spec = LogarithmicSpectrogramProcessor(log=np.log10, mul=1, add=1)
        diff = SpectrogramDifferenceProcessor(diff_ratio=0.25, positive_diffs=True, stack_diffs=np.hstack)
        # process each frame size with spec and diff sequentially
        multi.append(SequentialProcessor([frames, stft, filt, spec, diff]))
    feature_extractor = FeatureExtractor(SequentialProcessor([sig, multi, np.hstack]))

    feature_extraction_pipeline = Pipeline([("audio_loading", audio_loading),
                                            ("feature_extractor", feature_extractor)])
    return feature_extraction_pipeline


def train_onset_detection(config_file):
    with open(config_file, 'r') as file:
        settings = yaml.full_load(stream=file)

    experiment_settings = settings['experiment']
    param_grid = settings['param_grid']

    # Make Paths
    if not os.path.isdir(experiment_settings['out_folder']):
        os.mkdir(experiment_settings['out_folder'])
    if not os.path.isdir(os.path.join(experiment_settings['out_folder'], 'models')):
        os.mkdir(os.path.join(experiment_settings['out_folder'], 'models'))

    # replicate config file and store results there
    copyfile(config_file, os.path.join(experiment_settings['out_folder'], 'config.yaml'))

    feature_extraction_pipeline = create_feature_extraction_pipeline(sr=44100, frame_sizes=[2048], fps_hz=100)
    dump(feature_extraction_pipeline,
         os.path.join(experiment_settings['out_folder'], 'models', 'feature_extraction_pipeline.joblib'))

    base_esn = ESNRegressor(input_to_node=InputToNode(), node_to_node=NodeToNode(), regressor=IncrementalRegression())

    corpus = BoeckOnsetCorpus(audio_dir=os.path.join(experiment_settings["in_folder"], "onsets_audio"),
                              label_dir=os.path.join(experiment_settings["in_folder"], "onsets_annotations"),
                              split_dir=os.path.join(experiment_settings["in_folder"], "onsets_splits"))

    losses = []
    for k in range(8):
        if k == 0:
            training_files = np.concatenate(([corpus.get_utterances(fold=fold_id) for fold_id in [0, 1, 2, 3, 4, 5]]))
            validation_files = corpus.get_utterances(fold=6)
        elif k == 1:
            training_files = np.concatenate(([corpus.get_utterances(fold=fold_id) for fold_id in [1, 2, 3, 4, 5, 6]]))
            validation_files = corpus.get_utterances(fold=7)
        elif k == 2:
            training_files = np.concatenate(([corpus.get_utterances(fold=fold_id) for fold_id in [2, 3, 4, 5, 6, 7]]))
            validation_files = corpus.get_utterances(fold=0)
        elif k == 3:
            training_files = np.concatenate(([corpus.get_utterances(fold=fold_id) for fold_id in [3, 4, 5, 6, 7, 0]]))
            validation_files = corpus.get_utterances(fold=1)
        elif k == 4:
            training_files = np.concatenate(([corpus.get_utterances(fold=fold_id) for fold_id in [4, 5, 6, 7, 0, 1]]))
            validation_files = corpus.get_utterances(fold=2)
        elif k == 5:
            training_files = np.concatenate(([corpus.get_utterances(fold=fold_id) for fold_id in [5, 6, 7, 0, 1, 2]]))
            validation_files = corpus.get_utterances(fold=3)
        elif k == 6:
            training_files = np.concatenate(([corpus.get_utterances(fold=fold_id) for fold_id in [6, 7, 0, 1, 2, 3]]))
            validation_files = corpus.get_utterances(fold=4)
        elif k == 7:
            training_files = np.concatenate(([corpus.get_utterances(fold=fold_id) for fold_id in [7, 0, 1, 2, 3, 4]]))
            validation_files = corpus.get_utterances(fold=5)
        else:
            raise ValueError("maximum number of folds is 8. Currently, k is {0}".format(k))
        tmp_losses = Parallel(n_jobs=1)(delayed(opt_function)(base_esn, params, feature_extraction_pipeline, corpus,
                                                              training_files, validation_files, experiment_settings)
                                        for params in ParameterGrid(param_grid=param_grid))
        losses.append(tmp_losses)
    dump(losses, filename=os.path.join(experiment_settings["out_folder"], 'losses.lst'))


def validate_onset_detection(config_file):
    with open(config_file, 'r') as file:
        settings = yaml.full_load(stream=file)

    experiment_settings = settings['experiment']
    param_grid = settings['param_grid']

    # Make Paths
    if not os.path.isdir(experiment_settings['out_folder']):
        os.mkdir(experiment_settings['out_folder'])
    if not os.path.isdir(os.path.join(experiment_settings['out_folder'], 'models')):
        os.mkdir(os.path.join(experiment_settings['out_folder'], 'models'))

    # replicate config file and store results there
    copyfile(config_file, os.path.join(experiment_settings['out_folder'], 'config.yaml'))

    feature_extraction_pipeline = create_feature_extraction_pipeline(sr=44100, frame_sizes=[1024, 2048, 4096],
                                                                     fps_hz=100)
    dump(feature_extraction_pipeline,
         os.path.join(experiment_settings['out_folder'], 'models', 'feature_extraction_pipeline.joblib'))

    base_esn = ESNRegressor(input_to_node=InputToNode(), node_to_node=NodeToNode(), regressor=IncrementalRegression())

    corpus = BoeckOnsetCorpus(audio_dir=os.path.join(experiment_settings["in_folder"], "onsets_audio"),
                              label_dir=os.path.join(experiment_settings["in_folder"], "onsets_annotations"),
                              split_dir=os.path.join(experiment_settings["in_folder"], "onsets_splits"))

    scores = []
    for params in ParameterGrid(param_grid=param_grid):
        tmp_scores = Parallel(n_jobs=1)(delayed(score_function)(base_esn, params, feature_extraction_pipeline, corpus,
                                                                k, experiment_settings)
                                        for k in range(8))
        scores.append(tmp_scores)
    dump(scores, filename=os.path.join(experiment_settings["out_folder"], 'scores_8000_1_bi.lst'))


def test_onset_detection(config_file, in_file, out_file):
    with open(config_file, 'r') as file:
        settings = yaml.full_load(stream=file)

    experiment_settings = settings['experiment']
    param_grid = settings['param_grid']

    # Make Paths
    if not os.path.isdir(experiment_settings['out_folder']):
        os.mkdir(experiment_settings['out_folder'])
    if not os.path.isdir(os.path.join(experiment_settings['out_folder'], 'models')):
        os.mkdir(os.path.join(experiment_settings['out_folder'], 'models'))

    # replicate config file and store results there
    copyfile(config_file, os.path.join(experiment_settings['out_folder'], 'config.yaml'))

    try:
        feature_extraction_pipeline = load(
            os.path.join(experiment_settings['out_folder'], 'models', 'feature_extraction_pipeline.joblib'))
    except FileNotFoundError:
        feature_extraction_pipeline = \
            create_feature_extraction_pipeline(sr=44100, frame_sizes=[1024, 2048, 4096], fps_hz=100)
        dump(feature_extraction_pipeline,
             os.path.join(experiment_settings['out_folder'], 'models', 'feature_extraction_pipeline.joblib'))

    base_esn = ESNRegressor(input_to_node=InputToNode(), node_to_node=NodeToNode(), regressor=IncrementalRegression())
    base_esn.set_params(**param_grid)
    f_name = os.path.join(experiment_settings["out_folder"], "models", "esn_200000_True.joblib")
    try:
        esn = load(f_name)
    except FileNotFoundError:
        corpus = BoeckOnsetCorpus(audio_dir=os.path.join(experiment_settings["in_folder"], "onsets_audio"),
                                  label_dir=os.path.join(experiment_settings["in_folder"], "onsets_annotations"),
                                  split_dir=os.path.join(experiment_settings["in_folder"], "onsets_splits"))
        training_files = np.concatenate(([corpus.get_utterances(fold=fold_id) for fold_id in [0, 1, 2, 3, 4, 5, 6, 7]]))
        Parallel(n_jobs=1)(delayed(train_esn)(base_esn, params, feature_extraction_pipeline, corpus, training_files,
                                              experiment_settings)
                           for params in ParameterGrid(param_grid=param_grid))
        esn = load(f_name)

    U = feature_extraction_pipeline.transform(in_file)
    y_pred = esn.predict(X=U)
    onset_times_res = peak_picking(y_pred, 0.4)
    with open(out_file, 'w') as f:
        for onset_time in onset_times_res:
            f.write('{0}'.format(onset_time))
            f.write('\n')


def train_kmeans(feature_extraction_pipeline, corpus, training_utterances, hidden_layer_size=50):
    unlabeled_utterances = corpus.get_unlabeled_utterances()
    X = []
    for fids in training_utterances:
        U = feature_extraction_pipeline.transform(corpus.get_audiofilename(fids)).T
        X.append(U)
    """
    for fids in unlabeled_utterances:
        U = feature_extraction_pipeline.transform(corpus.get_audiofilename(fids)).T
        X.append(U)
    """
    if hidden_layer_size is not None:
        kmeans = MiniBatchKMeans(n_clusters=hidden_layer_size, n_init=20,  reassignment_ratio=0, max_no_improvement=50,
                                 init='k-means++', verbose=0, random_state=1)
    else:
        kmeans = MiniBatchKMeans(n_clusters=50, n_init=20, reassignment_ratio=0, max_no_improvement=50,
                                 init='k-means++', verbose=0, random_state=1)
    kmeans.fit(X=np.concatenate(X))
    return kmeans


def train_esn(base_esn, params, feature_extraction_pipeline, corpus, training_utterances, experiment_settings):
    print(params)
    kmeans = train_kmeans(feature_extraction_pipeline, corpus, training_utterances,
                          params["node_to_node__hidden_layer_size"])
    esn = clone(base_esn)
    esn.input_to_node = PredefinedWeightsInputToNode(
        predefined_input_weights=np.divide(kmeans.cluster_centers_,
                                           np.linalg.norm(kmeans.cluster_centers_, axis=1)[:, None]).T
    )
    esn.set_params(**params)

    for fids in training_utterances[:-1]:
        U = feature_extraction_pipeline.transform(corpus.get_audiofilename(fids)).T
        y = corpus.get_labels(corpus.get_labelfilename(fids), fps=100, n_frames=U.shape[0])
        esn.partial_fit(X=U, y=y, postpone_inverse=True)
    U = feature_extraction_pipeline.transform(corpus.get_audiofilename(training_utterances[-1])).T
    y = corpus.get_labels(corpus.get_labelfilename(training_utterances[-1]), fps=100, n_frames=U.shape[0])
    esn.partial_fit(X=U, y=y, postpone_inverse=False)

    serialize = False
    if serialize:
        dump(esn, os.path.join(experiment_settings["out_folder"],
                               "models",
                               "esn_" + str(esn.input_to_node.hidden_layer_size) +
                               '_' + str(esn.node_to_node.bi_directional) + '.joblib'))
    return esn


def opt_function(base_esn, params, feature_extraction_pipeline, corpus, training_utterances, validation_utterances,
                 experiment_settings):
    esn = train_esn(base_esn, params, feature_extraction_pipeline, corpus, training_utterances, experiment_settings)

    #  Validation
    train_loss = []
    for fids in training_utterances:
        U = feature_extraction_pipeline.transform(corpus.get_audiofilename(fids)).T
        y_true = corpus.get_labels(corpus.get_labelfilename(fids), fps=100, n_frames=U.shape[0])
        y_true[y_true < 1] = 0
        y_pred = esn.predict(X=U)
        train_loss.append([cosine(y_true, y_pred), mean_squared_error(y_true, y_pred)])

    val_loss = []
    for fids in validation_utterances:
        U = feature_extraction_pipeline.transform(corpus.get_audiofilename(fids)).T
        y_true = corpus.get_labels(corpus.get_labelfilename(fids), fps=100, n_frames=U.shape[0])
        y_true[y_true < 1] = 0
        y_pred = esn.predict(X=U)
        val_loss.append([cosine(y_true, y_pred), mean_squared_error(y_true, y_pred)])

    return [np.mean(train_loss, axis=0), np.mean(val_loss, axis=0)]


def score_function(base_esn, params, feature_extraction_pipeline, corpus, k, experiment_settings):
    if k == 0:
        training_files = np.concatenate(([corpus.get_utterances(fold=fold_id) for fold_id in [0, 1, 2, 3, 4, 5, 6]]))
        validation_files = corpus.get_utterances(fold=7)
    elif k == 1:
        training_files = np.concatenate(([corpus.get_utterances(fold=fold_id) for fold_id in [1, 2, 3, 4, 5, 6, 7]]))
        validation_files = corpus.get_utterances(fold=0)
    elif k == 2:
        training_files = np.concatenate(([corpus.get_utterances(fold=fold_id) for fold_id in [2, 3, 4, 5, 6, 7, 0]]))
        validation_files = corpus.get_utterances(fold=1)
    elif k == 3:
        training_files = np.concatenate(([corpus.get_utterances(fold=fold_id) for fold_id in [3, 4, 5, 6, 7, 0, 1]]))
        validation_files = corpus.get_utterances(fold=2)
    elif k == 4:
        training_files = np.concatenate(([corpus.get_utterances(fold=fold_id) for fold_id in [4, 5, 6, 7, 0, 1, 2]]))
        validation_files = corpus.get_utterances(fold=3)
    elif k == 5:
        training_files = np.concatenate(([corpus.get_utterances(fold=fold_id) for fold_id in [5, 6, 7, 0, 1, 2, 3]]))
        validation_files = corpus.get_utterances(fold=4)
    elif k == 6:
        training_files = np.concatenate(([corpus.get_utterances(fold=fold_id) for fold_id in [6, 7, 0, 1, 2, 3, 4]]))
        validation_files = corpus.get_utterances(fold=5)
    elif k == 7:
        training_files = np.concatenate(([corpus.get_utterances(fold=fold_id) for fold_id in [7, 0, 1, 2, 3, 4, 5]]))
        validation_files = corpus.get_utterances(fold=6)
    else:
        raise ValueError("maximum number of folds is 8. Currently, k is {0}".format(k))

    try:
        f_name = os.path.join(experiment_settings["out_folder"],
                              "models",
                              "esn_" + str(params["node_to_node__hidden_layer_size"]) +
                              "_" + str(params['node_to_node__hidden_layer_size']) + ".joblib")
        esn = load(f_name)
    except FileNotFoundError:
        esn = train_esn(base_esn, params, feature_extraction_pipeline, corpus, training_files, experiment_settings)

    # Training set
    Y_pred_train = []
    Onset_times_train = []
    for fids in training_files:
        U = feature_extraction_pipeline.transform(corpus.get_audiofilename(fids)).T
        y_true = corpus.get_onset_events(corpus.get_labelfilename(fids))
        Onset_times_train.append(y_true)
        y_pred = esn.predict(X=U)
        Y_pred_train.append(y_pred)
    train_scores = determine_peak_picking_threshold(odf=Y_pred_train,
                                                    threshold=np.linspace(start=0.2, stop=0.5, num=16),
                                                    Onset_times_ref=Onset_times_train)

    # Test set
    Y_pred_val = []
    Onset_times_val = []
    for fids in validation_files:
        U = feature_extraction_pipeline.transform(corpus.get_audiofilename(fids)).T
        y_true = corpus.get_onset_events(corpus.get_labelfilename(fids))
        Onset_times_val.append(y_true)
        y_pred = esn.predict(X=U)
        Y_pred_val.append(y_pred)
    val_scores = determine_peak_picking_threshold(odf=Y_pred_val,
                                                  threshold=np.linspace(start=0.2, stop=0.5, num=16),
                                                  Onset_times_ref=Onset_times_val)

    return train_scores, val_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate Echo State Network')
    parser.add_argument('-inf',  type=str)
    input_file = \
        r"Z:\Projekt-Musik-Datenbank\OnsetDetektion\onsets_audio\ah_development_percussion_bongo1.flac"
    output_file = \
        r"C:\Users\Steiner\Documents\Python\Automatic-Music-Transcription\ah_development_percussion_bongo1.onsets"
    args = parser.parse_args()
    train_onset_detection(args.inf)
