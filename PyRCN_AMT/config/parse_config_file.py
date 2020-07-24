import os
import numpy as np
import configparser


def parse_config_file(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)

    io_params = dict()
    io_params['in_folder'] = os.path.normpath(config['io']['in_folder'])
    io_params['out_folder'] = os.path.normpath(config['io']['out_folder'])

    esn_params = dict()
    fit_params = dict()
    if not (isinstance(eval(config['esn']['k_in']), np.ndarray) or isinstance(eval(config['esn']['k_in']), list)):
        esn_params['k_in'] = eval(config['esn']['k_in'])
    else:
        fit_params['k_in'] = eval(config['esn']['k_in'])
    if not (isinstance(eval(config['esn']['input_scaling']), np.ndarray) or isinstance(eval(config['esn']['input_scaling']), list)):
        esn_params['input_scaling'] = eval(config['esn']['input_scaling'])
    else:
        fit_params['input_scaling'] = eval(config['esn']['input_scaling'])
    if not (isinstance(eval(config['esn']['spectral_radius']), np.ndarray) or isinstance(eval(config['esn']['spectral_radius']), list)):
        esn_params['spectral_radius'] = eval(config['esn']['spectral_radius'])
    else:
        fit_params['spectral_radius'] = eval(config['esn']['spectral_radius'])
    if not (isinstance(eval(config['esn']['bias']), np.ndarray) or isinstance(eval(config['esn']['bias']), list)):
        esn_params['bias'] = eval(config['esn']['bias'])
    else:
        fit_params['bias'] = eval(config['esn']['bias'])
    if not (isinstance(eval(config['esn']['leakage']), np.ndarray) or isinstance(eval(config['esn']['leakage']), list)):
        esn_params['leakage'] = eval(config['esn']['leakage'])
    else:
        fit_params['leakage'] = eval(config['esn']['leakage'])
    if not (isinstance(eval(config['esn']['reservoir_size']), np.ndarray) or isinstance(eval(config['esn']['reservoir_size']), list)):
        esn_params['reservoir_size'] = eval(config['esn']['reservoir_size'])
    else:
        fit_params['reservoir_size'] = eval(config['esn']['reservoir_size'])
    if not (isinstance(eval(config['esn']['k_res']), np.ndarray) or isinstance(eval(config['esn']['k_res']), list)):
        esn_params['k_res'] = eval(config['esn']['k_res'])
    else:
        fit_params['k_res'] = eval(config['esn']['k_res'])

    esn_params['reservoir_activation'] = str(config['esn']['reservoir_activation'])

    if not (isinstance(eval(config['esn']['bi_directional']), np.ndarray) or isinstance(eval(config['esn']['bi_directional']), list)):
        esn_params['bi_directional'] = eval(config['esn']['bi_directional'])
    else:
        fit_params['bi_directional'] = eval(config['esn']['bi_directional'])

    esn_params['solver'] = str(config['esn']['solver'])

    if not (isinstance(eval(config['esn']['beta']), np.ndarray) or isinstance(eval(config['esn']['beta']), list)):
        esn_params['beta'] = eval(config['esn']['beta'])
    else:
        fit_params['beta'] = eval(config['esn']['beta'])
    if not (isinstance(eval(config['esn']['random_state']), np.ndarray) or isinstance(eval(config['esn']['random_state']), list)):
        esn_params['random_state'] = eval(config['esn']['random_state'])
    else:
        esn_params['random_state'] = eval(config['esn']['random_state'])

    loss_fn = config['train']['loss']
    n_jobs = int(config['train']['n_jobs'])

    # feature extraction
    feature_settings = dict()
    feature_settings['mono'] = config['input']['mono']
    feature_settings['fs'] = config['input']['fs']
    feature_settings['normalize'] = config['input']['normalize']
    feature_settings['frame_size'] = config['input']['frame_size']
    feature_settings['fps'] = config['input']['fps']
    feature_settings['filterbank'] = config['input']['filterbank']
    feature_settings['num_bands'] = config['input']['num_bands']
    feature_settings['fmin'] = config['input']['fmin']
    feature_settings['fmax'] = config['input']['fmax']
    feature_settings['norm_filters'] = config['input']['norm_filters']
    feature_settings['logarithm'] = config['input']['logarithm']
    feature_settings['mul'] = config['input']['mul']
    feature_settings['add'] = config['input']['add']
    feature_settings['diff_ratio'] = config['input']['diff_ratio']
    feature_settings['positive_diffs'] = config['input']['positive_diffs']
    feature_settings['scaler'] = config['input']['scaler']
    return io_params, esn_params, fit_params, feature_settings, loss_fn, n_jobs
