import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


def visualize_odf(odf, fps=100.):
    plt.figure()
    n = np.arange(len(odf)) / fps
    plt.plot(n, odf)
    plt.xlim([0, n[-1]])
    plt.ylim([0, 1])
    plt.xlabel('n')
    plt.ylabel('ODF[n]')
    plt.tight_layout()
    plt.show()


def visualize_odf_with_targets(odf, targets, fps=100.):
    plt.figure()
    n = np.arange(len(odf)) / fps
    for target in targets:
        plt.axvline(target, color='k')
    plt.plot(n, odf)
    plt.xlim([0, n[-1]])
    plt.ylim([0, 1])
    plt.xlabel('n')
    plt.ylabel('ODF[n]')
    plt.tight_layout()
    plt.show()


def visualize_odf_with_targets_predictions(odf, targets, predictions, fps=100.):
    plt.figure()
    n = np.arange(len(odf)) / fps
    for target in targets:
        plt.axvline(target, color='k')
    for prediction in predictions:
        plt.axvline(prediction, color='g')
    plt.plot(n, odf)
    plt.xlim([0, n[-1]])
    plt.ylim([0, 1])
    plt.xlabel('n')
    plt.ylabel('ODF[n]')
    plt.tight_layout()
    plt.show()


def visualize_features_with_targets(features, targets, fps=100.):
    plt.figure(figsize=(12, 6))
    plt.imshow(features.T, origin='lower', vmin=0, vmax=1, aspect='auto')
    plt.xlabel('n')
    plt.ylabel('X[n]')
    plt.grid()
    plt.xlim([0, features.shape[0]])
    plt.ylim([0, features.shape[1]])
    for target in targets:
        plt.axvline(target * fps, color='w')
    plt.tight_layout()
    plt.show()


def visualize_features_with_targets_predictions(features, targets, predictions, fps=100.):
    plt.figure(figsize=(12, 6))
    plt.imshow(features.T, origin='lower', vmin=0, vmax=1, aspect='auto')
    plt.xlabel('n')
    plt.ylabel('X[n]')
    plt.grid()
    plt.xlim([0, features.shape[0]])
    plt.ylim([0, features.shape[1]])
    for target in targets[0]:
        plt.axvline(target * fps, color='w')
    for prediction in predictions[0]:
        plt.axvline(prediction * fps, color='g')
    plt.tight_layout()
    plt.show()


def visualize_features_odf_with_targets(features, odf, targets, fps=100.):
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 6))
    n = np.arange(len(features))
    ax0.imshow(features.T, origin='lower', vmin=0, vmax=1, aspect='auto')
    ax0.set_xlabel('n')
    ax0.set_ylabel('X[n]')
    ax0.grid()
    ax0.set_xlim([0, features.shape[0]])
    ax0.set_ylim([0, features.shape[1]])
    ax1.set_xlim([0, features.shape[0]])
    for target in targets:
        ax0.axvline(target * fps, color='w')

    for target in targets:
        ax1.axvline(target * fps, color='k')
    ax1.plot(n, odf)
    ax1.set_xlabel('n')
    ax1.set_ylabel('odf[n]')
    ax1.grid()

    plt.tight_layout()
    plt.show()


def visualize_features_odf_with_targets_predictions(features, odf, targets, predictions, fps=100.):
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 6))
    n = np.arange(len(features)) / fps
    ax0.imshow(features.T, origin='lower', vmin=0, vmax=1, aspect='auto')
    ax0.set_xlabel('n')
    ax0.set_ylabel('X[n]')
    ax0.grid()
    for target in targets:
        ax0.axvline(target * fps, color='w')
    for prediction in predictions:
        ax0.axvline(prediction * fps, color='g')

    for target in targets:
        ax1.axvline(target, color='k')
    for prediction in predictions:
        ax1.axvline(prediction, color='g')
    ax1.plot(n, odf)
    ax1.set_xlabel('n')
    ax1.set_ylabel('odf[n]')
    ax1.grid()

    plt.tight_layout()
    plt.show()
