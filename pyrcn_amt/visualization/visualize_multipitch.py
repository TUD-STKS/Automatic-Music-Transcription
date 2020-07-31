import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


def visualize_multipitch(multipitch, fps=100.):
    plt.figure()
    n = np.arange(len(multipitch)) / fps
    plt.plot(n, multipitch)
    plt.xlim([0, n[-1]])
    plt.ylim([0, 1])
    plt.xlabel('n')
    plt.ylabel('ODF[n]')
    plt.tight_layout()
    plt.show()


def visualize_multipitch_with_targets(multipitch, targets, fps=100.):
    plt.figure()
    n = np.arange(len(multipitch)) / fps
    plt.plot(n, multipitch)
    plt.xlim([0, n[-1]])
    plt.ylim([0, 1])
    plt.xlabel('n')
    plt.ylabel('ODF[n]')
    for target in targets:
        plt.axvline(target, color='w')
    plt.tight_layout()
    plt.show()


def visualize_multipitch_with_targets_predictions(multipitch, targets, predictions, fps=100.):
    plt.figure()
    n = np.arange(len(multipitch)) / fps
    plt.plot(n, multipitch)
    plt.xlim([0, n[-1]])
    plt.ylim([0, 1])
    plt.xlabel('n')
    plt.ylabel('ODF[n]')
    for target in targets:
        plt.axvline(target, color='w')
    for prediction in predictions:
        plt.axvline(prediction, color='g')
    plt.tight_layout()
    plt.show()


def visualize_features_with_targets(features, targets, fps=100.):
    plt.figure(figsize=(12, 6))
    n = np.arange(len(features)) / fps
    plt.imshow(features.T, origin='lower', vmin=0, vmax=1, aspect='auto')
    plt.xlabel('n')
    plt.ylabel('X[n]')
    plt.grid()
    for target in targets[0]:
        plt.axvline(target, color='w')
    plt.tight_layout()
    plt.show()


def visualize_features_with_targets_predictions(features, targets, predictions, fps=100.):
    plt.figure(figsize=(12, 6))
    n = np.arange(len(features)) / fps
    plt.imshow(features.T, origin='lower', vmin=0, vmax=1, aspect='auto')
    plt.xlabel('n')
    plt.ylabel('X[n]')
    plt.grid()
    for target in targets[0]:
        plt.axvline(target, color='w')
    for prediction in predictions[0]:
        plt.axvline(prediction, color='g')
    plt.tight_layout()
    plt.show()


def visualize_features_multipitch_with_targets(features, multipitch, targets, fps=100.):
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 6))
    n = np.arange(len(features)) / fps
    ax0.imshow(features.T, origin='lower', vmin=0, vmax=1, aspect='auto')
    ax0.set_xlabel('n')
    ax0.set_ylabel('X[n]')
    ax0.grid()
    for target in targets:
        ax0.axvline(target, color='w')

    ax1.plot(multipitch)
    ax1.set_xlabel('n')
    ax1.set_ylabel('multipitch[n]')
    ax1.grid()
    for target in targets:
        ax0.axvline(target, color='w')

    plt.tight_layout()
    plt.show()


def visualize_features_multipitch_with_targets_predictions(features, multipitch, targets, predictions, fps=100.):
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 6))
    n = np.arange(len(features)) / fps
    ax0.imshow(features.T, origin='lower', vmin=0, vmax=1, aspect='auto')
    ax0.set_xlabel('n')
    ax0.set_ylabel('X[n]')
    ax0.grid()
    for target in targets:
        ax0.axvline(target, color='w')
    for prediction in predictions:
        ax0.axvline(prediction, color='g')

    ax1.plot(multipitch)
    ax1.set_xlabel('n')
    ax1.set_ylabel('multipitch[n]')
    ax1.grid()
    for target in targets:
        ax0.axvline(target, color='w')
    for prediction in predictions:
        ax0.axvline(prediction, color='g')

    plt.tight_layout()
    plt.show()
