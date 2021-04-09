import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerTuple
#Options
params = {'image.cmap': 'RdBu',
          # 'text.usetex': True,
          'font.size': 10,
          'axes.titlesize': 24,
          'axes.labelsize': 10,
          'lines.linewidth': 1,
          'lines.markersize': 5,
          'lines.markeredgewidth': 1,
          'xtick.labelsize': 10,
          'ytick.labelsize': 10,
          # 'text.latex.unicode': True,
          }
plt.rcParams.update(params)
# plt.rcParams['pdf.fonttype'] = 42
# plt.rcParams['ps.fonttype'] = 42
# matplotlib.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "Times New Roman"

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf')

width = 4.80315
height = width / 1.618


def visualize_odf(odf, fps=100.):
    fig = plt.figure()
    n = np.arange(len(odf)) / fps
    plt.plot(n, odf)
    plt.xlim([0, n[-1]])
    plt.ylim([0, 1])
    plt.xlabel('n')
    plt.ylabel('ODF[n]')
    fig.set_size_inches(width, height)
    plt.show(bbox_inches='tight', pad_inches=0)


def visualize_odf_with_targets(odf, targets, fps=100.):
    fig = plt.figure()
    n = np.arange(len(odf)) / fps
    for target in targets:
        plt.axvline(target, color='k')
    plt.plot(n, odf)
    plt.xlim([0, n[-1]])
    plt.ylim([0, 1])
    plt.xlabel('n')
    plt.ylabel('ODF[n]')
    fig.set_size_inches(width, height)
    plt.show(bbox_inches='tight', pad_inches=0)


def visualize_odf_with_targets_predictions(odf, targets, predictions, fps=100.):
    fig = plt.figure()
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
    fig.set_size_inches(width, height)
    plt.show(bbox_inches='tight', pad_inches=0)


def visualize_features_with_targets(features, targets, fps=100.):
    fig = plt.figure()
    plt.imshow(features.T, origin='lower', vmin=0, vmax=1, aspect='auto')
    plt.xlabel('n')
    plt.ylabel('X[n]')
    plt.grid()
    plt.xlim([0, features.shape[0]])
    plt.ylim([0, features.shape[1]])
    for target in targets:
        plt.axvline(target * fps, color='w')
    fig.set_size_inches(width, height)
    plt.show(bbox_inches='tight', pad_inches=0)


def visualize_features_with_targets_predictions(features, targets, predictions, fps=100.):
    fig = plt.figure()
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
    fig.set_size_inches(width, height)
    plt.show(bbox_inches='tight', pad_inches=0)


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
    fig.set_size_inches(width, height)
    plt.show(bbox_inches='tight', pad_inches=0)


def visualize_features_odf_with_targets_predictions(features, odf, targets, predictions, fps=100.):
    fig, (ax0, ax1) = plt.subplots(2, 1)
    n = np.arange(len(features))
    ax0.imshow(features.T, origin='lower', vmin=0, vmax=1, aspect='auto')
    # ax0.set_xlabel('n')
    ax0.set_ylabel('X[n]')
    ax0.set_xlim([0, features.shape[0] + 1])
    ax0.grid()
    for target in targets[:-1]:
        ax0.axvline(target * fps, color='w')
    p0 = ax0.axvline(targets[-1] * fps, color='w', label='Target')
    ax0.set_xticks([])
    colors = plt.cm.bwr(np.linspace(0,1,2))
    for target in targets[:-1]:
        ax1.axvline(target * fps, c='k')
    p3 = ax1.axvline(targets[-1] * fps, c='k', label='Reference')
    p1, = ax1.plot(n, odf[1], color=colors[0], label="KM-ESN")
    p2, = ax1.plot(n, odf[0], color=colors[1], label="Basic ESN")
    for prediction in predictions[1]:
        ax1.scatter(prediction * fps, 1, c=colors[0], edgecolors=colors[0], marker='x')
    p4 = ax1.scatter(predictions[1][-1] * fps, 1, c=colors[0], edgecolors=colors[0], marker='x', label="KM-ESN")
    for prediction in predictions[0]:
        ax1.scatter(prediction * fps, 0.9, c=colors[1], edgecolors=colors[1], marker='x')
    p5 = ax1.scatter(predictions[0][-1] * fps, 0.9, c=colors[1], edgecolors=colors[1], marker='x', label="Basic ESN")
    ax1.set_xlabel('n')
    ax1.set_ylabel('ODF[n]')
    ax1.set_xlim([0, features.shape[0] + 1])
    ax0.legend([(p0, p3), (p1, p4), (p2, p5)], ["Reference", "KM-ESN", "Basic ESN"],
               loc=9,
               ncol=3,
               bbox_to_anchor=(0.5, 1.45),
               handler_map={tuple: HandlerTuple(ndivide=None)})
    fig.set_size_inches(width, height)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    plt.show()
