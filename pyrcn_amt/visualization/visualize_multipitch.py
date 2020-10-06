import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


def visualize_multipitch(multipitch, fps=100.):
    plt.figure()
    n = np.arange(len(multipitch)) / fps
    plt.imshow(multipitch.T, origin='lower', vmin=0, vmax=1, aspect='auto', cmap='gray')
    plt.xlabel('n')
    plt.ylabel('y[n]')
    plt.grid()
    plt.tight_layout()
    plt.show()


def visualize_multipitch_with_targets(multipitch, targets, fps=100.):
    plot_matrix = np.zeros(shape=(multipitch.shape[0], multipitch.shape[1], 3))
    for t in range(multipitch.shape[0]):
        for p in range(multipitch.shape[1]):
            if targets[t, p] == 0 and multipitch[t, p] == 0:
                plot_matrix[t, p, 0] = 1
                plot_matrix[t, p, 1] = 1
                plot_matrix[t, p, 2] = 1
            elif targets[t, p] == 1 and multipitch[t, p] == 1:
                plot_matrix[t, p, 0] = 0
                plot_matrix[t, p, 1] = 0
                plot_matrix[t, p, 2] = 1
            elif targets[t, p] == 1 and multipitch[t, p] == 0:
                plot_matrix[t, p, 0] = 1
                plot_matrix[t, p, 1] = 0
                plot_matrix[t, p, 2] = 0
            elif targets[t, p] == 0 and multipitch[t, p] == 1:
                plot_matrix[t, p, 0] = 0
                plot_matrix[t, p, 1] = 1
                plot_matrix[t, p, 2] = 0
            else:
                raise ValueError
    plt.figure()
    n = np.arange(len(multipitch)) / fps
    plt.imshow(plot_matrix.swapaxes(0, 1), origin='lower', aspect='auto')
    plt.xlabel('n')
    plt.ylabel('y[n]')
    plt.grid()
    plt.tight_layout()
    plt.show()
