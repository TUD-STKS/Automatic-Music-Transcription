import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 30
plt.rc('text', usetex=True)


def visualize_multipitch(multipitch, fps=100.):
    plt.figure(figsize=(12, 4.8))
    n = np.arange(len(multipitch)) / fps
    plt.imshow(multipitch.T, origin='lower', aspect='auto', cmap='gray')  # vmin=0, vmax=1,
    plt.xlabel(r'$n$')
    plt.ylabel(r'$\mathbf{y}[n]$')
    plt.locator_params(axis='y', nbins=5)
    plt.grid()
    plt.colorbar()
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
    plt.figure(figsize=(12, 4.8))
    n = np.arange(len(multipitch)) / fps
    plt.imshow(plot_matrix.swapaxes(0, 1), origin='lower', aspect='auto')
    plt.xlabel(r'$n$')
    plt.ylabel(r'\textrm{MIDI Pitch}')
    plt.locator_params(axis='y', nbins=5)
    plt.grid()
    plt.tight_layout()
    plt.show()
