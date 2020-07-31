import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


def visualize_features(features, fps=100.):
    plt.figure(figsize=(12, 6))
    n = np.arange(len(features)) / fps
    plt.imshow(features.T, origin='lower', vmin=0, vmax=1, aspect='auto')
    plt.xlabel('n')
    plt.ylabel('X[n]')
    plt.grid()
    plt.tight_layout()
    plt.show()
