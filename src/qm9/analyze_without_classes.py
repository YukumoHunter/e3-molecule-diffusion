try:
    from rdkit import Chem
    from qm9.rdkit_functions import BasicMolecularMetrics
    use_rdkit = True
except ModuleNotFoundError:
    use_rdkit = False
import qm9.dataset as dataset
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sp_stats
from qm9 import bond_analyze


# 'atom_decoder': ['H', 'B', 'C', 'N', 'O', 'F', 'Al', 'Si', 'P', 'S', 'Cl', 'As', 'Br', 'I', 'Hg', 'Bi'],

analyzed_19 ={'atom_types': {1: 93818, 3: 21212, 0: 139496, 2: 8251, 4: 26},
            'distances': [0, 0, 0, 0, 0, 0, 0, 22566, 258690, 16534, 50256, 181302, 19676, 122590, 23874, 54834, 309290, 205426, 172004, 229940, 193180, 193058, 161294, 178292, 152184, 157242, 189186, 150298, 125750, 147020, 127574, 133654, 142696, 125906, 98168, 95340, 88632, 80694, 71750, 64466, 55740, 44570, 42850, 36084, 29310, 27268, 23696, 20254, 17112, 14130, 12220, 10660, 9112, 7640, 6378, 5350, 4384, 3650, 2840, 2362, 2050, 1662, 1414, 1216, 966, 856, 492, 516, 420, 326, 388, 326, 236, 140, 130, 92, 62, 52, 78, 56, 24, 8, 10, 12, 18, 2, 10, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}


#Histogram_discrete
def histogram_discrete_add(bins, elements):
    for e in elements:
        if e in bins:
            bins[e] += 1
        else:
            bins[e] = 1

def histogram_discrete_normalize(bins):
    total = 0.
    for key in bins:
        total += bins[key]
    for key in bins:
        bins[key] = bins[key] / total

def histogram_discrete_plot(bins, name='histogram', save_path=None):
    width = 1  # the width of the bars
    fig, ax = plt.subplots()
    x, y = [], []
    for key in bins:
        x.append(key)
        y.append(bins[key])

    ax.bar(x, y, width)
    plt.title(name)
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


# Histogram_cont
def histogram_cont_add(elements, num_bins=100, range=(0., 13.), ignore_zeros=False):
    bins = [0] * num_bins
    for e in elements:
        if not ignore_zeros or e > 1e-8:
            i = int(float(e) / range[1] * len(bins))
            i = min(i, len(bins) - 1)
            bins[i] += 1

def histogram_cont_plot(num_bins=100, save_path=None, range=(0., 13.), name='histogram'):
    bins = [0] * num_bins
    width = (range[1] - range[0])/len(bins)   # the width of the bars
    fig, ax = plt.subplots()

    x = np.linspace(range[0], range[1], num=len(bins) + 1)[:-1] + width / 2
    ax.bar(x, bins, width)
    plt.title(name)

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def histogram_cont_normalize_histogram(hist):
    hist = jnp.array(hist)
    prob = hist / jnp.sum(hist)
    return prob
    
def histogram_cont_plot_both(hist_b, num_bins=100, save_path=None, wandb=None, name='histogram'):
        bins = [0] * num_bins
        ## TO DO: Check if the relation of bins and linspace is correct
        hist_a = histogram_cont_normalize_histogram(bins)
        hist_b = histogram_cont_normalize_histogram(hist_b)

        #width = (range[1] - range[0]) / len(bins)  # the width of the bars
        fig, ax = plt.subplots()
        x = np.linspace(range[0], range[1], num=len(bins) + 1)[:-1]
        ax.step(x, hist_b)
        ax.step(x, hist_a)
        ax.legend(['True', 'Learned'])
        plt.title(name)

        if save_path is not None:
            plt.savefig(save_path)
            if wandb is not None:
                if wandb is not None:
                    # Log image(s)
                    im = plt.imread(save_path)
                    wandb.log({save_path: [wandb.Image(im, caption=save_path)]})
        else:
            plt.show()
        plt.close()

def coord2distances(x):
    x = x.unsqueeze(2)
    x_t = x.transpose(1, 2)
    dist = (x - x_t) ** 2
    dist = jnp.sqrt(jnp.sum(dist, axis=3))
    dist = dist.flatten()
    return dist