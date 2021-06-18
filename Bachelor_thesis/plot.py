import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_lines(ax, items, title, y_label, x_label):
    labels = list()
    n_epochs = None

    for label, item in items.items():
        if item:
            ax.plot(item)
            labels.append(label)
            n_epochs = len(item)

    ax.set_title(title)
    ax.set_xlabel(y_label)
    ax.set_ylabel(x_label)
    ax.legend(labels, loc='upper left')
    plt.xticks(np.arange(n_epochs), np.arange(1, n_epochs + 1))


def plot_conf_matrix(ax, conf_matrix, title, y_label, x_label, annot=True):
    sns.heatmap(conf_matrix, ax=ax, annot=annot, fmt='g')
    ax.set_title(title)
    ax.set_xlabel(y_label)
    ax.set_ylabel(x_label)
    plt.tight_layout()



