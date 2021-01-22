import os
import torch
from torch.utils.data import Dataset
import re
import dask.dataframe as dd
import numpy as np
import pandas as pd

from files import TextFile


class CustomDataset(Dataset):
    """
    Inspired by: https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    """
    _SAMPLE_NAME = re.compile(r"(?P<ID>\d+)_(?P<label>\d+)")

    def __init__(self, root_dir, annotations_path):
        self.root_dir = root_dir
        self.annotations = TextFile(annotations_path).read_lines()

    def __len__(self):
        return len(self.annotations)

    def _get_sample_label(self, sample_basename):
        name = os.path.splitext(sample_basename)[0]
        result = self._SAMPLE_NAME.match(name)
        groups = result.groupdict()
        return groups["label"]

    def __getitem__(self, index):
        sample_basename = self.annotations[index]

        label = int(self._get_sample_label(sample_basename))

        sample_full_path = os.path.join(self.root_dir, sample_basename)

        sample = torch.load(sample_full_path)

        return sample, label


class DaskDataset(Dataset):

    def __init__(self, samples_csv_path, labels_csv_path):
        self.samples = dd.read_csv(samples_csv_path)
        self.labels = dd.read_csv(labels_csv_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = torch.tensor(np.array(self.samples.loc[index]))
        label = torch.tensor(np.array(self.labels.loc[index]))
        return sample, label


class NumpyDataset(Dataset):
    def __init__(self, samples_path, labels_path):
        self.samples = np.load(samples_path)
        self.labels = np.load(labels_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = torch.tensor(self.samples[index])
        label = torch.tensor(self.labels[index])
        return sample, label


class NumpySplitDataset(Dataset):
    def __init__(self, samples_paths, labels_paths, chunk_sizes):
        self.samples_paths = samples_paths
        self.labels_paths = labels_paths
        self.n_samples = sum(chunk_sizes)
        self.transition_indices = chunk_sizes
        self.chunk = None
        self.samples = None
        self.labels = None
        self.base = None

    def set_transition_indices(self, chunk_sizes):
        self._transition_indices = []
        base = 0
        for chunk_size in chunk_sizes:
            transition_index = base + chunk_size
            self._transition_indices.append(transition_index)
            base += chunk_size

    def get_transition_indices(self):
        return self._transition_indices

    transition_indices = property(get_transition_indices, set_transition_indices)

    def __len__(self):
        return self.n_samples

    def _pick_chunk(self, index):
        previous_index = 0
        for chunk, transition_index in enumerate(self._transition_indices):
            if previous_index <= index < transition_index:
                return chunk

            previous_index = transition_index

    def __getitem__(self, index):   # TODO Fix access to any element
        # check if chuck changed
        chunk = self._pick_chunk(index)
        if self.chunk != chunk:
            self.chunk = chunk
            self.samples = np.load(self.samples_paths[self.chunk])
            self.labels = np.load(self.labels_paths[self.chunk])

            if chunk - 1 == -1:
                self.base = 0
            else:
                self.base = self._transition_indices[chunk - 1]

        chunk_index = index - self.base
        sample = torch.tensor(self.samples[chunk_index])
        label = torch.tensor(self.labels[chunk_index])
        return sample, label

