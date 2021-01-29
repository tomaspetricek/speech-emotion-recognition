import torch
from torch.utils.data import Dataset
import numpy as np


class NumpyDataset(Dataset):
    def __init__(self, sample_lengths, sample_path, label_path, index_picker):
        self.n_samples = len(sample_lengths)
        self.samples_indices = sample_lengths
        self.samples = np.load(sample_path)
        self.n_frames = self.samples.shape[0]
        self.labels = np.load(label_path)
        self.index_picker = index_picker

    def set_samples_indices(self, sample_lengths):
        self._samples_indices = []
        base = 0
        for sample_length in sample_lengths:
            first_index = base
            last_index = base + sample_length
            sample_indices = range(first_index, last_index)
            self._samples_indices.append(sample_indices)
            base += sample_length

    def get_samples_indices(self):
        return self._samples_indices

    samples_indices = property(get_samples_indices, set_samples_indices)

    def __len__(self):
        return self.n_samples

    def add_margin(self, frame_indices):
        sample_indices = []
        for frame_index in frame_indices:
            frame_margined_indices = self.index_picker.pick(frame_index)

            for index, frame_margined_index in enumerate(frame_margined_indices):
                if frame_margined_index < 0:
                    frame_margined_indices[index] = 0
                elif frame_margined_index > self.n_frames - 1:
                    frame_margined_indices[index] = self.n_frames - 1

            sample_indices.append(frame_margined_indices)

        sample = np.take(self.samples, sample_indices, axis=0)

        # reshape
        n_frames, window_length, n_features = sample.shape
        sample = np.array(np.reshape(sample, (n_frames, -1)))
        return sample

    def __getitem__(self, index):
        frame_indices = self._samples_indices[index]

        # add margin
        sample = self.add_margin(frame_indices)
        label = torch.from_numpy(self.labels[index])
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

    def __getitem__(self, index):
        # check if chunk changed
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

