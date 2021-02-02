import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class NumpyDataset(Dataset):
    def __init__(self, n_samples, samples_lengths, samples_path, labels_path, index_picker):
        self.n_samples = n_samples
        self.last_indices = samples_lengths
        self.samples = samples_path
        self.n_frames = self.samples.shape[0]
        self.labels = np.load(labels_path)
        self.index_picker = index_picker

    def set_samples(self, samples_path):
        self._samples = np.load(samples_path)

    def get_samples(self):
        return self._samples

    samples = property(get_samples, set_samples)

    def set_last_indices(self, sample_lengths):
        self._last_indices = []
        base = 0
        for sample_length in sample_lengths:
            last_index = base + sample_length - 1
            self._last_indices.append(last_index)
            base += sample_length

    def get_last_indices(self):
        return self._last_indices

    last_indices = property(get_last_indices, set_last_indices)

    def _get_sample_index(self, frame_index):
        previous_index = -1
        for sample_index, last_index in enumerate(self._last_indices):
            if previous_index < frame_index <= last_index:
                return sample_index

            previous_index = last_index

    def _get_sample_indices(self, sample_index):
        if sample_index - 1 < 0:
            first_index = 0
        else:
            first_index = sample_index + 1

        last_index = self._last_indices[sample_index]

        return range(first_index, last_index + 1)

    def add_margin(self, sample_indices):
        sample_margined_indices = []

        # check edges
        for sample_index in sample_indices:
            frame_indices = self.index_picker.pick(sample_index)

            for index, frame_index in enumerate(frame_indices):
                if frame_index < 0:
                    frame_indices[index] = 0
                elif frame_index > self.n_frames - 1:
                    frame_indices[index] = self.n_frames - 1

            sample_margined_indices.append(frame_indices)

        sample = np.take(self.samples, sample_margined_indices, axis=0)

        # reshape
        n_frames, window_length, n_features = sample.shape
        sample = np.array(np.reshape(sample, (n_frames, -1)))
        return sample

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass


class NumpyFrameDataset(NumpyDataset):

    def __len__(self):
        return self.n_frames

    def __getitem__(self, frame_index):
        sample_index = self._get_sample_index(frame_index)

        # add margin
        sample = self.add_margin([frame_index])
        sample = sample.flatten()
        sample = torch.tensor(sample)

        label = self.labels[sample_index]
        label = torch.tensor(label)
        return sample, label

class NumpySampleDataset(NumpyDataset):

    def __len__(self):
        return self.n_samples

    def __getitem__(self, sample_index):
        sample_indices = self._get_sample_indices(sample_index)

        # add margin
        sample = self.add_margin(sample_indices)

        sample = torch.from_numpy(sample)

        label = int(self.labels[sample_index])
        label = [label] * sample.shape[0]
        label = torch.tensor(label)
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

