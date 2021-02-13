import torch
from torch.utils.data import Dataset
import numpy as np


class NumpyDataset(Dataset):
    def __init__(self, n_samples, samples_lengths, samples_path, labels_path, left_margin, right_margin):
        self.n_samples = n_samples
        self.left_margin = left_margin
        self.right_margin = right_margin
        self.sample_lengths = samples_lengths
        self.n_frames = None    # set in set_samples
        self.samples = samples_path
        self.labels = labels_path
        self.samples_indices = None

    def set_samples(self, samples_path):
        self._samples = np.load(samples_path)
        self.n_frames = self._samples.shape[0]

        # add boundaries - copy of last frame at each end
        first_sample = list(self._samples[0])
        left_sample_margin = self.left_margin * [first_sample]
        left_sample_margin = np.array(left_sample_margin)
        
        last_sample = list(self._samples[-1])
        right_sample_margin = self.right_margin * [last_sample]
        right_sample_margin = np.array(right_sample_margin)
        self._samples = np.concatenate((left_sample_margin, self._samples, right_sample_margin), axis=0)

    def get_samples(self):
        return self._samples

    samples = property(get_samples, set_samples)

    def set_labels(self, labels_path):
        labels = np.load(labels_path)

        self._labels = []
        for index, sample_length in enumerate(self.sample_lengths):
            self._labels += [labels[index]] * sample_length

        # add margins
        first_label = labels[0]
        left_label_margin = self.left_margin * [first_label]

        last_label = labels[-1]
        right_label_margin = self.right_margin * [last_label]

        self._labels = left_label_margin + self._labels + right_label_margin

        # convert to numpy array
        self._labels = np.array(self._labels)

    def get_labels(self):
        return self._labels

    labels = property(get_labels, set_labels)

    def set_samples_indices(self, value):
        self._samples_indices = []
        first_index = self.left_margin

        for sample_length in self.sample_lengths:
            last_index = first_index + sample_length - 1
            self._samples_indices.append((first_index, last_index))
            first_index = last_index + 1

    def get_samples_indices(self):
        return self._samples_indices

    samples_indices = property(get_samples_indices, set_samples_indices)

    def add_margin(self, first_frame_index, last_frame_index):
        frames_indices = range(first_frame_index, last_frame_index + 1)
        n_features = self.samples.shape[1]
        n_margined_features = n_features * (self.left_margin + 1 + self.right_margin)
        n_frames = last_frame_index - first_frame_index + 1
        frames_margined = np.zeros(shape=(n_frames, n_margined_features))

        # add margin
        for frame_index in frames_indices:
            first_index = frame_index - self.left_margin
            last_index = frame_index + self.right_margin

            index = frame_index - first_frame_index
            frame_margined = self._samples[first_index:last_index + 1]
            frames_margined[index] = frame_margined.flatten()

        return frames_margined

    def _get_sample(self, index):
        pass

    def _get_label(self, index):
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass


class NumpyFrameDataset(NumpyDataset):

    def _get_sample(self, frame_index):
        # add margin
        first_index = last_index = frame_index
        sample = self.add_margin(first_index, last_index)
        sample = sample.flatten()
        sample = torch.tensor(sample)
        return sample

    def _get_label(self, frame_index):
        label = self._labels[frame_index]
        label = torch.tensor(label)
        return label

    def __len__(self):
        return self.n_frames

    def __getitem__(self, frame_index):
        frame_index = frame_index + self.left_margin
        sample = self._get_sample(frame_index)
        label = self._get_label(frame_index)
        return sample, label

class NumpySampleDataset(NumpyDataset):

    def _get_sample(self, sample_index):
        # add margin
        first_index, last_index = self._samples_indices[sample_index]
        sample = self.add_margin(first_index, last_index)
        sample = torch.tensor(sample)
        return sample

    def _get_label(self, sample_index):
        first_index, last_index = self._samples_indices[sample_index]
        label = self._labels[first_index:last_index + 1]
        label = torch.tensor(label)
        return label

    def __len__(self):
        return self.n_samples

    def __getitem__(self, sample_index):
        sample = self._get_sample(sample_index)
        label = self._get_label(sample_index)
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

