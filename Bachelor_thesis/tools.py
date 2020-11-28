import numpy as np


def add_margin(samples, index_picker):
    n_samples, _ = samples.shape
    samples_indices = []
    for sample_index, _ in enumerate(samples):
        sample_indices = index_picker.pick(sample_index)

        for index, samples_index in enumerate(sample_indices):
            if samples_index < 0:
                sample_indices[index] = 0
            elif samples_index > n_samples - 1:
                sample_indices[index] = n_samples - 1

        samples_indices.append(sample_indices)

    return np.take(samples, samples_indices, axis=0)


class IndexPicker:

    def __init__(self, left_margin, right_margin):
        self.left_margin = left_margin
        self.right_margin = right_margin

    def pick(self, index):
        from_index = index - self.left_margin
        to_index = index + self.right_margin
        return list(range(from_index, to_index + 1))
