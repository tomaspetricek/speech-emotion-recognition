import os
import torch
from torch.utils.data import Dataset

from files import TextFile

import re


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

        label = self._get_sample_label(sample_basename)

        sample_full_path = os.path.join(self.root_dir, sample_basename)

        sample = torch.load(sample_full_path)

        return sample, label
