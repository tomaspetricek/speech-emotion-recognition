from utils import *


class Dataset(object):
    """
    Represents a dataset.
    """

    def __init__(self, path, sample_format=WAV, samples=None):
        self.path = path
        self.sample_format = sample_format
        self.samples = samples

    def set_samples(self, value):
        if value is None:
            self._samples = get_file_paths(
                directory=self.path,
                file_extensions=[self.sample_format]
            )

    def get_samples(self):
        return self._samples

    samples = property(get_samples, set_samples)

    def clone(self, clone_path):
        # copy dataset content
        copy_directory_content(
            source=self.path,
            destination=clone_path,
        )

        # create dataset
        clone_dataset = Dataset(
            path=clone_path,
            sample_format=self.sample_format
        )

        return clone_dataset
