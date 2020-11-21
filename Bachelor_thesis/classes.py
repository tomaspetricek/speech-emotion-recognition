from utils import *
import pandas as pd
import numpy as np
from files import HTKFile, WAVFile

class Data:
    FILE = None
    COLUMNS = [
        "data"
    ]

    def parse(self, file_path):
        pass

class MFCCData(Data):
    FILE = HTKFile()
    SAMPLE_FORMAT = ".mfcc_0_d_a"
    COLUMNS = [
        "coefficients",
        "frame",
    ]

    def parse(self, file_path):
        results = self.FILE.read(file_path)
        data = []
        for index, result in enumerate(results):
            data.append([result, index + 1])

        return data

class WAVData(Data):
    SAMPLE_FORMAT = ".wav"
    FILE = WAVFile()

    COLUMNS = [
        "sample rate",
        "data"
    ]

    def parse(self, file_path):
        sample_rate, data = self.FILE.read(file_path)
        return [sample_rate, data]

class Label:
    COLUMNS = []
    SEPARATOR = None

    def parse(self, file_path):
        filename = os.path.basename(file_path)
        name, ext = os.path.splitext(filename)
        label = name.split(self.SEPARATOR)
        return list(map(int, label))

class RAVDESSLabel(Label):
    SEPARATOR = "-"

    COLUMNS = [
        "modality",
        "vocal channel",
        "emotion",
        "emotional intensity",
        "statement",
        "repetition",
        "actor"
    ]

    MODALITY_OPTIONS = {
        1: "full - AV",
        2: "video - only",
        3: "audio - only",
    }

    VOCAL_CHANNEL_OPTIONS = {
        1: "speech",
        2: "song",
    }

    EMOTION_OPTIONS = {
        1: "neutral",
        2: "calm",
        3: "happy",
        4: "sad",
        5: "angry",
        6: "fearful",
        7: "disgust",
        8: "surprised",
    }

    EMOTIONAL_INTENSITY_OPTIONS = {
        1: "normal",
        2: "strong",
    }

    STATEMENT_OPTIONS = {
        1: "Kids are talking by the door",
        2: "Dogs are sitting by the door",
    }

    REPETITION_OPTIONS = {
        1: "1 st repetition",
        2: "2 nd repetition",
    }

class Dataset:
    """
    Represents a dataset.
    """
    def __init__(self, path, data, label):
        self.path = path
        self.data = data
        self.label = label
        self.data_columns = []
        self.label_columns = []
        self.sample_columns = None
        self.file_paths = None
        self.samples = None

    def set_sample_columns(self, value):
        if self.data:
            self.data_columns = self.data.COLUMNS

        if self.label:
            self.label_columns = self.label.COLUMNS

        self._sample_columns = self.data_columns + self.label_columns

    def get_sample_columns(self):
        return self._sample_columns

    def set_file_paths(self, value):
        self._file_paths = None

        if value is None:
            self._file_paths = get_file_paths(
                directory=self.path,
                file_extensions=[self.data.SAMPLE_FORMAT]
            )

    def get_file_paths(self):
        return self._file_paths

    def set_samples(self, value):
        samples = []
        for file_path in self._file_paths:
            data = []
            if self.data_columns:
                data = self.data.parse(file_path)

            label = []
            if self.label_columns:
                label = self.label.parse(file_path)

            # check if 2D nested list
            if all(isinstance(i, list) for i in data):
                for d in data:
                    samples.append(d + label)
            else:
                samples.append(data + label)

        self._samples = pd.DataFrame(samples, columns=self._sample_columns)

    def get_samples(self):
        return self._samples

    sample_columns = property(get_sample_columns, set_sample_columns)
    file_paths = property(get_file_paths, set_file_paths)
    samples = property(get_samples, set_samples)

    def clone(self, clone_path, ignore_file_extensions=None):
        # copy dataset content
        copy_directory_content(
            source=self.path,
            destination=clone_path,
            ignore_file_extensions=ignore_file_extensions
        )

        # create dataset
        clone_dataset = Dataset(
            path=clone_path,
        )

        return clone_dataset


if __name__ == "__main__":
    from config import DATASET_PATH

    path = DATASET_PATH.format(language="english", name="RAVDESS", form="mfcc")

    ravdess_mfcc = Dataset(path, MFCCData(), RAVDESSLabel())





