from utils import *
import re
import pandas as pd
import numpy as np
from files import HTKFile


class Dataset:
    """
    Represents a dataset.
    """

    FILE = None
    SAMPLE_FORMAT = ".wav"

    LABEL_REGEX = None
    LABEL_SEPARATOR = None
    LABEL_COLUMNS = None

    DATA_COLUMNS = [
        "data"
    ]

    SAMPLE_COLUMNS = None

    def __init__(self, path):
        self.path = path
        self.samples = None

    def set_samples(self, value):

        self.file_paths = None

        if value is None:
            self.file_paths = get_file_paths(
                directory=self.path,
                file_extensions=[self.SAMPLE_FORMAT]
            )

        samples = []
        for file_path in self.file_paths:

            data = self.FILE.read(file_path)

            filename = os.path.basename(file_path)
            name, ext = os.path.splitext(filename)
            label = name.split(self.LABEL_SEPARATOR)
            label = list(map(int, label))

            samples.append([data] + label)

        self._samples = pd.DataFrame(samples, columns=self.SAMPLE_COLUMNS)

    def get_samples(self):
        return self._samples

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
            sample_format=self.SAMPLE_FORMAT,
        )

        return clone_dataset


class RAVDESS(Dataset):

    SAMPLE_FORMAT = ".mfcc_0_d_a"
    FILE = HTKFile()

    LABEL_SEPARATOR = "-"
    LABEL_REGEX = re.compile(r'(?P<modality>\d+)-(?P<vocal_channel>\d+)-(?P<emotion>\d+)-(?P<emotional_intensity>\d+)-'
                       r'(?P<statement>\d+)-(?P<repetition>\d+)-(?P<actor>\d+)')

    LABEL_COLUMNS = [
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

    DATA_COLUMNS = Dataset.DATA_COLUMNS

    SAMPLE_COLUMNS = DATA_COLUMNS + LABEL_COLUMNS


if __name__ == "__main__":
    from config import DATASET_PATH
    import time
    from pprint import pprint

    path = DATASET_PATH.format(language="english", name="RAVDESS", form="mfcc")

    start = time.time()
    ravdess = RAVDESS(path)
    print(ravdess.samples)
    # labels = np.array(ravdess.samples[ravdess.Sample.LABEL_INDEX])
    # print(labels)
    # emotions = list(labels[:, ravdess.Label.EMOTION_INDEX])
    # print(emotions)
    # pprint(list(map(ravdess.Label.EMOTION_OPTIONS.get, emotions)))
    # end = time.time()
    # print(end - start)

    print(len(ravdess.samples['data'][0][0]))








