from utils import *
import re
import pandas as pd
import numpy as np
from files import HTKFile, WAVFile


class Dataset:
    """
    Represents a dataset.
    """

    FILE = None
    SAMPLE_FORMAT = None

    LABEL_REGEX = None
    LABEL_SEPARATOR = None
    LABEL_COLUMNS = None

    DATA_COLUMNS = [
        "data"
    ]

    SAMPLE_COLUMNS = None

    def __init__(self, path):
        self.path = path
        self.file_paths = None
        self.samples = None

    def set_file_paths(self, value):
        self._file_paths = None

        if value is None:
            self._file_paths = get_file_paths(
                directory=self.path,
                file_extensions=[self.SAMPLE_FORMAT]
            )

    def get_file_paths(self):
        return self._file_paths

    def set_samples(self, value):
        samples = []
        for file_path in self._file_paths:
            data = []
            if self.DATA_COLUMNS:
                result = self.FILE.read(file_path)
                if type(result) is tuple:
                    data = [*result]
                else:
                    data = [result]

            label = []
            if self.LABEL_COLUMNS:
                filename = os.path.basename(file_path)
                name, ext = os.path.splitext(filename)
                label = name.split(self.LABEL_SEPARATOR)
                label = list(map(int, label))

            samples.append(data + label)

        self._samples = pd.DataFrame(samples, columns=self.SAMPLE_COLUMNS)

    def get_samples(self):
        return self._samples

    samples = property(get_samples, set_samples)
    file_paths = property(get_file_paths, set_file_paths)

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

class MFCC(Dataset):
    SAMPLE_FORMAT = ".mfcc_0_d_a"
    FILE = HTKFile()

    """
    https://www.sciencedirect.com/topics/computer-science/cepstral-coefficient
    """

    SAMPLE_COLUMNS = Dataset.DATA_COLUMNS


class WAV(Dataset):
    SAMPLE_FORMAT = ".wav"
    FILE = WAVFile()

    DATA_COLUMNS = [
        "sample rate",
        "data"
    ]

    SAMPLE_COLUMNS = DATA_COLUMNS

class RAVDESS(Dataset):
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

    SAMPLE_COLUMNS = LABEL_COLUMNS


class RAVDESS_MFCC(RAVDESS, MFCC):
    DATA_COLUMNS = MFCC.DATA_COLUMNS
    LABEL_COLUMNS = RAVDESS.LABEL_COLUMNS

    SAMPLE_COLUMNS = DATA_COLUMNS + LABEL_COLUMNS


class RAVDESS_WAV(RAVDESS, WAV):
    DATA_COLUMNS = WAV.DATA_COLUMNS
    LABEL_COLUMNS = RAVDESS.LABEL_COLUMNS

    SAMPLE_COLUMNS = DATA_COLUMNS + LABEL_COLUMNS


if __name__ == "__main__":
    from config import DATASET_PATH
    import time
    from pprint import pprint

    path = DATASET_PATH.format(language="english", name="RAVDESS", form="converted")

    ravdess_wav = RAVDESS_WAV(path)
    print(ravdess_wav.samples)

    path = DATASET_PATH.format(language="english", name="RAVDESS", form="mfcc")

    ravdess_mfcc = RAVDESS_MFCC(path)
    print(ravdess_mfcc.samples)
    # labels = np.array(ravdess.samples[ravdess.Sample.LABEL_INDEX])
    # print(labels)
    # emotions = list(labels[:, ravdess.Label.EMOTION_INDEX])
    # print(emotions)
    # pprint(list(map(ravdess.Label.EMOTION_OPTIONS.get, emotions)))
    # end = time.time()
    # print(end - start)

    # print(len(ravdess.samples['data'][0][0]))
    # print(ravdess.VOCAL_CHANNEL_OPTIONS)








