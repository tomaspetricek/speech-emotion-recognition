from os_utils import Directory
import pandas as pd
import numpy as np
from files import HTKFile, WAVFile
import re
import os


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
        pass

class RAVDESSLabel(Label):
    SEPARATOR = "-"

    COLUMNS = [
        "modality",
        "vocal channel",
        "emotion",
        "emotional intensity",
        "statement",
        "repetition",
        "speaker"
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

    def parse(self, file_path):
        filename = os.path.basename(file_path)
        name, ext = os.path.splitext(filename)
        label = name.split(self.SEPARATOR)
        return list(map(int, label))

class SAVEELabel(Label):
    SEPARATOR = None
    COLUMNS = [
        "speaker",
        "emotion",
        "statement",
    ]

    EMOTION_OPTIONS = {
        "a": "anger",
        "d": "disgust",
        "f": "fear",
        "h": "happiness",
        "n": "neutral",
        "sa": "sadness",
        "su": "surprise",
    }

    _REGEX = re.compile(r"(?P<emotion>[A-Za-z]+)(?P<statement>\d+)")

    def parse(self, file_path):
        path = os.path.normpath(file_path)
        path_separated = path.split(os.sep)
        speaker, filename = tuple(path_separated[-2:])
        name, ext = os.path.splitext(filename)
        result = self._REGEX.match(name)
        groups = result.groupdict()
        return [speaker, groups["emotion"], groups["statement"]]

class TESSLabel(Label):
    SEPARATOR = "_"
    COLUMNS = [
        "speaker",
        "statement",
        "emotion",
    ]

    def parse(self, file_path):
        filename = os.path.basename(file_path)
        name, ext = os.path.splitext(filename)
        label = name.split(self.SEPARATOR)
        return label

class EMOVOLabel(Label):
    SEPARATOR = "-"
    COLUMNS = [
        "emotion",
        "speaker",
        "sentence type",
    ]

    EMOTIONS_OPTIONS = {
        "dis": "disgust",
        "gio": "joy",
        "pau": "fear",
        "rab": "anger",
        "sor": "surprise",
        "tri": "sadness",
        "neu": "neutral",
    }

    def parse(self, file_path):
        filename = os.path.basename(file_path)
        name, ext = os.path.splitext(filename)
        label = name.split(self.SEPARATOR)
        return label

class Dataset(Directory):
    """
    Represents a dataset.
    """
    def __init__(self, path, data, label):
        super().__init__(path)
        self.data = data
        self.label = label
        self.data_columns = []
        self.label_columns = []
        self.sample_columns = None
        self.samples = None

    def set_sample_columns(self, value):
        if self.data:
            self.data_columns = self.data.COLUMNS

        if self.label:
            self.label_columns = self.label.COLUMNS

        self._sample_columns = self.data_columns + self.label_columns

    def get_sample_columns(self):
        return self._sample_columns

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
    samples = property(get_samples, set_samples)


if __name__ == "__main__":
    # path_emovo = "/Users/tomaspetricek/TUL/TUL_2020:21/BP/Speech_Emotion_Recognition/Datasets/italian/EMOVO/mfcc/f1/dis-f1-b1.mfcc_0_d_a"
    #
    # print(EMOVOLabel().parse(path_emovo))
    #
    # path_tess = "/Users/tomaspetricek/TUL/TUL_2020:21/BP/Speech_Emotion_Recognition/Datasets/english/TESS/mfcc/OAF_angry/OAF_back_angry.mfcc_0_d_a"
    #
    # print(TESSLabel().parse(path_tess))







