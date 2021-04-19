import pandas as pd
import numpy as np

from files import HTKFile, WAVFile, Directory
# from .files import HTKFile, WAVFile, Directory for pytest
import re
import os

NEUTRAL = 0
ANGER = 1
FEAR = 2
SADNESS = 3
HAPPINESS = 4
DISGUST = 5
SURPRISE = 6

ALL_EMOTIONS_VERBOSE = [
    "neutrální",
    "hněv",
    "strach",
    "smutek",
    "spokojenost",
    "odpor",
    "překvapení",
]
neutral = 0
anger = 1
happiness = 2
sadness = 3

FOUR_EMOTIONS_CONVERSION_TABLE = {
    NEUTRAL: neutral,
    ANGER: anger,
    DISGUST: anger,
    HAPPINESS: happiness,
    SURPRISE: happiness,
    SADNESS: sadness,
    FEAR: sadness,
}

FOUR_EMOTIONS_VERBOSE = [
    "neutrální",
    "hněv",
    "spokojenost",
    "smutek",
]

neutral_ = 0
positive = 1
negative = 2

THREE_EMOTIONS_CONVERSION_TABLE = {
    NEUTRAL: neutral_,
    ANGER: negative,
    DISGUST: negative,
    HAPPINESS: positive,
    SURPRISE: positive,
    SADNESS: negative,
    FEAR: negative,
}

THREE_EMOTIONS_VERBOSE = [
    "neutrální",
    "pozitivní",
    "negativní",
]


class Data:
    FILE = None
    COLUMNS = [
        "data"
    ]

    def load(self, file_path):
        pass


class MFCCData(Data):
    FILE = HTKFile
    SAMPLE_FORMAT = ".mfcc_0_d_a"
    COLUMNS = [
        "coefficients",
    ]

    def load(self, file_path):
        return self.FILE(file_path).read()


class WAVData(Data):
    SAMPLE_FORMAT = ".wav"
    FILE = WAVFile

    COLUMNS = [
        "sample rate",
        "data"
    ]

    def load(self, file_path):
        sample_rate, data = self.FILE(file_path).read()
        return [sample_rate, data]


class Label:
    COLUMNS = []
    SEPARATOR = None

    def parse(self, file_path):
        pass


class UnifiedLabel(Label):
    COLUMNS = [
        "emotion"
    ]

    EMOTION_OPTIONS = {
        NEUTRAL: "neutral",
        ANGER: "anger",
        FEAR: "fear",
        SADNESS: "sadness",
        HAPPINESS: "happiness",
        DISGUST: "disgust",
        SURPRISE: "surprise",
    }

    EMOTION_CONVERSION = None


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


class RAVDESSUnifiedLabel(UnifiedLabel):
    EMOTION_CONVERSION = {
        1: NEUTRAL,
        2: NEUTRAL,  # CALM to NEUTRAL
        3: HAPPINESS,
        4: SADNESS,
        5: ANGER,
        6: FEAR,
        7: DISGUST,
        8: SURPRISE,
    }

    def parse(self, file_path):
        label = RAVDESSLabel().parse(file_path)
        emotion = label[2]
        return [self.EMOTION_CONVERSION[emotion]]


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


class SAVEEUnifiedLabel(UnifiedLabel):
    EMOTION_CONVERSION = {
        "a": ANGER,
        "d": DISGUST,
        "f": FEAR,
        "h": HAPPINESS,
        "n": NEUTRAL,
        "sa": SADNESS,
        "su": SURPRISE,
    }

    def parse(self, file_path):
        label = SAVEELabel().parse(file_path)
        emotion = label[1]
        return [self.EMOTION_CONVERSION[emotion]]


class TESSLabel(Label):  # TODO Correct labeling Pleasant_surprise
    SEPARATOR = "_"
    COLUMNS = [
        "speaker",
        "statement",
        "emotion",
    ]

    EMOTIONS_OPTIONS = {
        "fear": "",
        "ps": "",
        "angry": "",
        "disgust": "",
        "happy": "",
        "neutral": "",
        "sad": "",
    }

    def parse(self, file_path):
        filename = os.path.basename(file_path)
        name, ext = os.path.splitext(filename)
        label = name.split(self.SEPARATOR)
        return label


class TESSUnifiedLabel(UnifiedLabel):
    EMOTION_CONVERSION = {
        "fear": FEAR,
        "ps": SURPRISE,
        "angry": ANGER,
        "disgust": DISGUST,
        "happy": HAPPINESS,
        "neutral": NEUTRAL,
        "sad": SADNESS,
    }

    def parse(self, file_path):
        label = TESSLabel().parse(file_path)
        emotion = label[2]
        return [self.EMOTION_CONVERSION[emotion]]


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


class EMOVOUnifiedLabel(UnifiedLabel):
    EMOTION_CONVERSION = {
        "dis": DISGUST,
        "gio": HAPPINESS,
        "pau": FEAR,
        "rab": ANGER,
        "sor": SURPRISE,
        "tri": SADNESS,
        "neu": NEUTRAL,
    }

    def parse(self, file_path):
        label = EMOVOLabel().parse(file_path)
        emotion = label[0]
        return [self.EMOTION_CONVERSION[emotion]]


class CallCentersUnifiedLabel(UnifiedLabel):
    SEPARATOR = "_"

    def parse(self, file_path):
        filename = os.path.basename(file_path)
        name, ext = os.path.splitext(filename)
        label = name.split(self.SEPARATOR)
        emotion = label[1]
        return int(emotion)


class Dataset(Directory):
    """
    Represents a dataset.
    """

    def __init__(self, path, data, label):
        super().__init__(path)
        self.data = data
        self.label = label
        self.samples = None
        self.labels = None

    def set_samples(self, value):
        self._samples = []

        for file_path in self._file_paths:
            sample = self.data.load(file_path)
            sample = np.array(sample)
            self._samples.append(sample)

    def get_samples(self):
        return self._samples

    def set_labels(self, value):
        self._labels = []

        for file_path in self._file_paths:
            label = self.label.parse(file_path)
            self._labels.append(label)

    def get_labels(self):
        return self._labels

    samples = property(get_samples, set_samples)
    labels = property(get_labels, set_labels)

    def combine(self, *datasets):
        for dataset in datasets:
            self._samples += dataset.samples
            self._labels += dataset.labels
            self._file_paths += dataset.file_paths


if __name__ == "__main__":
    ravdess_path = "/Users/tomaspetricek/TUL/TUL_2020:21/BP/Speech_Emotion_Recognition/Datasets/english/RAVDESS/converted/Audio_Speech_Actors_01-24/Actor_01/03-01-01-01-01-01-01.wav"
    savee_path = "/Users/tomaspetricek/TUL/TUL_2020:21/BP/Speech_Emotion_Recognition/Datasets/english/SAVEE/converted/DC/a01.wav"
    tess_path = "/Users/tomaspetricek/TUL/TUL_2020:21/BP/Speech_Emotion_Recognition/Datasets/english/TESS/mfcc/OAF_angry/OAF_back_angry.wav"
    emovo_path = "/Users/tomaspetricek/TUL/TUL_2020:21/BP/Speech_Emotion_Recognition/Datasets/italian/EMOVO/mfcc/f1/dis-f1-b1.wav"

    print(EMOVOUnifiedLabel().parse(emovo_path))
