from utils import *
import re

class Dataset:
    """
    Represents a dataset.
    """

    class Sample:
        FILE_PATH_INDEX = 0
        DATA_INDEX = 1
        LABEL_INDEX = 2
        LENGTH = 3

    class Label:
        LENGTH = None

    class Data:
        pass

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
            sample_format=self.sample_format,
        )

        return clone_dataset


class RAVDESS(Dataset):
    class Label(Dataset.Label):
        REGEX = re.compile(r'(?P<modality>\d+)-(?P<vocal_channel>\d+)-(?P<emotion>\d+)-(?P<emotional_intensity>\d+)-'
                           r'(?P<statement>\d+)-(?P<repetition>\d+)-(?P<actor>\d+)')
        MODALITY_INDEX = 0
        VOCAL_CHANNEL_INDEX = 1
        EMOTION = 2
        EMOTIONAL_INTENSITY_INDEX = 3
        STATEMENT_INDEX = 4
        REPETITION_INDEX = 5
        ACTOR_INDEX = 6
        LENGTH = 7

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

    class Data(Dataset.Data):
        pass


if __name__ == "__main__":
    print(RAVDESS.Label.REGEX.match("03-01-01-01-01-01-01"))
    print(RAVDESS.Label.MODALITY_OPTIONS[1])
