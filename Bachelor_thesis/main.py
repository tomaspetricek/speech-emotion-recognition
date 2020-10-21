from preprocessing.convertors import AudioFormatConverter
from utils import *

DATASET_PATH = "/Users/tomaspetricek/TUL/TUL_2020:21/BP/Speech_Emotion_Recognition/Datasets/{language}/{name}/{" \
                   "form}"


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

        copy_directory_content(
            source=self.path,
            destination=clone_path,
        )

        clone_dataset = Dataset(
            path=clone_path,
            sample_format=self.sample_format
        )

        return clone_dataset


if __name__ == "__main__":

    language_ = "english"
    name_ = "TESS"

    original_dataset = Dataset(
        path=DATASET_PATH.format(
            language=language_,
            name=name_,
            form="original"
        )
    )

    for sample in original_dataset.samples:
        print(sample)

    # test_dataset = original_dataset.clone(
    #     clone_path=Dataset.DATASET_PATH.format(
    #         language=language_,
    #         name=name_,
    #         form="test"
    #     )
    # )

    # converted_dataset = Dataset(
    #     path=DATASET_PATH.format(
    #         language=language_,
    #         name=name_,
    #         form="converted"
    #     )
    # )
    #
    # converter = AudioFormatConverter(
    #     input_files=input_files,
    #     output_files=output_files,
    #     audio_channel=AudioFormatConverter.MONO,
    #     sample_rate=AudioFormatConverter._16KHz
    # )
    #
    # # start = timer()
    # converter.convert()
    # # end = timer()
    # # print("time: {:>8.2f}s".format(end - start))    # RAVDESS time: 54.56s, SAVEE time: 23.40s, TESS time: 108.75s

