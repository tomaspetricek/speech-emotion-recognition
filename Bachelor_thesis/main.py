from preprocessing.convertors import AudioFormatConverter
from utils import *
import subprocess

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


if __name__ == "__main__":

    language_ = "italian"
    name_ = "EMOVO"

    original_dataset = Dataset(
        path=DATASET_PATH.format(
            language=language_,
            name=name_,
            form="original"
        )
    )

    # # change dataset files permissions
    # change_permissions(
    #     files=original_dataset.samples,
    #     permission=755
    # )

    # samples_ = original_dataset.samples
    # for sample in samples_:
    #     print(sample)
    #
    # print(len(samples_))

    # converted_dataset = original_dataset.clone(
    #     clone_path=DATASET_PATH.format(
    #         language=language_,
    #         name=name_,
    #         form="converted"
    #     )
    # )

    # converted_dataset = Dataset(
    #     path=DATASET_PATH.format(
    #         language=language_,
    #         name=name_,
    #         form="converted"
    #     )
    # )

    # converter = AudioFormatConverter(
    #     input_files=original_dataset.samples,
    #     output_files=converted_dataset.samples,
    #     audio_channel=AudioFormatConverter.MONO,
    #     sample_rate=AudioFormatConverter._16KHz
    # )
    #
    # # start = timer()
    # converter.convert()
    # # end = timer()
    # # print("time: {:>8.2f}s".format(end - start))    # RAVDESS time: 54.56s, SAVEE time: 23.40s, TESS time: 108.75s

