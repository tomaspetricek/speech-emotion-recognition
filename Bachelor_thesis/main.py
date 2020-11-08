from processing.convertors import AudioFormatConverter, MFCCConverter
from classes import Dataset
from pprint import pprint
import numpy as np
from enums import DATASET_PATH
from files import TextFile, HTKFile
from utils import change_file_extension


if __name__ == "__main__":

    input_file = "/Users/tomaspetricek/TUL/TUL_2020:21/BP/Speech_Emotion_Recognition/Test/03-01-01-01-01-01-01.wav"
    output_file = "/Users/tomaspetricek/TUL/TUL_2020:21/BP/Speech_Emotion_Recognition/Test/03-01-01-01-01-01-01.mfc"

    # MFCCConverter(input_files=[input_file], output_files=[output_file]).convert()

    language_ = "italian"
    name_ = "EMOVO"

    original_dataset = Dataset(
        path=DATASET_PATH.format(
            language=language_,
            name=name_,
            form="original"
        )
    )

    # converted_dataset = original_dataset.clone(
    #     clone_path=DATASET_PATH.format(
    #         language=language_,
    #         name=name_,
    #         form="mfcc",
    #     ),
    #     ignore_file_extensions=['.wav']
    # )
    #
    # print(converted_dataset.samples)

    input_files = original_dataset.samples
    output_files = change_file_extension(input_files, ".mfcc_0_d_a")
    print(output_files)

    # converter = AudioFormatConverter(
    #     input_files=input_files,
    #     output_files=output_files,
    #     audio_channel=AudioFormatConverter.MONO,
    #     sample_rate=AudioFormatConverter._16KHz
    # )

    # # start = timer()
    # converter.convert()
    # # end = timer()
    # # print("time: {:>8.2f}s".format(end - start))    # RAVDESS time: 54.56s, SAVEE time: 23.40s, TESS time: 108.75s


