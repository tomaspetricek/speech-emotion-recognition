from processing.convertors import AudioFormatConverter, MFCCConverter
from classes import Dataset
from pprint import pprint
import numpy as np
from enums import DATASET_PATH
from files import TextFile, HTKFile
from utils import change_file_extension


if __name__ == "__main__":

    language_ = "english"
    name_ = "RAVDESS"

    original_dataset = Dataset(
        path=DATASET_PATH.format(
            language=language_,
            name=name_,
            form="converted"
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

    input_files = original_dataset.samples

    output_files_ = change_file_extension(input_files, ".mfcc_0_d_a")
    output_files = []
    for output_file in output_files_:
        output_files.append(output_file.replace("converted", "mfcc"))

    print(len(input_files))
    print(len(output_files))

    converter = MFCCConverter(
        input_files=input_files,
        output_files=output_files
    )

    converter.convert()



