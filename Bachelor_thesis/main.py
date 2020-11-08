from processing.convertors import AudioFormatConverter, MFCCConverter
from classes import Dataset
from libraries.PyHTK.HTK import HTKFile
from pprint import pprint
import numpy as np
import enums


if __name__ == "__main__":

    input_file = "/Users/tomaspetricek/TUL/TUL_2020:21/BP/Speech_Emotion_Recognition/Test/03-01-01-01-01-01-01.wav"
    output_file = "/Users/tomaspetricek/TUL/TUL_2020:21/BP/Speech_Emotion_Recognition/Test/03-01-01-01-01-01-01.mfc"

    # MFCCConverter(input_files=[input_file], output_files=[output_file]).convert()
    htk_file = HTKFile()
    htk_file.load(filename=output_file)

    data = np.array(htk_file.data)
    print(data.shape)
    print(data[300])

    # language_ = "italian"
    # name_ = "EMOVO"
    #
    # original_dataset = Dataset(
    #     path=DATASET_PATH.format(
    #         language=language_,
    #         name=name_,
    #         form="original"
    #     )
    # )
    #
    # # # change dataset files permissions
    # # change_permissions(
    # #     files=original_dataset.samples,
    # #     permission=755
    # # )
    #
    # # samples_ = original_dataset.samples
    # # for sample in samples_:
    # #     print(sample)
    # #
    # # print(len(samples_))
    #
    # # converted_dataset = original_dataset.clone(
    # #     clone_path=DATASET_PATH.format(
    # #         language=language_,
    # #         name=name_,
    # #         form="converted"
    # #     )
    # # )
    #
    # converted_dataset = Dataset(
    #     path=DATASET_PATH.format(
    #         language=language_,
    #         name=name_,
    #         form="converted"
    #     )
    # )
    #
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

