from processing.convertors import AudioFormatConverter, MFCCConverter
from classes import Dataset
from libraries.PyHTK.HTK import HTKFile
from pprint import pprint
import numpy as np
import enums
from files import TextFile


if __name__ == "__main__":

    input_file = "/Users/tomaspetricek/TUL/TUL_2020:21/BP/Speech_Emotion_Recognition/Test/03-01-01-01-01-01-01.wav"
    output_file = "/Users/tomaspetricek/TUL/TUL_2020:21/BP/Speech_Emotion_Recognition/Test/03-01-01-01-01-01-01.mfc"

    # MFCCConverter(input_files=[input_file], output_files=[output_file]).convert()

    # load HTK file
    htk_file = HTKFile()
    htk_file.load(filename=output_file)
    # print(htk_file.nSamples)
    # print(htk_file.sampPeriod)
    # # print(htk_file.basicKind)

    # get data
    data = np.array(htk_file.data).flatten()
    print(data.shape)

    # compare to test file
    test_filename = "/Users/tomaspetricek/TUL/TUL_2020:21/BP/Speech_Emotion_Recognition/Test/_mfcc_kontrola/03-01-01-01-01-01-01.mfcc_0_d_a.txt"

    test_file = TextFile(test_filename)

    test_data = test_file.read(skip_n_rows=4)
    # covert to list of float
    test_data = list(map(float, test_data))
    # convert to numpy array
    test_data = np.array(test_data)
    print(test_data.shape)

    # check if arrays are totally equal
    print(np.array_equal(data, test_data))

    # print them out next to each other
    for data, test_data in zip(data, test_data):
        print(data, test_data)

    # data = np.array(htk_file.data)
    # print(data.shape)
    # print(data[300])

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

