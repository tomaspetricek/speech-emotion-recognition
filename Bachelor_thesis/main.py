from preprocessing.convertors import AudioFormatConverter
from utils import *
from timeit import default_timer as timer

if __name__ == "__main__":

    raw_directory_path = "/Users/tomaspetricek/TUL/TUL_2020:21/BP/Speech_Emotion_Recognition/Datasets/TESS/raw"
    converted_directory_path = "/Users/tomaspetricek/TUL/TUL_2020:21/BP/Speech_Emotion_Recognition/Datasets/TESS/converted"

    # # copy directory
    # copy_directory_content(
    #     source=raw_directory_path,
    #     destination=converted_directory_path,
    # )

    raw_file_paths = get_file_paths(
        directory=raw_directory_path,
        file_extensions=[WAV]
    )

    # for file_name in raw_file_paths:
    #     print(file_name)
    #
    # print(len(raw_file_paths))

    converted_file_paths = get_file_paths(
        directory=converted_directory_path,
        file_extensions=[WAV]
    )

    # for file_name in converted_file_paths:
    #     print(file_name)
    #
    # print(len(converted_file_paths))

    input_files = raw_file_paths
    output_files = converted_file_paths

    converter = AudioFormatConverter(
        input_files=input_files,
        output_files=output_files,
        audio_channel=AudioFormatConverter.MONO,
        sample_rate=AudioFormatConverter._16KHz
    )

    start = timer()
    converter.convert()
    end = timer()
    print("time: {:>8.2f}s".format(end - start))    # RAVDESS time: 54.56s, SAVEE time: 23.40s, TESS time: 108.75s

