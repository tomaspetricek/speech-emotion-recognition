from preprocessing.convertors import AudioFormatConverter
from utils import *

from timeit import default_timer as timer

if __name__ == "__main__":
    # copy_directory_content(
    #     source="/Users/tomaspetricek/TUL/TUL_2020:21/BP/Speech_Emotion_Recognition/Datasets/RAVDESS/raw",
    #     destination="/Users/tomaspetricek/TUL/TUL_2020:21/BP/Speech_Emotion_Recognition/Datasets/RAVDESS/test",
    # )

    raw_file_paths = get_file_paths(
        directory="/Users/tomaspetricek/TUL/TUL_2020:21/BP/"
                  "Speech_Emotion_Recognition/Datasets/RAVDESS/raw",
        file_extensions=[WAV]
    )

    # for file_name in raw_file_paths:
    #     print(file_name)
    #
    # print(len(raw_file_paths))

    converted_file_paths = get_file_paths(
        directory="/Users/tomaspetricek/TUL/TUL_2020:21/BP/"
                  "Speech_Emotion_Recognition/Datasets/RAVDESS/converted",
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
    print("time: {:>8.2f}s".format(end - start))    # time:    54.56s

