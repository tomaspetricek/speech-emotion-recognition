from preprocessing.convertors import AudioFormatConverter
from utils import *


if __name__ == "__main__":
    raw_file_paths = get_file_paths(
        directory="/Users/tomaspetricek/TUL/TUL_2020:21/BP/"
                  "Speech_Emotion_Recognition/Datasets/RAVDESS/raw",
        file_extension=FILE_EXTENSION_VAW
    )

    for file_name in raw_file_paths:
        print(file_name)

    print(len(raw_file_paths))


    # input_directory_name = "raw"
    # output_directory_name = "converted"
    #
    # path = "/Users/tomaspetricek/TUL/TUL_2020:21/BP/Speech_Emotion_Recognition/Datasets/RAVDESS/" \
    #        "{directory_name}/Audio_Speech_Actors_01-24/Actor_01/03-01-01-01-01-01-01.wav"
    #
    # input_file = path.format(directory_name=input_directory_name)
    # output_file = path.format(directory_name=output_directory_name)
    #
    # input_files = [input_file]
    # output_files = [output_file]
    #
    # converter = AudioFormatConverter(
    #     input_files=input_files,
    #     output_files=output_files,
    #     audio_channel=AudioFormatConverter.AUDIO_CHANNEL_MONO,
    #     sample_rate=AudioFormatConverter.SAMPLE_RATE_16KHz
    # )
    #
    # converter.convert()
