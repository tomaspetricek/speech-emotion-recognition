from processing.convertors import AudioFormatConverter, MFCCConverter
from os import walk
import os

def get_file_paths(input_dir_path, output_dir_path, output_extension):
    _, _, input_filenames = next(walk(input_dir_path))
    input_filenames = sorted(input_filenames)

    input_file_paths = []
    output_file_paths = []
    for input_filename in input_filenames:
        input_file_paths.append(os.path.join(input_dir_path, input_filename))

        # change extension
        base, ext = os.path.splitext(input_filename)
        output_filename = base + output_extension

        output_file_paths.append(os.path.join(output_dir_path, output_filename))

    return input_file_paths, output_file_paths

def main():
    input_dir_path = "/Users/tomaspetricek/TUL/TUL_2020:21/BP/Speech_Emotion_Recognition/Datasets/czech/CallCenters/converted"

    output_dir_path = "/Users/tomaspetricek/TUL/TUL_2020:21/BP/Speech_Emotion_Recognition/Datasets/czech/CallCenters/mfcc"

    output_extension = ".mfcc_0_d_a"
    input_file_paths, output_file_paths = get_file_paths(input_dir_path, output_dir_path, output_extension)

    # convertor = AudioFormatConverter(
    #     input_file_paths,
    #     output_file_paths,
    #     audio_channel=AudioFormatConverter.MONO,
    #     sample_rate=AudioFormatConverter._16KHz
    # )

    convertor = MFCCConverter(
        input_file_paths,
        output_file_paths
    )

    convertor.convert()


if __name__ == "__main__":
    main()
