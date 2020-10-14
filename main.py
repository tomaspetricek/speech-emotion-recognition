from preprocessing.convertors import AudioFormatConverter

"""
Clone directory
https://stackoverflow.com/questions/40828450/how-to-copy-folder-structure-under-another-directory
"""


if __name__ == "__main__":

    input_directory_name = "raw"
    output_directory_name = "converted"

    path = "/Users/tomaspetricek/TUL/TUL_2020:21/BP/Speech_Emotion_Recognition/Datasets/RAVDESS/" \
           "{directory_name}/Audio_Speech_Actors_01-24/Actor_01/03-01-01-01-01-01-01.wav"

    input_file = path.format(directory_name=input_directory_name)
    output_file = path.format(directory_name=output_directory_name)

    input_files = [input_file]
    output_files = [output_file]

    converter = AudioFormatConverter(
        input_files=input_files,
        output_files=output_files,
        audio_channel=AudioFormatConverter.AUDIO_CHANNEL_MONO,
        sample_rate=AudioFormatConverter.SAMPLE_RATE_16KHz
    )

    converter.convert()





