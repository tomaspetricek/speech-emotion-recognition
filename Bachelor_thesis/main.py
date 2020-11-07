from preprocessing.convertors import AudioFormatConverter
from classes import Dataset


DATASET_PATH = "/Users/tomaspetricek/TUL/TUL_2020:21/BP/Speech_Emotion_Recognition/Datasets/{language}/{name}/{" \
                   "form}"


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

    converted_dataset = Dataset(
        path=DATASET_PATH.format(
            language=language_,
            name=name_,
            form="converted"
        )
    )

    converter = AudioFormatConverter(
        input_files=original_dataset.samples,
        output_files=converted_dataset.samples,
        audio_channel=AudioFormatConverter.MONO,
        sample_rate=AudioFormatConverter._16KHz
    )

    # start = timer()
    converter.convert()
    # end = timer()
    # print("time: {:>8.2f}s".format(end - start))    # RAVDESS time: 54.56s, SAVEE time: 23.40s, TESS time: 108.75s

