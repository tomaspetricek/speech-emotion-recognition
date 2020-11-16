from processing.convertors import AudioFormatConverter, MFCCConverter
from classes import Dataset
from pprint import pprint
import numpy as np
from enums import DATASET_PATH
from files import TextFile, HTKFile
from utils import change_file_extension


if __name__ == "__main__":
    path = "/Users/tomaspetricek/TUL/TUL_2020:21/BP/Speech_Emotion_Recognition/Datasets/english/RAVDESS/mfcc" \
           "/Audio_Speech_Actors_01-24/Actor_01/03-01-01-01-01-01-01.mfcc_0_d_a"

    htk_file = HTKFile(filename=path)
    htk_file.read()
    data = np.array(htk_file.data).flatten()

    print(len(data))
    for d in data:
        print(d)


