from processing.convertors import AudioFormatConverter, MFCCConverter
from classes import Dataset
from pprint import pprint
import numpy as np
from enums import DATASET_PATH
from files import TextFile, HTKFile
from utils import change_file_extension
import re

if __name__ == "__main__":
    # path = "/Users/tomaspetricek/TUL/TUL_2020:21/BP/Speech_Emotion_Recognition/Datasets/english/RAVDESS/mfcc" \
    #        "/Audio_Speech_Actors_01-24/Actor_01/03-01-01-01-01-01-01.mfcc_0_d_a"
    #
    # htk_file = HTKFile(filename=path)
    # htk_file.read()
    # data = np.array(htk_file.data).flatten()
    #
    # print(len(data))
    # for d in data:
    #     print(d)

    test = "03-01-01-01-01-01-01"
    "modality"
    "vocal_channel"
    "emotion"
    "emotional_intensity"
    "statement"
    "repetition"
    "actor"

    RAVDESS_LABEL_REGEX = re.compile(r'(?P<modality>\d+)-(?P<vocal_channel>\d+)-(?P<emotion>\d+)-(?P<emotional_intensity>\d+)-(?P<statement>\d+)-(?P<repetition>\d+)-(?P<actor>\d+)')

    match = RAVDESS_LABEL_REGEX.match(test)
    print(match.groupdict())
    print(list(map(int, match.groups())))


