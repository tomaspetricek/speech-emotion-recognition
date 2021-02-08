from processing.convertors import AudioFormatConverter, MFCCConverter
# from classes import Dataset
# from pprint import pprint
import numpy as np
# from enums import DATASET_PATH
# from files import TextFile, HTKFile
# from os_utils import change_file_extension
# import re


from processing.convertors import AudioFormatConverter
from os import walk
import os

from files import HTKFile


if __name__ == "__main__":
    file_path = "/Users/tomaspetricek/TUL/TUL_2020:21/BP/Speech_Emotion_Recognition/Datasets/czech/CallCenters/mfcc/001_1.mfcc_0_d_a"
    data = HTKFile(file_path).read()









