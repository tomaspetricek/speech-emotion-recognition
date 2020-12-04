import sys
import os
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from Bachelor_thesis.datasets import TESSLabel


@pytest.mark.parametrize('path, expected_label', [
    ("/Users/tomaspetricek/TUL/TUL_2020:21/BP/Speech_Emotion_Recognition/Datasets/english/TESS/mfcc/OAF_angry/OAF_back_angry.mfcc_0_d_a", ['OAF', 'back', 'angry']),
    ("/Users/tomaspetricek/TUL/TUL_2020:21/BP/Speech_Emotion_Recognition/Datasets/english/TESS/mfcc/OAF_Pleasant_surprise/OAF_back_ps.mfcc_0_d_a", ['OAF', 'back', 'ps']),
])
def test_TESSLabel_parse(path, expected_label):
    label = TESSLabel()
    assert expected_label == label.parse(path)


