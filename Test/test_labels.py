import sys
import os
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from Bachelor_thesis.datasets import TESSLabel

print(TESSLabel)

def test_TESSLabel_parse():
    pass

