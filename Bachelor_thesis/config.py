import re


RAVDESS_LABEL_REGEX = re.compile(r'(?P<modality>\d+)-(?P<vocal_channel>\d+)-(?P<emotion>\d+)-(?P<emotional_intensity>\d+)-(?P<statement>\d+)-(?P<repetition>\d+)-(?P<actor>\d+)')