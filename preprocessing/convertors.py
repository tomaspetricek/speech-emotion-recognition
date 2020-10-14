import os
from functools import partial

class AudioFormatConverter(object):
    COMMAND = "ffmpeg -i {input_file} -ar {sample_rate} -ac {audio_channel} -y {output_file}"

    # sample rate options
    SAMPLE_RATE_16KHz = 16000

    # audio channel channel options
    AUDIO_CHANNEL_MONO = 1

    def __init__(self, input_files, output_files, audio_channel, sample_rate):
        self.input_files = input_files
        self.output_files = output_files
        self.audio_channel = audio_channel
        self.sample_rate = sample_rate
        self.command = partial(self.COMMAND.format, sample_rate=self.sample_rate, audio_channel=self.audio_channel)

    def run(self):
        for input_file, output_file in zip(self.input_files, self.output_files):
            command = self.command(input_file=input_file, output_file=output_file)
            os.system(command)
