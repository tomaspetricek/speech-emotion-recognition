import subprocess
"""
subprocess tutorial link: https://youtu.be/2Fp1N6dof0Y
"""
from functools import partial

EXECUTION_SUCCESSFUL = 0

class AudioFormatConverter(object):
    """
    Represents an audio format converter that
        1. reads in an input file
        2. changes sample rate and audio channel
        3. saves it to an output file
    """
    COMMAND = "ffmpeg -i {input_file} -ar {sample_rate} -ac {audio_channel} -y {output_file}"

    # sample rate options
    _16KHz = 16000

    # audio channel channel options
    MONO = 1

    def __init__(self, input_files, output_files, audio_channel, sample_rate):
        self.input_files = input_files
        self.output_files = output_files
        self.audio_channel = audio_channel
        self.sample_rate = sample_rate
        # set command params that are the same for all
        self.command = partial(self.COMMAND.format, sample_rate=self.sample_rate, audio_channel=self.audio_channel)

    def convert(self):
        """
        Converts input files.
        """

        for input_file, output_file in zip(self.input_files, self.output_files):
            # complete command
            command = self.command(input_file=input_file, output_file=output_file)
            # execute command
            execution = subprocess.run(
                command,
                shell=True,  # True when command is a string
                check=True,  # True when we want to stop when error occurs
                capture_output=True,  # True when we want to capture output
                text=True   # get output as a string
            )
