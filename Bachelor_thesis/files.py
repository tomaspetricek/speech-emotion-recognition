from libraries.PyHTK.HTK import HTKFile as PyHTKFile
# from .libraries.PyHTK.HTK import HTKFile as PyHTKFile # for pytest
from scipy.io import wavfile
import os, shutil
import subprocess


class Directory:

    def __init__(self, path):
        self.path = path
        self.file_paths = None

    def set_file_paths(self, value):
        """
        Gets all files with the same file_extension in a directory

        Source: https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
        """
        self._file_paths = []

        # walk the tree.
        for root, directories, files in os.walk(self.path):
            for file_name in files:
                _, file_extension = os.path.splitext(file_name)
                if file_extension != "":
                    # join the two strings in order to form the full file_path.
                    path = os.path.join(root, file_name)
                    # add it to the list.
                    self._file_paths.append(path)

    def get_file_paths(self):
        return self._file_paths

    file_paths = property(get_file_paths, set_file_paths)

    def copy_content(self, destination_dir, symlinks=False):
        """
        Copies the content of source dictionary to destination dictionary

        Source: https://stackoverflow.com/questions/1868714/how-do-i-copy-an-entire-directory-of-files-into-an-existing
        -directory-using-pyth
        """
        self._copy_content(self.path, destination_dir, symlinks)

    def _copy_content(self, source_dir, destination_dir=None, symlinks=False):
        # check if destination directory exists
        if not os.path.exists(destination_dir):
            # create destination directory
            os.makedirs(destination_dir)
        # go through content of source dictionary
        for item in os.listdir(source_dir):
            source_ = os.path.join(source_dir, item)
            destination_ = os.path.join(destination_dir, item)
            # check if item is directory
            if os.path.isdir(source_):
                # recursion
                self._copy_content(source_, destination_, symlinks)
            else:
                if not os.path.exists(destination_) or os.stat(source_).st_mtime - os.stat(
                        destination_).st_mtime > 1:
                    # copy file
                    shutil.copy2(source_, destination_)

    def copy_structure(self, destination_dir):
        """
        Copies directory inner structure (directories only).
        """
        def ignore_files(directory, files):
            return [f for f in files if os.path.isfile(os.path.join(directory, f))]

        shutil.copytree(self.path, destination_dir, ignore=ignore_files)


class File:
    def __init__(self, path):
        self.path = path

    def read(self):
        pass

    def write(self, data):
        pass

    def change_permissions(self, permission=755):
        """
        Changes files permissions.
        """
        COMMAND = "chmod {permission} {file}"

        command = COMMAND.format(
            permission=permission,
            file=self.path
        )
        subprocess.run(
            command,
            shell=True,  # True when command is a string
            check=True,  # True when we want to stop when error occurs
            capture_output=True,  # True when we want to capture output
            text=True  # get output as a string
        )

    def change_extension(self, new_extension):
        """
        Changes files extension.
        """
        base, ext = os.path.splitext(self.path)
        # os.rename(input_file, base + extension)
        output_file = base + new_extension
        return output_file


class HTKFile(File):

    def __init__(self, path):
        super().__init__(path)
        self.data = None

    def read(self):
        htk_file = PyHTKFile()
        htk_file.load(filename=self.path)
        self.data = htk_file.data
        return self.data

class WAVFile(File):

    def __init__(self, path):
        super().__init__(path)
        self.data = None
        self.sample_rate = None

    def read(self):
        self.sample_rate, self.data = wavfile.read(self.path)
        return self.sample_rate, self.data

class TextFile(File):
    """
    Represents text file.
    """

    def __init__(self, path, encoding="utf-8"):
        super().__init__(path)
        self.encoding = encoding
        self.data = None

    def read(self):
        self.data = None

        try:
            with open(self.path, "r", encoding=self.encoding) as text_file:
                self.data = text_file.read()
        except Exception:
            raise FileNotFoundError("Could not find file.")

        return self.data

    def read_lines(self):
        """
        Reads data in lines.
        """
        lines = []

        try:
            with open(self.path, mode="r", encoding=self.encoding) as text_file:
                for line in text_file:
                    lines.append(line.strip())
        except Exception:
            raise FileNotFoundError("Could not find file.")

        return lines

    def write(self, data):
        try:
            with open(self.path, "w", encoding=self.encoding) as text_file:
                text_file.write(data)
        except Exception:
            raise FileNotFoundError("Could not find file.")

    def write_lines(self, lines):
        """
        Writes data into a file line by line.
        """
        try:
            with open(self.path, "w", encoding=self.encoding) as text_file:
                for line in lines:
                    # save line
                    text_file.write(line + "\n")
        except Exception:
            raise FileNotFoundError("Could not find file.")


class DatasetInfoFile(TextFile):
    def read(self):
        self.data = None

        try:
            with open(self.path, "r", encoding=self.encoding) as text_file:
                n_chunks = int(text_file.readline())

                chunk_sizes = []
                for _ in range(n_chunks):
                    chunk_sizes.append(int(text_file.readline().strip()))

                samples_filenames = []
                for _ in range(n_chunks):
                    samples_filenames.append(text_file.readline().strip())

                labels_filenames = []
                for _ in range(n_chunks):
                    labels_filenames.append(text_file.readline().strip())

        except Exception:
            raise FileNotFoundError("Could not find file.")

        return chunk_sizes, samples_filenames, labels_filenames

