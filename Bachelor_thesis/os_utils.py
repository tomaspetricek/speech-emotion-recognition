import os, shutil
import subprocess

WAV = ".wav"

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


class File:
    def __init__(self, path):
        self.path = path

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

