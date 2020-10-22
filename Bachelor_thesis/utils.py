import os, shutil
import subprocess

WAV = ".wav"


def get_file_paths(directory, file_extensions=None):
    """
    Gets all files with the same file_extension in a directory

    Source: https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
    """
    if file_extensions is None:
        file_extensions = []

    file_paths = []

    # walk the tree.
    for root, directories, files in os.walk(directory):
        for file_name in files:
            _, file_extension = os.path.splitext(file_name)
            if file_extension in file_extensions:
                # join the two strings in order to form the full file_path.
                file_path = os.path.join(root, file_name)
                # add it to the list.
                file_paths.append(file_path)

    return file_paths


def copy_directory_content(source, destination, symlinks=False, ignore_file_extensions=None):
    """
    Copies the content of source dictionary to destination dictionary

    Source: https://stackoverflow.com/questions/1868714/how-do-i-copy-an-entire-directory-of-files-into-an-existing
    -directory-using-pyth
    """
    if ignore_file_extensions is None:
        ignore_file_extensions = []

    # check if destination directory exists
    if not os.path.exists(destination):
        # create destination directory
        os.makedirs(destination)
    # go through content of source dictionary
    for item in os.listdir(source):
        source_ = os.path.join(source, item)
        destination_ = os.path.join(destination, item)
        # check if item is directory
        if os.path.isdir(source_):
            # recursion
            copy_directory_content(source_, destination_, symlinks, ignore_file_extensions)
        else:
            filename, file_extension = os.path.splitext(destination_)
            if file_extension not in ignore_file_extensions:
                if not os.path.exists(destination_) or os.stat(source_).st_mtime - os.stat(destination_).st_mtime > 1:
                    # copy file
                    shutil.copy2(source_, destination_)

def change_permissions(files, permission=755):
    """
    Changes files permissions.
    """
    COMMAND = "chmod {permission} {file}"

    for file in files:
        command = COMMAND.format(
            permission=permission,
            file=file
        )
        subprocess.run(
            command,
            shell=True,  # True when command is a string
            check=True,  # True when we want to stop when error occurs
            capture_output=True,  # True when we want to capture output
            text=True  # get output as a string
        )
