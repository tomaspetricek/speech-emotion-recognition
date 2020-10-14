import os, shutil


FILE_EXTENSION_VAW = ".wav"
def get_file_paths(directory, file_extension):
    """
    Gets all files with the same file_extension in a directory

    Source: https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
    """
    file_paths = []

    # walk the tree.
    for root, directories, files in os.walk(directory):
        for file_name in files:
            # join the two strings in order to form the full file_path.
            if file_name.endswith(file_extension):
                file_path = os.path.join(root, file_name)
                file_paths.append(file_path)  # add it to the list.

    return file_paths


def copytree(source, destination, symlinks=False, ignore=None):
    """
    Copies the whole content of source dictionary to destination dictionary

    Source: https://stackoverflow.com/questions/1868714/how-do-i-copy-an-entire-directory-of-files-into-an-existing
    -directory-using-pyth
    """
    # check if destination directory exists
    if not os.path.exists(destination):
        # create destination directory
        os.makedirs(destination)
    for item in os.listdir(source):
        source_ = os.path.join(source, item)
        destination_ = os.path.join(destination, item)
        if os.path.isdir(source_):
            copytree(source_, destination_, symlinks, ignore)
        else:
            if not os.path.exists(destination_) or os.stat(source_).st_mtime - os.stat(destination_).st_mtime > 1:
                shutil.copy2(source_, destination_)
