import os

FILE_EXTENSION_VAW = ".wav"
def get_file_paths(directory, file_extension):
    """
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

"""
Clone directory
https://stackoverflow.com/questions/40828450/how-to-copy-folder-structure-under-another-directory
https://stackoverflow.com/questions/1868714/how-do-i-copy-an-entire-directory-of-files-into-an-existing-directory-using-pyth
"""