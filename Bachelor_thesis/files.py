from libraries.PyHTK.HTK import HTKFile as PyHTKFile

class File:

    def read(self, file_path):
        pass

    def write(self, file_path):
        pass

class TextFile(File):

    def __init__(self, encoding="utf-8"):
        self.encoding = encoding
        self.data = None

    def read(self, filename, skip_n_rows=0):
        self.data = []

        try:
            with open(filename, "r", encoding=self.encoding) as file:
                for row_n, line in enumerate(file):
                    if row_n + 1 > skip_n_rows:
                        self.data.append(line.strip())

        except FileNotFoundError as error:
            print(error)

        return self.data


class HTKFile(File):

    def __init__(self):
        self.data = None

    def read(self, file_path):
        htk_file = PyHTKFile()
        htk_file.load(filename=file_path)
        self.data = htk_file.data
        return self.data

