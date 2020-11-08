from libraries.PyHTK.HTK import HTKFile as PyHTKFile

class File:

    def read(self):
        pass

    def write(self):
        pass

class TextFile(File):

    def __init__(self, filename, encoding="utf-8"):
        self.filename = filename
        self.encoding = encoding
        self.data = None

    def read(self, skip_n_rows=0):
        self.data = []

        try:
            with open(self.filename, "r", encoding=self.encoding) as file:
                for row_n, line in enumerate(file):
                    if row_n + 1 > skip_n_rows:
                        self.data.append(line.strip())

        except FileNotFoundError as error:
            print(error)

        return self.data


class HTKFile(File):

    def __init__(self, filename):
        self.filename = filename
        self.data = None

    def read(self):
        htk_file = PyHTKFile()
        htk_file.load(filename=self.filename)
        self.data = htk_file.data

        return self.data

