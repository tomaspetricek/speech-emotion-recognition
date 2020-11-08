class File:

    def read(self):
        pass

    def write(self):
        pass

class TextFile(File):

    def __init__(self, filename, encoding="utf-8"):
        self.filename = filename
        self.encoding = encoding

    def read(self, skip_n_rows=0):
        data = []

        try:
            with open(self.filename, "r", encoding=self.encoding) as file:
                for row_n, line in enumerate(file):
                    if row_n + 1 > skip_n_rows:
                        data.append(line.strip())

        except FileNotFoundError as error:
            print(error)

        return data
