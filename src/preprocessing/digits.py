import pandas as pd

class Digits:

    def __init__(self, file_name):
        self.file_name = file_name


    def load_data(self):
        inputs = pd.read_csv(self.file_name, header=None, sep=" ")
        targets = inputs[0]

        # drop target column and last column
        inputs.drop(0, axis=1, inplace=True)
        inputs.dropna(axis=1, inplace=True)

        return inputs, targets
