import random
import pandas as pd


class Point:
    def __init__(self, location, label=None):
        self.location = location
        self.label = label


class TestTrainPoint(Point):
    def __init__(self, location, label, test_label=None):
        super().__init__(location, label)
        self.test_label = test_label


class DataSet:
    def __init__(self, points=None):
        self.points = []
        if points:
            self.points = points

        self.locations = []
        self.labels = []

    def location_label_split(self):
        for point in self.points:
            self.locations.append(point.location)
            self.labels.append(point.label)

    def dataframe_to_dataset(self, file_location, labelColumn):
        df = pd.read_csv(file_location)
        labels = df.loc[:, labelColumn]
        cols = [col for col in df.columns if col != labelColumn]

        for i in range(0, len(labels)):
            entry_values = []
            for col in cols:
                location_i = df.loc[i, col]
                entry_values.append(location_i)

            point = Point(entry_values, labels[i])
            self.points.append(point)

