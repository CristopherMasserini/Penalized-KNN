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
    def __init__(self, points):
        self.points = points
        self.locations = []
        self.labels = []
        self.test_set = []
        self.train_set = []

    def location_label_split(self):
        for point in self.points:
            self.locations.append(point.location)
            self.labels.append(point.label)

    def dataframe_to_dataset(self):
        pass

    def split_dataset(self, split_size):
        for point in self.points:
            if random.random() <= split_size:
                point_test = TestTrainPoint(point.location, point.label)
                self.test_set.append(point_test)
            else:
                self.train_set.append(point)

