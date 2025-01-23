class Point:
    def __init__(self, location, label=None):
        self.location = location
        self.label = label


class DataSet:
    def __init__(self, points):
        self.points = points
