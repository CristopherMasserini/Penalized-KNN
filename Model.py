import math


class KNNPenalized:
    def __init__(self, nclassifiers):
        self.nclassifiers = nclassifiers
        self.penalties = self.calculate_step_penalties()

    def calculate_step_penalties(self):
        penalties = []
        try:
            step = 1 / self.nclassifiers
            penalties = [round(1 - (i*step), 5) for i in range(0, self.nclassifiers)]
        except ZeroDivisionError:
            pass

        return penalties

    def euclidean_distance(self, a, b):
        try:
            return math.dist(a, b)
        except ValueError:
            return None

    def taxi_distance(self, a, b):
        if len(a) != len(b):
            return None
        else:
            distance = 0
            for i in range(0, len(a)):
                distance += abs(a[i] - b[i])
            return distance
