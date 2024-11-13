import math


class KNNPenalized:
    def __init__(self, nclassifiers):
        self.nclassifiers = nclassifiers
        self.penalties = [1-(i/self.nclassifiers) for i in range(0, self.nclassifiers)]

    def euclidean_distance(self, a, b):
        return math.dist(a, b)

    def taxi_distance(self, a, b):
        distance = 0
        for i in range(0, len(a)):
            distance += abs(a[i] - b[i])
        return distance

