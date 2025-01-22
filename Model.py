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

    def closest_point_check(self, point, all_points, dist_type='euclidean'):
        # Assumption that point is not in all_points
        # Other distance type is 'taxi'
        closest_point = None
        closest_point_distance = None
        for i, check_point in enumerate(all_points):
            if dist_type == 'euclidean':
                point_dist = self.euclidean_distance(point, check_point)
            else:
                point_dist = self.taxi_distance(point, check_point)

            if i == 0:
                closest_point = check_point
                closest_point_distance = point_dist
            else:
                if point_dist < closest_point_distance:
                    closest_point = check_point
                    closest_point_distance = point_dist

        return closest_point, closest_point_distance

    def closest_n_points(self, n, point, all_points, dist_type='euclidean'):
        closest_neighbors = {}
        for i in range(0, n):
            closest_point, closest_point_distance = self.closest_point_check(point, all_points, dist_type)
            closest_neighbors[i] = {'coord': closest_point, 'dist': closest_point_distance}
            all_points.remove(closest_point)

        return closest_neighbors
