import math
from sklearn import metrics
import random
from .Data import Point, TestTrainPoint, DataSet


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

    def closest_point_check(self, point: Point, all_points, dist_type='euclidean'):
        # Assumption that point is not in all_points
        # Other distance type is 'taxi'
        closest_point = None
        closest_point_distance = None
        for i, check_point in enumerate(all_points):
            point_loc = point.location
            if dist_type == 'euclidean':
                point_dist = self.euclidean_distance(point_loc, check_point.location)
            else:
                point_dist = self.taxi_distance(point_loc, check_point.location)

            if i == 0:
                closest_point = check_point
                closest_point_distance = point_dist
            else:
                if point_dist < closest_point_distance:
                    closest_point = check_point
                    closest_point_distance = point_dist

        return closest_point, closest_point_distance

    def closest_n_points(self, n, point: Point, all_points, dist_type='euclidean'):
        search_points = all_points.copy()
        closest_neighbors = {}
        for i in range(0, n):
            closest_point_info = self.closest_point_check(point, search_points, dist_type)
            closest_neighbors[i] = {'coord': closest_point_info[0].location,
                                    'dist': closest_point_info[1],
                                    'label': closest_point_info[0].label}
            search_points.remove(closest_point_info[0])

        return closest_neighbors

    def classify_point(self, point: Point, all_points, dist_type='euclidean'):
        n_closest_neighbors = self.closest_n_points(self.nclassifiers,
                                                    point,
                                                    all_points,
                                                    dist_type)

        label_scores = {}
        for i, penalty in enumerate(self.penalties):
            neighbor_label = n_closest_neighbors[i]['label']
            if neighbor_label in label_scores:
                label_scores[neighbor_label] += penalty
            else:
                label_scores[neighbor_label] = penalty

        return max(label_scores, key=label_scores.get)

    def classify_points(self, input_points: DataSet, dataset: DataSet, dist_type='euclidean', testing=False):
        for point in input_points.points:
            all_points = dataset.points
            label_classified = self.classify_point(point, all_points, dist_type)
            if not testing:
                point.label = label_classified
            else:
                point.test_label = label_classified

        return input_points

    def split_dataset(self, dataset, split_size):
        test_set = DataSet()
        train_set = DataSet()
        for point in dataset.points:
            if random.random() <= split_size:
                point_test = TestTrainPoint(point.location, point.label)
                test_set.points.append(point_test)
            else:
                train_set.points.append(point)

        return test_set, train_set

    def test_model(self, all_data: DataSet, split_size, label_names, dist_type='euclidean'):
        test_set, train_set = self.split_dataset(all_data, split_size)
        test_points_labeled = self.classify_points(test_set, train_set, dist_type=dist_type, testing=True)

        label_given = []
        label_pred = []

        for point in test_points_labeled.points:
            label_pred.append(point.test_label)
            label_given.append(point.label)

        print(metrics.classification_report(label_given, label_pred, target_names=label_names))
        return test_points_labeled
