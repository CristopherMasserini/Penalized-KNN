import Files
import Model
import pytest
import math
from Files import Point


def test_buildModelClass():
    model = Model.KNNPenalized(5)
    assert model.nclassifiers == 5


def test_modelPenalties():
    model1 = Model.KNNPenalized(0)
    model2 = Model.KNNPenalized(5)
    model3 = Model.KNNPenalized(10)
    assert model1.penalties == []
    assert model2.penalties == [1, 0.8, 0.6, 0.4, 0.2]
    assert model3.penalties == [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]


@pytest.mark.parametrize("a,b,expected",
                         [([1, 2], [1, 2], 0),
                          ([1, 2], [1, 2, 3], None),
                          ([0, 0], [6, 6], 6 * math.sqrt(2)),
                          ([1, 2], [4, 6], 5),
                          ([1, 2, 3], [4, 6, 8], math.sqrt(50)),
                          ])
def test_modelEuclideanDistance(a, b, expected):
    model = Model.KNNPenalized(0)
    result = model.euclidean_distance(a, b)
    assert result == expected


@pytest.mark.parametrize("a,b,expected",
                         [([1, 2], [1, 2], 0),
                          ([1, 2], [1, 2, 3], None),
                          ([0, 0], [6, 6], 12),
                          ([1, 2], [4, 6], 7),
                          ([1, 2, 3], [4, 6, 8], 12),
                          ])
def test_modelTaxiDistance(a, b, expected):
    model = Model.KNNPenalized(0)
    result = model.taxi_distance(a, b)
    assert result == expected


@pytest.mark.parametrize("dist_type",
                         ['euclidean', 'taxi'])
def test_modelClosestPointCheck(dist_type):
    model = Model.KNNPenalized(3)
    point = Point([0, 0])
    all_points = [Point([1, 1], 'A'),
                  Point([1, 2], 'B'),
                  Point([3, 5], 'C'),
                  Point([0, 1], 'D')]
    closest_point = model.closest_point_check(point, all_points, dist_type)
    assert closest_point[0].location == [0, 1]
    assert closest_point[1] == 1.0
    assert closest_point[0].label == 'D'


def test_closestNPoints():
    model = Model.KNNPenalized(3)
    point = Point([0, 0], 'A')
    all_points = [Point([1, 1], 'A'),
                  Point([1, 2], 'B'),
                  Point([3, 5], 'C'),
                  Point([0, 1], 'D')]
    nearest = model.closest_n_points(3, point, all_points, 'taxi')
    assert nearest == {0: {'coord': [0, 1], 'dist': 1, 'label': 'D'},
                       1: {'coord': [1, 1], 'dist': 2, 'label': 'A'},
                       2: {'coord': [1, 2], 'dist': 3, 'label': 'B'}}


def test_classifyPoint():
    model = Model.KNNPenalized(4)
    point = Point([0, 0], None)
    all_points = [Point([1, 1], 'A'),
                  Point([1, 4], 'B'),
                  Point([3, 5], 'C'),
                  Point([0, 1], 'D'),
                  Point([-1, 2], 'A'),
                  Point([-1, -1], 'A'),
                  Point([1, -3], 'D')
                  ]
    classification = model.classify_point(point, all_points, 'taxi')
    assert classification == 'A'


def test_classifyPoints():
    model = Model.KNNPenalized(4)

    point1 = Point([0, 0], None)
    point2 = Point([5, 4], None)
    input_points = Files.DataSet([point1, point2])

    dataset = Files.DataSet([Point([1, 1], 'A'),
                             Point([1, 4], 'B'),
                             Point([3, 5], 'C'),
                             Point([0, 1], 'D'),
                             Point([-1, 2], 'A'),
                             Point([-1, -1], 'A'),
                             Point([1, -3], 'D'),
                             ])

    classified_points = model.classify_points(input_points, dataset, 'taxi')
    assert classified_points.points[0].label == 'A'
    assert classified_points.points[1].label == 'C'


def test_splitDataset():
    model = Model.KNNPenalized(4)

    point1 = Files.Point((5, 2), 'Class1')
    point2 = Files.Point((6, 9), 'Class2')
    point3 = Files.Point((-1, -3), 'Class3')
    point4 = Files.Point((-9, 10), 'Class4')
    dataset = Files.DataSet([point1, point2, point3, point4])
    train_set, test_set = model.split_dataset(dataset, 0.5)
    assert len(test_set.points) > 0
    assert len(train_set.points) > 0


def test_testModel():
    model = Model.KNNPenalized(4)

    dataset = Files.DataSet([Point([1, 1], 'A'),
                             Point([1, 4], 'B'),
                             Point([3, 5], 'C'),
                             Point([0, 1], 'D'),
                             Point([-1, 2], 'A'),
                             Point([-1, -1], 'A'),
                             Point([1, -3], 'D'),
                             Point([0, 0], 'B'),
                             Point([5, 4], 'C')
                             ])

    classified_points = model.test_model(dataset, 0.2)
    assert classified_points.points[0].test_label is not None
