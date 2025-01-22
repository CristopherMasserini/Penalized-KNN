import Model
import pytest
import math


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
                          ([0, 0], [6, 6], 6*math.sqrt(2)),
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
    model = Model.KNNPenalized(5)
    point = [0, 0]
    all_points = [[1, 1], [1, 2], [3, 5], [6, 7], [9, 10], [0, 1]]
    closest_point = model.closest_point_check(point, all_points, dist_type)
    assert closest_point == ([0, 1], 1.0)


def test_closestNPoints():
    model = Model.KNNPenalized(5)
    point = [0, 0]
    all_points = [[1, 1], [1, 2], [3, 5], [6, 7], [9, 10], [0, 1]]
    nearest = model.closest_n_points(3, point, all_points, 'taxi')
    assert nearest == {0: {'coord': [0, 1], 'dist': 1},
                       1: {'coord': [1, 1], 'dist': 2},
                       2: {'coord': [1, 2], 'dist': 3}}
