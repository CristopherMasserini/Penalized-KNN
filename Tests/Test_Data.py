import Data


def test_buildPointClass():
    point = Data.Point((5, 2), 'Class1')
    assert point.location == (5, 2)
    assert point.label == 'Class1'


def test_buildDataSetClass():
    point1 = Data.Point((5, 2), 'Class1')
    point2 = Data.Point((6, 9), 'Class2')
    point3 = Data.Point((-1, -3), 'Class3')
    dataset = Data.DataSet([point1, point2, point3])
    assert dataset.points[1] == point2


