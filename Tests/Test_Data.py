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


def test_locationLabelSplit():
    point1 = Data.Point((5, 2), 'Class1')
    point2 = Data.Point((6, 9), 'Class2')
    point3 = Data.Point((-1, -3), 'Class3')
    dataset = Data.DataSet([point1, point2, point3])
    dataset.location_label_split()
    assert dataset.locations == [(5, 2), (6, 9), (-1, -3)]
    assert dataset.labels == ['Class1', 'Class2', 'Class3']


def test_splitDataset():
    point1 = Data.Point((5, 2), 'Class1')
    point2 = Data.Point((6, 9), 'Class2')
    point3 = Data.Point((-1, -3), 'Class3')
    point4 = Data.Point((-9, 10), 'Class4')
    dataset = Data.DataSet([point1, point2, point3, point4])
    dataset.split_dataset(0.5)
    assert len(dataset.test_set) > 0
    assert len(dataset.train_set) > 0



