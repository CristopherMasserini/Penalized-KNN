import Files


def test_buildPointClass():
    point = Files.Point((5, 2), 'Class1')
    assert point.location == (5, 2)
    assert point.label == 'Class1'


def test_buildTestTrainPointClass():
    point = Files.TestTrainPoint((5, 2), 'Class1')
    assert point.test_label is None
    point.test_label = 'TESTCLASS'
    assert point.test_label == 'TESTCLASS'


def test_buildDataSetClass():
    point1 = Files.Point((5, 2), 'Class1')
    point2 = Files.Point((6, 9), 'Class2')
    point3 = Files.Point((-1, -3), 'Class3')
    dataset = Files.DataSet([point1, point2, point3])
    assert dataset.points[1] == point2


def test_locationLabelSplit():
    point1 = Files.Point((5, 2), 'Class1')
    point2 = Files.Point((6, 9), 'Class2')
    point3 = Files.Point((-1, -3), 'Class3')
    dataset = Files.DataSet([point1, point2, point3])
    dataset.location_label_split()
    assert dataset.locations == [(5, 2), (6, 9), (-1, -3)]
    assert dataset.labels == ['Class1', 'Class2', 'Class3']


def test_dataframeToDataset():
    file = 'Test_Datafile.csv'
    dataset = Files.DataSet()
    dataset.dataframe_to_dataset(file, 'Label')
    assert len(dataset.points) == 4
