import Data


def test_buildPointClass():
    point = Data.Point((5, 2), 'Class1')
    assert point.location == (5, 2)
    assert point.label == 'Class1'
