"""Basic unit tests for the assignment"""

from DIPTestSuite import DIPTestSuite

__unittest = True


class BaseTestSuite(DIPTestSuite):
    """Simple test for the output format for the tasks."""
    OutputTypeFailure: str = "Did not receive a list"
    LengthFailure: str = "Wrong number of elements"
    ElementTypeFailure: str = "Element was not a tuple"
    ElementLengthFailure: str = "Wrong number of values in a list element"
    CoordinateTypeFailure: str = "Coordinate was not a float"

    def test_simple_task_1(self):
        k: int = 5
        df = self.assignment.dataD2
        self.dataFrameTest(df)

        centers = self.assignment.task1(df, k)
        self.assertIsInstance(centers, list, self.OutputTypeFailure)

        self.assertEqual(len(centers), k, self.LengthFailure)
        for center in centers:
            self.assertIsInstance(center, tuple, self.ElementTypeFailure)
            self.assertEqual(len(center), 2, self.ElementLengthFailure)
            for coordinate in center:
                self.assertIsInstance(coordinate, float, self.CoordinateTypeFailure)

    def test_simple_task_2(self):
        k: int = 5
        df = self.assignment.dataD3
        self.dataFrameTest(df)

        centers = self.assignment.task2(df, k)
        self.assertIsInstance(centers, list, self.OutputTypeFailure)

        self.assertEqual(len(centers), k, self.LengthFailure)
        for center in centers:
            self.assertIsInstance(center, tuple, self.ElementTypeFailure)
            self.assertEqual(len(center), 3, self.ElementLengthFailure)
            for coordinate in center:
                self.assertIsInstance(coordinate, float, self.CoordinateTypeFailure)

    def test_simple_task_3(self):
        k: int = 5
        df = self.assignment.dataD2WithLabels
        self.dataFrameTest(df)

        centers = self.assignment.task3(df, k)
        self.assertIsInstance(centers, list, self.OutputTypeFailure)

        self.assertEqual(len(centers), 2, self.LengthFailure)
        for center in centers:
            self.assertIsInstance(center, tuple, self.ElementTypeFailure)
            self.assertEqual(len(center), 2, self.ElementLengthFailure)
            for coordinate in center:
                self.assertIsInstance(coordinate, float, self.CoordinateTypeFailure)

    def test_simple_task_4(self):
        lowK: int = 2
        highK: int = 3
        df = self.assignment.dataD2
        self.dataFrameTest(df)

        measures = self.assignment.task4(df, lowK, highK)
        self.assertIsInstance(measures, list, self.OutputTypeFailure)

        self.assertEqual(len(measures), highK - lowK + 1, self.LengthFailure)
        for measureTuple in measures:
            self.assertIsInstance(measureTuple, tuple, self.ElementTypeFailure)
            self.assertEqual(len(measureTuple), 2, self.ElementLengthFailure)

            k, measure = measureTuple
            self.assertIsInstance(k, int, "value for k was not given as int")
            self.assertIsInstance(measure, float, "measure was not given as float")

