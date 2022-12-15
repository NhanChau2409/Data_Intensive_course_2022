"""Assignment unit tests that compare the output values to reference values."""

from tests.DIPTestSuite import DIPTestSuite
from tests.TestUtils import DefaultTolerance
from tests.TestUtils import checkArray
from tests.TestUtils import checkArrays
from tests.TestUtils import getErrorMessage

__unittest = True


class ResultTestSuite(DIPTestSuite):
    """Tests the compare the output values to hardcoded reference values."""

    def test_result_task_1(self):
        df = self.assignment.dataD2
        self.dataFrameTest(df)

        K: int = 5
        # reference values when K-means seed is 1 and the centers are in the original scale
        ReferenceCenters = [
            (-0.312, 5.874),
            (0.916, 1.320),
            (-0.119, 1.601),
            (0.875, 4.599),
            (0.228, 7.030)
        ]
        # reference values when K-means seed is 1 and the coordinates are in [0, 1] scale
        ReferenceCentersUnscaled = [
            (0.328, 0.596),
            (0.925, 0.129),
            (0.421, 0.158),
            (0.906, 0.465),
            (0.590, 0.715)
        ]

        centers = self.assignment.task1(df, K)
        self.assertTrue(
            checkArrays(
                inputArray=centers,
                referenceArray1=ReferenceCenters,
                referenceArray2=ReferenceCentersUnscaled,
                tolerance=DefaultTolerance
            ),
            getErrorMessage(centers)
        )

    def test_result_task_2(self):
        df = self.assignment.dataD3
        self.dataFrameTest(df)

        K: int = 5
        # reference values when K-means seed is 1 and the centers are in the original scale
        ReferenceCenters = [
            (0.105, 2.126, 1921.846),
            (0.146, 8.993, 3600.213),
            (-0.983, 1.996, 835.407),
            (0.740, 3.523, 3327.593),
            (0.771, 7.916, -775.895)
        ]
        # reference values when K-means seed is 1 and the coordinates are in [0, 1] scale
        ReferenceCentersUnscaled = [
            (0.584, 0.210, 0.518),
            (0.603, 0.902, 0.791),
            (0.066, 0.197, 0.342),
            (0.885, 0.350, 0.746),
            (0.900, 0.793, 0.081)
        ]

        centers = self.assignment.task2(df, K)
        self.assertTrue(
            checkArrays(
                inputArray=centers,
                referenceArray1=ReferenceCenters,
                referenceArray2=ReferenceCentersUnscaled,
                tolerance=DefaultTolerance
            ),
            getErrorMessage(centers)
        )

    def test_result_task_3(self):
        df = self.assignment.dataD2WithLabels
        self.dataFrameTest(df)

        K: int = 5
        # reference values when K-means seed is 1 and the centers are in the original scale
        ReferenceCenters = [
            (-0.292, 5.890),
            (0.872, 4.598)
        ]
        # reference values when K-means seed is 1 and the coordinates are in [0, 1] scale
        ReferenceCentersUnscaled = [
            (0.337, 0.595),
            (0.904, 0.465)
        ]

        centers = self.assignment.task3(df, K)
        self.assertTrue(
            checkArrays(
                inputArray=centers,
                referenceArray1=ReferenceCenters,
                referenceArray2=ReferenceCentersUnscaled,
                tolerance=DefaultTolerance
            ),
            getErrorMessage(centers)
        )

    def test_result_task_4(self):
        df = self.assignment.dataD2
        self.dataFrameTest(df)

        lowK: int = 2
        highK: int = 13
        # reference values when K-means seed is 1
        ReferenceMeasures = [
            (2, 0.637),
            (3, 0.827),
            (4, 0.882),
            (5, 0.961),
            (6, 0.835),
            (7, 0.728),
            (8, 0.739),
            (9, 0.623),
            (10, 0.736),
            (11, 0.505),
            (12, 0.641),
            (13, 0.524)
        ]

        centers = self.assignment.task4(df, lowK, highK)
        self.assertTrue(
            checkArray(
                array1=centers,
                array2=ReferenceMeasures,
                tolerance=2.5 * DefaultTolerance
            ),
            getErrorMessage(centers)
        )
