import unittest
import warnings

from pyspark.sql import DataFrame
from pyspark.sql import SparkSession

from assignment22.assignment import Assignment

__unittest = True


class DIPTestSuite(unittest.TestCase):
    SparkSessionNotInitialized: str = "Spark session was not initialized"
    DataFrameNotInitialized: str = "Data frame was not initialized"

    def setUp(self) -> None:
        super().setUp()
        warnings.simplefilter("ignore", ResourceWarning)
        self.assignment = Assignment

    def dataFrameTest(self, df: DataFrame) -> None:
        # check first that the Spark session is initialized
        self.assertIsInstance(self.assignment.spark, SparkSession, self.SparkSessionNotInitialized)
        self.assertIsInstance(df, DataFrame, self.DataFrameNotInitialized)
