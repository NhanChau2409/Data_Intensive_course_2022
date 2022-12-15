"""The assignment for Data-Intensive Programming 2022"""

from typing import List, Tuple

from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.storagelevel import StorageLevel
from math import isclose
import numpy as np
from matplotlib import pyplot as plt





class Assignment:
    spark: SparkSession = SparkSession.builder\
        .master("local") \
        .appName("ex3") \
        .config("spark.driver.host", "localhost") \
        .getOrCreate()
        
    # Set 5 partitions is enough for these dataset and one device
    spark.sparkContext.setLogLevel("WARN")
    spark.conf.set("spark.sql.shuffle.partitions", "5")


    # the data frame to be used in tasks 1 and 4
    dataD2: DataFrame = spark.read.csv("data/dataD2.csv", inferSchema=True, header=True)\
        .dropna(how="any")\
        .persist(StorageLevel.MEMORY_ONLY)  # store data in memory for more efficiency & clean data by drop null value


    # the data frame to be used in task 2
    dataD3: DataFrame = spark.read.csv("data/dataD3.csv", inferSchema=True, header=True)\
        .dropna(how="any")\
        .persist(StorageLevel.MEMORY_ONLY)  # store data in memory for more efficiency & clean data by drop null value


    # The data frame to be used in task 3 (based on dataD2 but containing numeric labels)
    # Prepare data for machine learning algorithm below
    label_indexer = StringIndexer(inputCol="LABEL", outputCol="LABEL_index")\
        .setStringOrderType("alphabetAsc")
        
    dataD2WithLabels: DataFrame = label_indexer.fit(dataD2).transform(dataD2)\
        .dropna(how="any")\
        .persist(StorageLevel.MEMORY_ONLY)  # store data in memory for more efficiency & clean data by drop null value

    @staticmethod
    def pipeline(df: DataFrame, input_cols: List) -> DataFrame:
        '''Make a pipeline for vector assembler then scale them'''
        vectorize_to_scale: VectorAssembler = VectorAssembler()\
            .setInputCols(input_cols)\
            .setOutputCol('features')
                
        scaler: MinMaxScaler = MinMaxScaler()\
            .setInputCol(vectorize_to_scale.getOutputCol())\
            .setOutputCol('scaled_features')\
            .setMax(1)\
            .setMin(0)

        pipeline: Pipeline = Pipeline(stages=[vectorize_to_scale, scaler])
            
        transformed_data = pipeline.fit(df).transform(df)
        
        return transformed_data

    @staticmethod
    def task1(df: DataFrame, k: int) -> List[Tuple[float, float]]:
        '''Find cluster of 2D data with KMeans from Mlib'''
        data = df.drop("LABEL")
        
        kmean: KMeans = KMeans()\
            .setK(k)\
            .setSeed(1)
        
        transformed_data = Assignment.pipeline(data, data.columns)
        
        kmodel = kmean.fit(transformed_data)
        
        return [tuple(i) for i in kmodel.clusterCenters()]

    @staticmethod
    def task2(df: DataFrame, k: int) -> List[Tuple[float, float, float]]:
        '''Find cluster for 3D data with KMeans from Mlib'''
        data = df.drop("LABEL")
        
        kmean: KMeans = KMeans()\
            .setK(k)\
            .setSeed(1)
        
        transformed_data = Assignment.pipeline(data, data.columns)
        
        kmodel = kmean.fit(transformed_data)
        
        return [tuple(i) for i in kmodel.clusterCenters()]

    @staticmethod
    def task3(df: DataFrame, k: int) -> List[Tuple[float, float]]:
        '''Find cluster of with label "fatal" of 2D data'''
        
        data = df.drop('LABEL')
        transformed_data = Assignment.pipeline(data, data.columns)
        
        kmean: KMeans = KMeans()\
            .setK(k)\
            .setSeed(1)
        
        
        kmodel = kmean.fit(transformed_data)
        
        return [tuple(i[:2]) for i in kmodel.clusterCenters() if round(i[2]) == 0]
        

    # Parameter low is the lowest k and high is the highest one.
    @staticmethod
    def task4(df: DataFrame, low: int, high: int) -> List[Tuple[int, float]]:
        '''Try many different numbers of cluster to find out which is optimal with silhouete method'''
        data = df.drop("LABEL")
        
        transformed_data = Assignment.pipeline(data, data.columns)
            
        silhouette_scores=[]
        
        evaluator = ClusteringEvaluator()

        for k in range(low, high+1):
            
            kmean: KMeans = KMeans()\
                .setK(k)\
                .setSeed(1)
                
            kmean_transformed = kmean.fit(transformed_data).transform(transformed_data)
            silhouette_scores.append((k, evaluator.evaluate(kmean_transformed)))
        
        # using matplotlib to plot silhouette_scores
        plot_data = np.array(silhouette_scores)
        x, y = plot_data.T
        plt.plot(x,y)
        plt.savefig('foo.pdf')
        # plt.show()
        return silhouette_scores