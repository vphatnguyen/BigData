from pathlib import Path

from pyspark.sql import SparkSession


DATASET_PATH = Path(__file__).with_name("Amazon_Reviews.csv")


def load_data(spark: SparkSession):
    """Read the source CSV without changing its values."""
    return (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .option("quote", '"')
        .option("escape", '"')
        .csv(str(DATASET_PATH))
    )


def inspect_data(reviews_df) -> None:
    print("Spark Version:")
    print(reviews_df.sparkSession.version)
    print("\nTotal Records:")
    print(reviews_df.count())
    print("\nTotal Columns:")
    print(len(reviews_df.columns))
    print("\nSchema:")
    reviews_df.printSchema()
    print("Sample Data:")
    reviews_df.show(5, truncate=False)
    print("Number of Partitions:")
    print(reviews_df.rdd.getNumPartitions())


def main() -> None:
    spark = (
        SparkSession.builder
        .appName("AmazonReviewsStep3Read")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        reviews_df = load_data(spark)
        inspect_data(reviews_df)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()