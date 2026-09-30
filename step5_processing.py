from pyspark.sql import SparkSession

from amazon_reviews_experiment import DATASET_PATH, load_data, process_data


def main() -> None:
    spark = (
        SparkSession.builder
        .appName("AmazonReviewsStep5Processing")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        reviews_df = load_data(spark, DATASET_PATH)
        _, processing_results = process_data(reviews_df)
        print("Step 5 Processing completed successfully.")
        print("Processing results:", processing_results)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()