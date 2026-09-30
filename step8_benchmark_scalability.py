from pyspark import StorageLevel
from pyspark.sql import SparkSession

from amazon_reviews_experiment import (
    DATASET_PATH,
    benchmark,
    load_data,
    process_data,
    run_scalability_test,
)


def main() -> None:
    spark = (
        SparkSession.builder
        .appName("AmazonReviewsStep8BenchmarkScalability")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.default.parallelism", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        print("========== BASE BENCHMARK ==========")
        performance = benchmark(spark, DATASET_PATH)
        print("Read Time: {:.6f} seconds".format(performance["read_time"]))
        print("Processing Time: {:.6f} seconds".format(performance["processing_time"]))
        print("Analytics Time: {:.6f} seconds".format(performance["analytics_time"]))
        print("Total Time: {:.6f} seconds".format(performance["total_time"]))

        raw_df = load_data(spark, DATASET_PATH)
        cleaned_df, _ = process_data(raw_df)
        cleaned_df.persist(StorageLevel.MEMORY_AND_DISK).count()

        print("\n========== SCALABILITY: SYNTHETIC SCALED DATASET ==========")
        results = run_scalability_test(spark, cleaned_df)
        print("\nScale | Records | Read Time | Processing Time | Analytics Time | Total Time")
        for row in results:
            print(
                "{scale} | {records} | {read_time:.6f} | {processing_time:.6f} | "
                "{analytics_time:.6f} | {total_time:.6f}".format(**row)
            )
        print("\nStep 8 Benchmark and Scalability completed successfully.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()