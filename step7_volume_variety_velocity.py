from pyspark.sql import SparkSession

from amazon_reviews_experiment import (
    DATASET_PATH,
    analyze_variety,
    analyze_velocity,
    analyze_volume,
    load_data,
    process_data,
)


def main() -> None:
    spark = (
        SparkSession.builder
        .appName("AmazonReviewsStep7VolumeVarietyVelocity")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        raw_df = load_data(spark, DATASET_PATH)
        cleaned_df, _ = process_data(raw_df)

        print("\n========== VOLUME ==========")
        volume = analyze_volume(cleaned_df)
        print("Volume results:", volume)

        print("\n========== VARIETY ==========")
        variety = analyze_variety(cleaned_df)
        print("Variety results:", {
            "unique_countries": variety["unique_countries"],
            "text_stats": variety["text_stats"],
        })

        print("\n========== VELOCITY ==========")
        velocity = analyze_velocity(cleaned_df)
        print("Velocity statistics (seconds):", velocity["stats"])
        print("Velocity execution plan:")
        velocity["velocity_df"].explain()

        print("Step 7 Volume, Variety and Velocity completed successfully.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()