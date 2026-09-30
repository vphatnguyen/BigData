from time import perf_counter

from pyspark.sql import SparkSession

from amazon_reviews_experiment import DATASET_PATH, load_data, process_data


QUERIES = {
    "Query 1 - Total reviews": "SELECT COUNT(*) AS total_reviews FROM reviews",
    "Query 2 - Average rating": "SELECT AVG(rating) AS average_rating FROM reviews",
    "Query 3 - Rating distribution": "SELECT rating, COUNT(*) AS total_reviews FROM reviews GROUP BY rating ORDER BY rating",
    "Query 4 - Reviews by country": "SELECT country, COUNT(*) AS total_reviews FROM reviews GROUP BY country ORDER BY total_reviews DESC",
    "Query 5 - Reviews by time": "SELECT review_year AS year, review_month AS month, COUNT(*) AS total_reviews FROM reviews GROUP BY review_year, review_month ORDER BY year, month",
    "Query 6 - Top reviewers": "SELECT reviewer_name, review_count FROM reviews ORDER BY review_count DESC NULLS LAST LIMIT 10",
}


def main() -> None:
    spark = (
        SparkSession.builder
        .appName("AmazonReviewsStep6Analytics")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        raw_df = load_data(spark, DATASET_PATH)
        cleaned_df, _ = process_data(raw_df)
        cleaned_df.createOrReplaceTempView("reviews")

        for title, query in QUERIES.items():
            print(f"\n========== {title} ==========")
            start = perf_counter()
            result = spark.sql(query)
            result.show(20, truncate=False)
            result.collect()
            elapsed = perf_counter() - start
            print(f"{title} Time: {elapsed:.6f} seconds")

        print("\n========== Query 4 Execution Plan ==========")
        spark.sql(QUERIES["Query 4 - Reviews by country"]).explain()
        print("Step 6 Analytics completed successfully.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()