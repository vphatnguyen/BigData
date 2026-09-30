from pathlib import Path
import shutil
import tempfile
from time import perf_counter

from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql.types import BooleanType, DoubleType, LongType, StringType, StructField, StructType
from pyspark.sql import functions as F


BASE_DIR = Path(__file__).parent
DATASET_PATH = BASE_DIR / "Amazon_Reviews.csv"
OUTPUT_DIR = BASE_DIR / "output"


def load_data(spark: SparkSession, path: Path = DATASET_PATH) -> DataFrame:
    return (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .option("quote", '"')
        .option("escape", '"')
        .csv(str(path))
    )


def inspect_data(reviews_df: DataFrame) -> dict:
    record_count = reviews_df.count()
    print("Spark Version:", reviews_df.sparkSession.version)
    print("Total Records:", record_count)
    print("Total Columns:", len(reviews_df.columns))
    print("Schema:")
    reviews_df.printSchema()
    print("Sample Data:")
    reviews_df.show(5, truncate=False)
    partitions = reviews_df.rdd.getNumPartitions()
    print("Number of Partitions:", partitions)
    return {"records": record_count, "columns": len(reviews_df.columns), "partitions": partitions}


def process_data(reviews_df: DataFrame) -> tuple[DataFrame, dict]:
    records_before = reviews_df.count()
    null_counts = reviews_df.select(
        [F.sum(F.when(F.col(column).isNull(), 1).otherwise(0)).alias(column)
         for column in reviews_df.columns]
    ).first().asDict()

    deduplicated_df = reviews_df.dropDuplicates()   
    records_after = deduplicated_df.count()

    cleaned_df = (
        deduplicated_df
        .withColumn("reviewer_name", F.col("Reviewer Name"))
        .withColumn("profile_link", F.col("Profile Link"))
        .withColumn("country", F.col("Country"))
        .withColumn(
            "review_count",
            F.expr(
                "try_cast(regexp_replace(regexp_extract(`Review Count`, '(\\\\d[\\\\d,]*)', 1), ',', '') AS INT)"
            ),
        )
        .withColumn("review_date", F.expr("try_to_timestamp(`Review Date`)"))
        .withColumn("rating", F.expr("try_cast(regexp_extract(`Rating`, '([1-5])', 1) AS INT)"))
        .withColumn("review_title", F.col("Review Title"))
        .withColumn("review_text", F.col("Review Text"))
        .withColumn("date_of_experience", F.col("Date of Experience"))
        .withColumn("review_year", F.year("review_date"))
        .withColumn("review_month", F.month("review_date"))
        .withColumn("review_text_length", F.length(F.coalesce(F.col("review_text"), F.lit(""))))
        .select(
            "reviewer_name", "profile_link", "country", "review_count", "review_date",
            "rating", "review_title", "review_text", "date_of_experience", "review_year",
            "review_month", "review_text_length"
        )
    )
    cleaned_count = cleaned_df.count()
    print("NULL COUNTS:")
    for column, count in null_counts.items():
        print(f"{column} | {count}")
    print("Records Before:", records_before)
    print("Records After:", records_after)
    print("Duplicates Removed:", records_before - records_after)
    print("Schema after Processing:")
    cleaned_df.printSchema()
    print("5 records after Processing:")
    cleaned_df.show(5, truncate=False)
    return cleaned_df, {
        "null_counts": null_counts,
        "records_before": records_before,
        "records_after": records_after,
        "duplicates_removed": records_before - records_after,
        "cleaned_count": cleaned_count,
    }


def analyze_volume(cleaned_df: DataFrame) -> dict:
    return {"records": cleaned_df.count(), "partitions": cleaned_df.rdd.getNumPartitions()}


def analyze_variety(cleaned_df: DataFrame) -> dict:
    unique_countries = cleaned_df.select("country").where(F.col("country").isNotNull()).distinct().count()
    country_counts = cleaned_df.groupBy("country").count().orderBy(F.desc("count"))
    text_stats = cleaned_df.select(
        F.min("review_text_length").alias("minimum"),
        F.max("review_text_length").alias("maximum"),
        F.avg("review_text_length").alias("average"),
    ).first().asDict()
    print("Total Unique Countries:", unique_countries)
    print("Top 10 Country:")
    country_counts.show(10, truncate=False)
    print("Review Text Length Statistics:", text_stats)
    return {
        "unique_countries": unique_countries,
        "country_counts": country_counts,
        "text_stats": text_stats,
    }


def analyze_velocity(cleaned_df: DataFrame) -> dict:
    dated_df = cleaned_df.where(F.col("review_date").isNotNull())
    window = Window.orderBy("review_date")
    velocity_df = (
        dated_df
        .withColumn("previous_review_date", F.lag("review_date").over(window))
        .withColumn(
            "time_diff_seconds",
            F.col("review_date").cast("long") - F.col("previous_review_date").cast("long"),
        )
        .orderBy("review_date")
    )
    velocity_stats = velocity_df.where(F.col("time_diff_seconds").isNotNull()).select(
        F.min("time_diff_seconds").alias("minimum_seconds"),
        F.max("time_diff_seconds").alias("maximum_seconds"),
        F.avg("time_diff_seconds").alias("average_seconds"),
        F.expr("percentile_approx(time_diff_seconds, 0.5)").alias("median_seconds"),
    ).first().asDict()
    by_day = dated_df.groupBy(F.to_date("review_date").alias("date")).count().orderBy("date")
    by_month = dated_df.groupBy(
        F.date_format("review_date", "yyyy-MM").alias("year_month")
    ).count().orderBy("year_month")
    print("Velocity Statistics (seconds):", velocity_stats)
    print("Reviews by Day:")
    by_day.show(20, truncate=False)
    print("Reviews by Month:")
    by_month.show(20, truncate=False)
    return {"velocity_df": velocity_df, "stats": velocity_stats, "by_day": by_day, "by_month": by_month}


def run_sql_analytics(cleaned_df: DataFrame) -> tuple[dict, dict]:
    cleaned_df.createOrReplaceTempView("reviews")
    queries = {
        "query_1_total_reviews": "SELECT COUNT(*) AS total_reviews FROM reviews",
        "query_2_average_rating": "SELECT AVG(rating) AS average_rating FROM reviews",
        "query_3_rating_distribution": "SELECT rating, COUNT(*) AS total_reviews FROM reviews GROUP BY rating ORDER BY rating",
        "query_4_reviews_by_country": "SELECT country, COUNT(*) AS total_reviews FROM reviews GROUP BY country ORDER BY total_reviews DESC",
        "query_5_reviews_by_time": "SELECT review_year AS year, review_month AS month, COUNT(*) AS total_reviews FROM reviews GROUP BY review_year, review_month ORDER BY year, month",
        "query_6_top_reviewers": "SELECT reviewer_name, review_count FROM reviews ORDER BY review_count DESC NULLS LAST LIMIT 10",
    }
    results = {}
    timings = {}
    for name, query in queries.items():
        start = perf_counter()
        result = cleaned_df.sparkSession.sql(query)
        result.show(20, truncate=False)
        result_rows = [row.asDict() for row in result.collect()]
        timings[name] = perf_counter() - start
        results[name] = result_rows
        print(f"{name} Time: {timings[name]:.6f} seconds")
    return results, timings


def benchmark(spark: SparkSession, path: Path = DATASET_PATH) -> dict:
    start_total = perf_counter()
    start_read = perf_counter()
    raw_df = load_data(spark, path)
    raw_count = raw_df.count()
    read_time = perf_counter() - start_read
    start_processing = perf_counter()
    cleaned_df, _ = process_data(raw_df)
    cleaned_df.cache().count()
    processing_time = perf_counter() - start_processing
    start_analytics = perf_counter()
    cleaned_df.groupBy("rating").count().collect()
    analytics_time = perf_counter() - start_analytics
    return {
        "records": raw_count,
        "columns": len(raw_df.columns),
        "read_time": read_time,
        "processing_time": processing_time,
        "analytics_time": analytics_time,
        "total_time": perf_counter() - start_total,
    }


def run_scalability_test(spark: SparkSession, cleaned_df: DataFrame) -> list[dict]:
    base_count = cleaned_df.count()
    results = []
    for scale in (1, 2, 5, 10):
        start_total = perf_counter()
        scaled_df = cleaned_df
        for _ in range(scale - 1):
            scaled_df = scaled_df.union(cleaned_df)
        start_processing = perf_counter()
        scaled_df = scaled_df.repartition(4).persist(StorageLevel.MEMORY_AND_DISK)
        processed_count = scaled_df.count()
        processing_time = perf_counter() - start_processing
        temporary_path = Path(tempfile.mkdtemp(prefix=f"amazon_reviews_{scale}x_", dir=BASE_DIR))
        try:
            scaled_df.write.mode("overwrite").parquet(str(temporary_path))
            start_read = perf_counter()
            read_df = spark.read.parquet(str(temporary_path))
            read_df.cache().count()
            read_time = perf_counter() - start_read
            start_analytics = perf_counter()
            read_df.groupBy("rating").count().collect()
            analytics_time = perf_counter() - start_analytics
        finally:
            shutil.rmtree(temporary_path, ignore_errors=True)
        row = {
            "scale": f"{scale}x",
            "records": processed_count,
            "read_time": read_time,
            "processing_time": processing_time,
            "analytics_time": analytics_time,
            "total_time": perf_counter() - start_total,
            "synthetic": True,
        }
        results.append(row)
        print(row)
        if scale != 1:
            scaled_df.unpersist()
    print(f"Synthetic scaling base records: {base_count}")
    return results


def save_results(
    cleaned_df: DataFrame,
    variety: dict,
    velocity: dict,
    sql_results: dict,
    performance: dict,
    scalability: list[dict],
) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    cleaned_df.write.mode("overwrite").option("header", True).csv(str(OUTPUT_DIR / "processed_reviews"))
    variety["country_counts"].write.mode("overwrite").option("header", True).csv(str(OUTPUT_DIR / "reviews_by_country"))
    velocity["by_day"].write.mode("overwrite").option("header", True).csv(str(OUTPUT_DIR / "reviews_by_time"))
    velocity["velocity_df"].write.mode("overwrite").option("header", True).csv(str(OUTPUT_DIR / "velocity_analysis"))

    spark = cleaned_df.sparkSession
    spark.createDataFrame(sql_results["query_3_rating_distribution"]).write.mode("overwrite").option("header", True).csv(str(OUTPUT_DIR / "rating_distribution"))
    spark.createDataFrame(sql_results["query_4_reviews_by_country"]).write.mode("overwrite").option("header", True).csv(str(OUTPUT_DIR / "reviews_by_country_sql"))
    spark.createDataFrame(sql_results["query_5_reviews_by_time"]).write.mode("overwrite").option("header", True).csv(str(OUTPUT_DIR / "reviews_by_time_sql"))
    spark.createDataFrame(sql_results["query_6_top_reviewers"]).write.mode("overwrite").option("header", True).csv(str(OUTPUT_DIR / "top_reviewers"))
    spark.createDataFrame([sql_results["query_2_average_rating"][0]]).write.mode("overwrite").option("header", True).csv(str(OUTPUT_DIR / "rating_statistics"))
    scalability_schema = StructType([
        StructField("scale", StringType(), False),
        StructField("records", LongType(), False),
        StructField("read_time", DoubleType(), True),
        StructField("processing_time", DoubleType(), False),
        StructField("analytics_time", DoubleType(), False),
        StructField("total_time", DoubleType(), False),
        StructField("synthetic", BooleanType(), False),
    ])
    spark.createDataFrame(scalability, schema=scalability_schema).write.mode("overwrite").option("header", True).csv(str(OUTPUT_DIR / "performance_results"))

    summary_lines = [
        "========== DATASET ==========",
        f"Records: {performance['records']}",
        f"Columns: {performance['columns']}",
        f"Partitions: {cleaned_df.rdd.getNumPartitions()}",
        "\n========== VOLUME ==========",
        f"Read Time: {performance['read_time']:.6f} seconds",
        f"Processing Time: {performance['processing_time']:.6f} seconds",
        f"Analytics Time: {performance['analytics_time']:.6f} seconds",
        f"Total Time: {performance['total_time']:.6f} seconds",
        "\n========== VARIETY ==========",
        f"Unique Countries: {variety['unique_countries']}",
        f"Review Text Statistics: {variety['text_stats']}",
        "\n========== VELOCITY (seconds) ==========",
        f"{velocity['stats']}",
        "\n========== ANALYTICS ==========",
        f"Average Rating: {sql_results['query_2_average_rating']}",
        f"Top Country: {sql_results['query_4_reviews_by_country'][:1]}",
        f"Top Reviewer: {sql_results['query_6_top_reviewers'][:1]}",
        "\n========== SCALABILITY (Synthetic Scaled Dataset) ==========",
        "Scale | Records | Read Time | Processing Time | Analytics Time | Total Time",
    ]
    for row in scalability:
        summary_lines.append(
            f"{row['scale']} | {row['records']} | {row['read_time']:.6f} | {row['processing_time']:.6f} | "
            f"{row['analytics_time']:.6f} | {row['total_time']:.6f}"
        )
    (BASE_DIR / "experiment_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")


def main() -> None:
    spark = (
        SparkSession.builder
        .appName("AmazonReviewsBigDataExperiment")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.default.parallelism", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    try:
        raw_df = load_data(spark)
        inspect_data(raw_df)
        cleaned_df, processing = process_data(raw_df)
        cleaned_df.persist(StorageLevel.MEMORY_AND_DISK).count()
        variety = analyze_variety(cleaned_df)
        velocity = analyze_velocity(cleaned_df)
        print("Execution plan for country aggregation:")
        variety["country_counts"].explain()
        print("Execution plan for velocity Window:")
        velocity["velocity_df"].explain()
        sql_results, sql_timings = run_sql_analytics(cleaned_df)
        print("SQL timings:", sql_timings)
        print("Execution plan for Spark SQL:")
        spark.sql("SELECT country, COUNT(*) AS total_reviews FROM reviews GROUP BY country").explain()
        performance = benchmark(spark)
        scalability = run_scalability_test(spark, cleaned_df)
        performance["processing"] = processing
        performance["sql_timings"] = sql_timings
        save_results(cleaned_df, variety, velocity, sql_results, performance, scalability)
        print("Results saved to:", OUTPUT_DIR)
        print("Summary saved to:", BASE_DIR / "experiment_summary.txt")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()