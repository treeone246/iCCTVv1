"""Spark Structured Streaming job for PPE violation Kafka topic."""

from __future__ import annotations

import argparse
import os


def main() -> int:
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import col, from_json, from_unixtime, to_timestamp
        from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType
    except Exception:
        print("pyspark is not installed. Install with: pip install pyspark")
        return 2

    parser = argparse.ArgumentParser(description="Spark stream for PPE violations from Kafka")
    parser.add_argument("--bootstrap-servers", default="127.0.0.1:9092")
    parser.add_argument("--topic", default="ppe.violations.raw")
    parser.add_argument("--checkpoint", default="outputs/spark_checkpoints/ppe_violations")
    parser.add_argument("--output-path", default="outputs/spark_violation_curated")
    parser.add_argument("--trigger-seconds", type=int, default=5)
    parser.add_argument("--postgres-jdbc-url", default="")
    parser.add_argument("--postgres-user", default="")
    parser.add_argument("--postgres-password", default="")
    parser.add_argument("--postgres-table", default="violation_logs_stream")
    args = parser.parse_args()

    spark = (
        SparkSession.builder.appName("ppe-violation-stream")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    alert_schema = StructType(
        [
            StructField("alert_id", StringType(), True),
            StructField("person_id", IntegerType(), True),
            StructField("display_id", StringType(), True),
            StructField("item", StringType(), True),
            StructField("status", StringType(), True),
            StructField("reason", StringType(), True),
            StructField("timestamp", DoubleType(), True),
            StructField("negative_conf", DoubleType(), True),
        ]
    )
    envelope_schema = StructType(
        [
            StructField("event_type", StringType(), True),
            StructField("camera_id", StringType(), True),
            StructField("published_ts", DoubleType(), True),
            StructField("alert", alert_schema, True),
        ]
    )

    source_df = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", args.bootstrap_servers)
        .option("subscribe", args.topic)
        .option("startingOffsets", "latest")
        .load()
    )

    parsed = source_df.selectExpr("CAST(value AS STRING) AS json_value").select(
        from_json(col("json_value"), envelope_schema).alias("doc")
    )

    flat = parsed.select(
        col("doc.camera_id").alias("camera_id"),
        col("doc.alert.alert_id").alias("alert_id"),
        col("doc.alert.person_id").alias("person_id"),
        col("doc.alert.display_id").alias("display_id"),
        col("doc.alert.item").alias("ppe_item"),
        col("doc.alert.status").alias("status"),
        col("doc.alert.reason").alias("reason"),
        col("doc.alert.negative_conf").alias("confidence"),
        to_timestamp(from_unixtime(col("doc.alert.timestamp"))).alias("event_time"),
    )

    # Filter and dedupe within watermark to prevent repeated spam rows.
    curated = (
        flat.filter(col("status") == "ACTIVE")
        .filter(col("confidence") >= 0.2)
        .withWatermark("event_time", "5 minutes")
        .dropDuplicates(["camera_id", "display_id", "ppe_item", "status"])
    )

    checkpoint_dir = args.checkpoint
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_path, exist_ok=True)

    if args.postgres_jdbc_url:
        def _write_batch(df, epoch_id):  # type: ignore
            (
                df.write.mode("append")
                .format("jdbc")
                .option("url", args.postgres_jdbc_url)
                .option("dbtable", args.postgres_table)
                .option("user", args.postgres_user)
                .option("password", args.postgres_password)
                .save()
            )

        query = (
            curated.writeStream.foreachBatch(_write_batch)
            .outputMode("append")
            .option("checkpointLocation", checkpoint_dir)
            .trigger(processingTime=f"{args.trigger_seconds} seconds")
            .start()
        )
    else:
        query = (
            curated.writeStream.format("parquet")
            .outputMode("append")
            .option("path", args.output_path)
            .option("checkpointLocation", checkpoint_dir)
            .trigger(processingTime=f"{args.trigger_seconds} seconds")
            .start()
        )

    query.awaitTermination()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
