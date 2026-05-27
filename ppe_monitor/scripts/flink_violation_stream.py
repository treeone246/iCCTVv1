"""Apache Flink streaming job: Kafka PPE alerts -> curated PostgreSQL table.

This job is intended to run on a server/NUC (not necessarily on Jetson).
It consumes `ppe.violations.raw` produced by the PPE monitor app, applies
stream filtering and 1-minute keyed aggregation (anti-spam), then writes
curated rows into PostgreSQL via Flink JDBC sink.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _sql_escape(value: str) -> str:
    return str(value).replace("'", "''")


def _build_per_item_condition(
    *,
    per_item_thresholds: dict[str, Any],
    default_threshold: float,
) -> str:
    clauses: list[str] = []
    for item, threshold in per_item_thresholds.items():
        try:
            thr = float(threshold)
        except (TypeError, ValueError):
            continue
        item_l = _sql_escape(str(item).strip().lower())
        clauses.append(f"(LOWER(alert.item) = '{item_l}' AND COALESCE(alert.negative_conf, 0.0) >= {thr})")
    default_clause = f"(COALESCE(alert.negative_conf, 0.0) >= {float(default_threshold)})"
    if not clauses:
        return default_clause
    # Item-specific thresholds first, then default fallback.
    return "(" + " OR ".join(clauses + [default_clause]) + ")"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Flink stream for PPE violation curation")
    parser.add_argument("--config", default="config.yaml", help="Path to app config.yaml")
    parser.add_argument("--topic", default="", help="Kafka topic override")
    parser.add_argument("--bootstrap-servers", default="", help="Kafka bootstrap override")
    parser.add_argument("--group-id", default="", help="Kafka group override")
    parser.add_argument("--postgres-host", default="", help="PostgreSQL host override")
    parser.add_argument("--postgres-port", type=int, default=0, help="PostgreSQL port override")
    parser.add_argument("--postgres-db", default="", help="PostgreSQL database override")
    parser.add_argument("--postgres-user", default="", help="PostgreSQL user override")
    parser.add_argument("--postgres-password", default="", help="PostgreSQL password override")
    parser.add_argument("--sink-table", default="", help="Curated sink table name override")
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=0,
        help="Aggregation window seconds override (default from config or 60)",
    )
    parser.add_argument(
        "--pipeline-jars",
        default="",
        help=(
            "Semicolon-separated list of connector jars "
            "(Kafka + JDBC + PostgreSQL driver)."
        ),
    )
    args = parser.parse_args()

    try:
        from pyflink.table import EnvironmentSettings, TableEnvironment
    except Exception as exc:  # pragma: no cover
        print("PyFlink is required. Install on Flink worker host with: pip install pyflink")
        print(f"Import error: {type(exc).__name__}: {exc}")
        return 2

    project_root = Path(__file__).resolve().parent.parent
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (project_root / cfg_path).resolve()
    config = _load_config(cfg_path)

    ingest_cfg = config.get("violation_ingest", {}) or {}
    kafka_cfg = ingest_cfg.get("kafka", {}) or {}
    filter_cfg = ingest_cfg.get("filter", {}) or {}
    pg_cfg = config.get("postgres_logging", {}) or {}
    flink_cfg = ingest_cfg.get("flink", {}) or {}

    topic = args.topic or str(kafka_cfg.get("topic", "ppe.violations.raw"))
    bootstrap = args.bootstrap_servers or str(kafka_cfg.get("bootstrap_servers", "127.0.0.1:9092"))
    group_id = args.group_id or str(flink_cfg.get("group_id", kafka_cfg.get("group_id", "ppe-flink-curator")))

    pg_host = args.postgres_host or str(pg_cfg.get("host", "127.0.0.1"))
    pg_port = int(args.postgres_port or int(pg_cfg.get("port", 5432)))
    pg_db = args.postgres_db or str(pg_cfg.get("database", "ppe_monitor"))
    pg_user = args.postgres_user or str(pg_cfg.get("user", "ppe_app"))
    pg_password = args.postgres_password or str(pg_cfg.get("password", ""))
    sink_table = args.sink_table or str(flink_cfg.get("sink_table", "violation_logs_flink"))
    window_seconds = int(args.window_seconds or int(flink_cfg.get("window_seconds", 60)))
    if window_seconds <= 0:
        window_seconds = 60

    per_item_thresholds = filter_cfg.get("min_negative_confidence_per_item", {}) or {}
    default_threshold = float(filter_cfg.get("min_negative_confidence_default", 0.0))
    negative_conf_sql = _build_per_item_condition(
        per_item_thresholds=per_item_thresholds,
        default_threshold=default_threshold,
    )

    env_settings = EnvironmentSettings.in_streaming_mode()
    t_env = TableEnvironment.create(env_settings)

    if args.pipeline_jars:
        t_env.get_config().set("pipeline.jars", args.pipeline_jars)
    t_env.get_config().set("parallelism.default", str(int(flink_cfg.get("parallelism", 1))))
    t_env.get_config().set("table.exec.state.ttl", str(flink_cfg.get("state_ttl", "30 min")))

    source_ddl = f"""
    CREATE TEMPORARY TABLE raw_violations (
      event_type STRING,
      camera_id STRING,
      published_ts DOUBLE,
      alert ROW<
        alert_id STRING,
        person_id INT,
        display_id STRING,
        item STRING,
        status STRING,
        reason STRING,
        timestamp DOUBLE,
        evidence_available BOOLEAN,
        person_status STRING,
        helmet_color STRING,
        positive_conf DOUBLE,
        negative_conf DOUBLE
      >,
      proc_time AS PROCTIME()
    ) WITH (
      'connector' = 'kafka',
      'topic' = '{_sql_escape(topic)}',
      'properties.bootstrap.servers' = '{_sql_escape(bootstrap)}',
      'properties.group.id' = '{_sql_escape(group_id)}',
      'scan.startup.mode' = 'latest-offset',
      'format' = 'json',
      'json.fail-on-missing-field' = 'false',
      'json.ignore-parse-errors' = 'true'
    )
    """

    sink_ddl = f"""
    CREATE TEMPORARY TABLE curated_violations (
      violation_id STRING,
      event_time TIMESTAMP(3),
      device_id STRING,
      camera_id STRING,
      person_display_id STRING,
      ppe_item STRING,
      status STRING,
      confidence DOUBLE,
      local_roi_path STRING,
      roi_exists BOOLEAN,
      reason STRING,
      created_at TIMESTAMP(3),
      first_seen_ts DOUBLE,
      last_seen_ts DOUBLE,
      event_count BIGINT,
      PRIMARY KEY (violation_id) NOT ENFORCED
    ) WITH (
      'connector' = 'jdbc',
      'url' = 'jdbc:postgresql://{_sql_escape(pg_host)}:{pg_port}/{_sql_escape(pg_db)}',
      'table-name' = '{_sql_escape(sink_table)}',
      'username' = '{_sql_escape(pg_user)}',
      'password' = '{_sql_escape(pg_password)}',
      'driver' = 'org.postgresql.Driver',
      'sink.buffer-flush.max-rows' = '{int(flink_cfg.get("sink_buffer_max_rows", 200))}',
      'sink.buffer-flush.interval' = '{int(flink_cfg.get("sink_buffer_interval_ms", 1000))}ms',
      'sink.max-retries' = '3'
    )
    """

    insert_sql = f"""
    INSERT INTO curated_violations
    SELECT
      MD5(
        CONCAT(
          camera_id, '|',
          COALESCE(NULLIF(alert.display_id, ''), CONCAT('person_', CAST(alert.person_id AS STRING))), '|',
          LOWER(alert.item), '|',
          CAST(TUMBLE_START(proc_time, INTERVAL '{window_seconds}' SECOND) AS STRING)
        )
      ) AS violation_id,
      TUMBLE_END(proc_time, INTERVAL '{window_seconds}' SECOND) AS event_time,
      '{_sql_escape(str(pg_cfg.get("device_id", "jetson-edge-01")))}' AS device_id,
      camera_id,
      COALESCE(NULLIF(alert.display_id, ''), CONCAT('person_', CAST(alert.person_id AS STRING))) AS person_display_id,
      LOWER(alert.item) AS ppe_item,
      'ACTIVE' AS status,
      MAX(COALESCE(alert.negative_conf, 0.0)) AS confidence,
      CAST(NULL AS STRING) AS local_roi_path,
      FALSE AS roi_exists,
      MAX(COALESCE(alert.reason, '')) AS reason,
      CURRENT_TIMESTAMP AS created_at,
      MIN(COALESCE(alert.timestamp, published_ts)) AS first_seen_ts,
      MAX(COALESCE(alert.timestamp, published_ts)) AS last_seen_ts,
      COUNT(*) AS event_count
    FROM raw_violations
    WHERE UPPER(COALESCE(alert.status, '')) = 'ACTIVE'
      AND alert.item IS NOT NULL
      AND {negative_conf_sql}
    GROUP BY
      TUMBLE(proc_time, INTERVAL '{window_seconds}' SECOND),
      camera_id,
      COALESCE(NULLIF(alert.display_id, ''), CONCAT('person_', CAST(alert.person_id AS STRING))),
      LOWER(alert.item)
    """

    print("Creating source and sink tables...")
    t_env.execute_sql(source_ddl)
    t_env.execute_sql(sink_ddl)
    print("Starting Flink streaming insert job...")
    table_result = t_env.execute_sql(insert_sql)
    # Keep job alive.
    table_result.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
