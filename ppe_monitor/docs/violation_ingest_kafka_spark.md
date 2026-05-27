# Violation Ingest (Filter/Queue/Kafka/Flink/Spark)

This app now has a filtered ingest path for violation alerts before PostgreSQL:

1. Active alerts from pipeline  
2. `violation_ingest.filter` suppression rules  
3. `violation_ingest.queue` batch queue  
4. PostgreSQL sink (`postgres_logging`)  
5. Optional Kafka fan-out (`violation_ingest.mode: kafka`)  

## 1) Critical DB requirement

If `postgres_logging.password` is blank in `config.yaml`, set env var before starting uvicorn:

```bash
export PPE_MONITOR_PG_PASSWORD='sepeed246'
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 2) Runtime status endpoints

```bash
curl -s http://127.0.0.1:8000/api/violation-ingest/status
curl -s http://127.0.0.1:8000/api/postgres-logging/status
```

Expected healthy signs:
- `db_password_set: true`
- `connect_fail_count: 0` (or stable, not increasing)
- `inserted_count` increasing when alerts are active

## 3) Check database writes

```bash
PGPASSWORD='sepeed246' psql -h 127.0.0.1 -U ppe_app -d ppe_monitor -c \
"SELECT event_time,camera_id,person_display_id,ppe_item,status,confidence,local_roi_path \
 FROM violation_logs ORDER BY created_at DESC LIMIT 20;"
```

## 4) Anti-spam controls (config.yaml)

Tune these in `violation_ingest.filter`:
- `repeat_suppression_seconds`
- `max_events_per_minute_per_key`
- `min_negative_confidence_default`
- `min_negative_confidence_per_item`

## 5) Optional Kafka mode

Install optional deps:

```bash
pip install -r requirements-kafka-spark.txt
```

Set config:

```yaml
violation_ingest:
  mode: "kafka"
  kafka:
    enabled: true
    bootstrap_servers: "127.0.0.1:9092"
    topic: "ppe.violations.raw"
    also_write_local: true
```

Run app normally (producer side), then start consumer worker:

```bash
python scripts/run_violation_kafka_consumer.py --config config.yaml
```

## 6) Flink stream processor (recommended for real-time backend)

Flink lets you keep Jetson lightweight while handling dedupe/state/aggregation centrally.

1. Set app to publish-only mode:

```yaml
violation_ingest:
  mode: "kafka_flink"
  kafka:
    enabled: true
    also_write_local: false
  flink:
    enabled: true
```

2. Create sink table in PostgreSQL (run once):

```bash
psql -h 127.0.0.1 -U ppe_app -d ppe_monitor -f scripts/sql/create_violation_logs_flink.sql
```

3. Start Flink job:

```bash
python scripts/flink_violation_stream.py \
  --config config.yaml \
  --pipeline-jars "file:///opt/flink/lib/flink-sql-connector-kafka.jar;file:///opt/flink/lib/flink-connector-jdbc.jar;file:///opt/flink/lib/postgresql.jar"
```

4. Monitor app + ingest status:

```bash
curl -s http://127.0.0.1:8000/api/violation-ingest/status
```

In `kafka_flink` mode, app status should show `local_sink_enabled: false` and Kafka publish counters increasing.

## 7) Optional Spark stream job

Read Kafka topic and produce curated stream output:

```bash
python scripts/spark_violation_stream.py \
  --bootstrap-servers 127.0.0.1:9092 \
  --topic ppe.violations.raw \
  --checkpoint outputs/spark_checkpoints/ppe_violations \
  --output-path outputs/spark_violation_curated
```

You can also write Spark batches to PostgreSQL using `--postgres-jdbc-url ...`.
