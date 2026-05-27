CREATE TABLE IF NOT EXISTS violation_logs_flink (
  violation_id TEXT PRIMARY KEY,
  event_time TIMESTAMPTZ NOT NULL,
  device_id TEXT NOT NULL,
  camera_id TEXT NOT NULL,
  person_display_id TEXT NOT NULL,
  ppe_item TEXT NOT NULL,
  status TEXT NOT NULL,
  confidence DOUBLE PRECISION,
  local_roi_path TEXT,
  roi_exists BOOLEAN DEFAULT FALSE,
  reason TEXT,
  created_at TIMESTAMPTZ DEFAULT now(),
  first_seen_ts DOUBLE PRECISION,
  last_seen_ts DOUBLE PRECISION,
  event_count BIGINT DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_violation_flink_time
  ON violation_logs_flink(event_time DESC);

CREATE INDEX IF NOT EXISTS idx_violation_flink_camera_time
  ON violation_logs_flink(camera_id, event_time DESC);

CREATE INDEX IF NOT EXISTS idx_violation_flink_person_item
  ON violation_logs_flink(person_display_id, ppe_item);
