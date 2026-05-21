const canvas = document.getElementById("overlayCanvas");
const ctx = canvas.getContext("2d");
const personsPanel = document.getElementById("personsPanel");
const alertsPanel = document.getElementById("alertsPanel");
const ppeStatusPanel = document.getElementById("ppeStatusPanel");

const metricTracked = document.getElementById("metricTracked");
const metricViolations = document.getElementById("metricViolations");
const metricCompliance = document.getElementById("metricCompliance");
const metricFps = document.getElementById("metricFps");
const liveBadge = document.getElementById("liveBadge");
const computeOverall = document.getElementById("computeOverall");
const computeModels = document.getElementById("computeModels");
const computeMemory = document.getElementById("computeMemory");
const computeJetson = document.getElementById("computeJetson");
const behaviorSummary = document.getElementById("behaviorSummary");
const behaviorMeta = document.getElementById("behaviorMeta");
const behaviorPatterns = document.getElementById("behaviorPatterns");
const behaviorReview = document.getElementById("behaviorReview");
const behaviorTraining = document.getElementById("behaviorTraining");

const settingsBtn = document.getElementById("settingsBtn");
const closeSettingsBtn = document.getElementById("closeSettingsBtn");
const settingsDrawer = document.getElementById("settingsDrawer");

const STATUS_COLOR = {
  COMPLIANT: "#1f9d55",
  VIOLATION: "#d93025",
  INDETERMINATE: "#d97a00",
};

const acknowledgedAlerts = new Set();
let pendingBlob = null;
const REQUIRED_ITEMS = ["helmet", "gloves", "coverall", "boots", "goggles"];

settingsBtn.addEventListener("click", () => settingsDrawer.classList.remove("hidden"));
closeSettingsBtn.addEventListener("click", () => settingsDrawer.classList.add("hidden"));

function connect() {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${protocol}://${window.location.host}/ws/stream`);
  ws.binaryType = "blob";

  ws.onopen = () => {
    liveBadge.textContent = "LIVE";
    liveBadge.style.color = "#d93025";
  };

  ws.onclose = () => {
    liveBadge.textContent = "DISCONNECTED";
    liveBadge.style.color = "#536271";
    setTimeout(connect, 1500);
  };

  ws.onmessage = async (event) => {
    if (typeof event.data === "string") {
      const metadata = JSON.parse(event.data);
      if (pendingBlob) {
        await drawFrameAndOverlay(pendingBlob, metadata);
        pendingBlob = null;
      }
      updatePanels(metadata);
      return;
    }
    pendingBlob = event.data;
  };
}

async function drawFrameAndOverlay(blob, metadata) {
  const bitmap = await createImageBitmap(blob);
  if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
  }

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);

  drawPeople(metadata.persons || []);
  drawPPEDetections(metadata.ppe_detections || []);
}

function drawPeople(persons) {
  for (const person of persons) {
    const color = STATUS_COLOR[person.overall_status] || "#536271";
    const [x1, y1, x2, y2] = person.bbox;
    ctx.lineWidth = 1;
    ctx.strokeStyle = color;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    ctx.fillStyle = color;
    ctx.font = "12px Segoe UI";
    ctx.fillText(`ID ${person.person_id}`, x1, Math.max(12, y1 - 4));

    drawSkeleton(person.keypoints, color);
  }
}

function drawSkeleton(keypoints, color) {
  const pairs = [
    ["left_shoulder", "right_shoulder"],
    ["left_shoulder", "left_elbow"],
    ["left_elbow", "left_wrist"],
    ["right_shoulder", "right_elbow"],
    ["right_elbow", "right_wrist"],
    ["left_shoulder", "left_hip"],
    ["right_shoulder", "right_hip"],
    ["left_hip", "right_hip"],
    ["left_hip", "left_knee"],
    ["left_knee", "left_ankle"],
    ["right_hip", "right_knee"],
    ["right_knee", "right_ankle"],
  ];

  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  for (const [a, b] of pairs) {
    const pa = keypoints[a];
    const pb = keypoints[b];
    if (!pa || !pb) {
      continue;
    }
    if (pa.conf < 0.4 || pb.conf < 0.4) {
      continue;
    }
    ctx.beginPath();
    ctx.moveTo(pa.x, pa.y);
    ctx.lineTo(pb.x, pb.y);
    ctx.stroke();
  }

  for (const kp of Object.values(keypoints)) {
    if (kp.conf < 0.4) {
      continue;
    }
    ctx.beginPath();
    ctx.arc(kp.x, kp.y, 2.5, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
  }
}

function drawPPEDetections(detections) {
  for (const det of detections) {
    const isYoloe = det.source === "yoloe_aux";
    ctx.strokeStyle = isYoloe ? "#13a6a6" : "#005eaa";
    ctx.lineWidth = 1;
    ctx.strokeRect(det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1);
    ctx.fillStyle = isYoloe ? "#13a6a6" : "#005eaa";
    ctx.font = "11px Segoe UI";
    const src = isYoloe ? "YOLOE" : "BEST";
    ctx.fillText(`${det.label} ${det.conf.toFixed(2)} [${src}]`, det.x1, Math.max(12, det.y1 - 3));
  }
}

function normalizeState(state) {
  if (state === "COMPLIANT" || state === "VIOLATION" || state === "INDETERMINATE") {
    return state;
  }
  return "INDETERMINATE";
}

function makeItemSummary(persons) {
  const summary = {};
  for (const item of REQUIRED_ITEMS) {
    summary[item] = { COMPLIANT: 0, VIOLATION: 0, INDETERMINATE: 0, total: 0 };
  }

  for (const person of persons) {
    const states = person.per_item_state || {};
    for (const item of REQUIRED_ITEMS) {
      const state = normalizeState(states[item]);
      summary[item][state] += 1;
      summary[item].total += 1;
    }
  }
  return summary;
}

function renderPPEStatusDashboard(persons, metrics) {
  ppeStatusPanel.innerHTML = "";
  const summary = makeItemSummary(persons);
  const personRollup = { COMPLIANT: 0, VIOLATION: 0, INDETERMINATE: 0 };
  for (const person of persons) {
    const state = normalizeState(person.overall_status);
    personRollup[state] += 1;
  }

  const rollupCard = document.createElement("div");
  rollupCard.className = "ppe-status-card status-neutral";
  rollupCard.innerHTML = `
    <div class="ppe-status-title">PERSON STATUS</div>
    <div class="ppe-status-stats">OK: ${personRollup.COMPLIANT} | BAD: ${personRollup.VIOLATION} | UNK: ${personRollup.INDETERMINATE}</div>
  `;
  ppeStatusPanel.appendChild(rollupCard);

  const runtimeCard = document.createElement("div");
  runtimeCard.className = "ppe-status-card status-neutral";
  const modelName = (metrics.ppe_model || "").split("/").pop() || "unknown";
  runtimeCard.innerHTML = `
    <div class="ppe-status-title">MODEL RUNTIME</div>
    <div class="ppe-status-stats">${modelName} (${metrics.ppe_task || "detect"}, fusion:${metrics.ppe_fusion_mode || "nms"})</div>
    <div class="ppe-status-stats">raw/frame BEST2:${metrics.ppe_primary_raw ?? 0} YOLOE:${metrics.verifier_aux_raw ?? 0} merged:${metrics.ppe_merged ?? 0}</div>
    <div class="ppe-status-stats">infer calls Pose:${metrics.pose_infer_calls ?? 0} BEST2:${metrics.ppe_infer_calls ?? 0} YOLOE(aux):${metrics.verifier_aux_infer_calls ?? 0} verifier(crop):${metrics.verifier_crop_infer_calls ?? 0} ollama:${metrics.verifier_ollama_calls ?? 0}</div>
  `;
  ppeStatusPanel.appendChild(runtimeCard);

  for (const item of REQUIRED_ITEMS) {
    const row = summary[item];
    const card = document.createElement("div");
    const dominantState =
      row.VIOLATION > 0 ? "VIOLATION" : row.COMPLIANT > 0 ? "COMPLIANT" : "INDETERMINATE";

    const stateClass =
      dominantState === "COMPLIANT"
        ? "status-good"
        : dominantState === "VIOLATION"
        ? "status-bad"
        : "status-warn";

    card.className = `ppe-status-card ${stateClass}`;
    card.innerHTML = `
      <div class="ppe-status-title">${item.toUpperCase()}</div>
      <div class="ppe-status-stats">OK: ${row.COMPLIANT} | BAD: ${row.VIOLATION} | UNK: ${row.INDETERMINATE}</div>
    `;
    ppeStatusPanel.appendChild(card);
  }
}

function renderComputePanel(metrics) {
  const totalG = Number(metrics.estimated_gflops_per_sec ?? 0);
  const totalT = Number(metrics.estimated_tflops_per_sec ?? 0);
  const totalTOPS = Number(metrics.estimated_tops_per_sec ?? 0);
  const totalF = Number(metrics.estimated_flops_per_sec ?? 0);
  const util = Number(metrics.estimated_compute_utilization_pct ?? 0);
  computeOverall.textContent =
    `Total: ${totalG.toFixed(1)} GFLOP/s (${totalT.toFixed(3)} TFLOP/s, ${totalTOPS.toFixed(3)} TOPS)\n` +
    `Absolute: ${totalF.toLocaleString()} FLOP/s\n` +
    `Estimated utilization: ${util.toFixed(1)}%`;

  computeModels.textContent =
    `Pose: ${Number(metrics.pose_estimated_gflops_per_sec ?? 0).toFixed(1)} GFLOP/s @ ${Number(metrics.pose_infer_per_sec ?? 0).toFixed(1)} infer/s\n` +
    `PPE BEST2: ${Number(metrics.ppe_estimated_gflops_per_sec ?? 0).toFixed(1)} GFLOP/s @ ${Number(metrics.ppe_infer_per_sec ?? 0).toFixed(1)} infer/s\n` +
    `Verifier YOLOE(aux): ${Number(metrics.verifier_aux_estimated_gflops_per_sec ?? 0).toFixed(1)} GFLOP/s @ ${Number(metrics.verifier_aux_infer_per_sec ?? 0).toFixed(1)} infer/s\n` +
    `Verifier YOLOE(crop): ${Number(metrics.verifier_crop_estimated_gflops_per_sec ?? 0).toFixed(1)} GFLOP/s @ ${Number(metrics.verifier_crop_infer_per_sec ?? 0).toFixed(1)} infer/s\n` +
    `Verifier Ollama: ${Number(metrics.verifier_ollama_estimated_gflops_per_sec ?? 0).toFixed(1)} GFLOP/s @ ${Number(metrics.verifier_ollama_calls_per_sec ?? 0).toFixed(1)} call/s`;

  const procRss = Number(metrics.process_rss_mb ?? 0);
  const procVms = Number(metrics.process_vms_mb ?? 0);
  const sysUsed = Number(metrics.system_memory_used_mb ?? 0);
  const sysTotal = Number(metrics.system_memory_total_mb ?? 0);
  const sysPct = Number(metrics.system_memory_utilization_pct ?? 0);
  computeMemory.textContent =
    `Process RSS: ${procRss.toFixed(1)} MB\n` +
    `Process VMS: ${procVms.toFixed(1)} MB\n` +
    `System: ${sysUsed.toFixed(1)} / ${sysTotal.toFixed(1)} MB (${sysPct.toFixed(1)}%)`;
}

function updatePanels(payload) {
  const metrics = payload.metrics || {};
  metricTracked.textContent = String(metrics.tracked_count ?? 0);
  metricViolations.textContent = String(metrics.active_violations ?? 0);
  metricCompliance.textContent = `${Number(metrics.compliance_rate ?? 0).toFixed(1)}%`;
  metricFps.textContent = String(metrics.fps ?? 0);

  const persons = payload.persons || [];
  renderPPEStatusDashboard(persons, metrics);
  renderComputePanel(metrics);

  personsPanel.innerHTML = "";
  for (const person of persons) {
    const card = document.createElement("div");
    card.className = "person-card";
    card.innerHTML = `<div><strong>Person ${person.person_id}</strong> - ${person.overall_status}</div>`;

    const chipList = document.createElement("div");
    chipList.className = "chip-list";
    for (const [item, state] of Object.entries(person.per_item_state || {})) {
      const reason = (person.per_item_reason || {})[item] || "";
      const chip = document.createElement("span");
      const cls =
        state === "COMPLIANT" ? "ok" : state === "VIOLATION" ? "bad" : "warn";
      chip.className = `chip ${cls}`;
      chip.textContent = `${item}: ${state}${reason ? ` (${reason})` : ""}`;
      if (reason) {
        chip.title = reason;
      }
      chipList.appendChild(chip);
    }
    card.appendChild(chipList);
    personsPanel.appendChild(card);
  }

  alertsPanel.innerHTML = "";
  const activeIds = new Set((payload.active_alerts || []).map((a) => a.alert_id));
  for (const existingId of Array.from(acknowledgedAlerts)) {
    if (!activeIds.has(existingId)) {
      acknowledgedAlerts.delete(existingId);
    }
  }

  for (const alert of payload.active_alerts || []) {
    const card = document.createElement("div");
    card.className = "alert-card";
    const ts = new Date((alert.timestamp || 0) * 1000).toLocaleTimeString();
    card.innerHTML = `
      <div><strong>${alert.item}</strong> - Person ${alert.person_id}</div>
      <div>${alert.reason}</div>
      <div>${ts}</div>
    `;

    if (alert.evidence_jpeg_base64) {
      const image = document.createElement("img");
      image.className = "alert-evidence";
      image.alt = `Evidence for ${alert.item} person ${alert.person_id}`;
      image.src = `data:image/jpeg;base64,${alert.evidence_jpeg_base64}`;
      card.appendChild(image);
    }

    const ackBtn = document.createElement("button");
    ackBtn.className = "icon-btn";
    const isAcknowledged = acknowledgedAlerts.has(alert.alert_id);
    ackBtn.textContent = isAcknowledged ? "Acknowledged" : "Acknowledge";
    ackBtn.disabled = isAcknowledged;
    ackBtn.onclick = () => {
      acknowledgedAlerts.add(alert.alert_id);
      ackBtn.textContent = "Acknowledged";
      ackBtn.disabled = true;
    };
    card.appendChild(ackBtn);
    alertsPanel.appendChild(card);
  }
}

function formatLocalTime(ts) {
  if (!ts) {
    return "N/A";
  }
  const date = new Date(ts);
  if (Number.isNaN(date.getTime())) {
    return String(ts);
  }
  return date.toLocaleString();
}

function renderBehaviorPanel(latest, memory) {
  const model = latest?.model || "qwen3:4b";
  const generatedAt = latest?.generated_at || "";
  const summary = String(latest?.summary || "").trim();
  behaviorSummary.textContent = summary || "No behavior summary available yet.";
  behaviorMeta.textContent = `Model: ${model}\nGenerated: ${formatLocalTime(generatedAt)}`;

  const patterns = Array.isArray(latest?.detected_patterns) ? latest.detected_patterns : [];
  if (patterns.length === 0) {
    behaviorPatterns.textContent = "None";
  } else {
    const text = patterns
      .slice(0, 6)
      .map((item) => {
        const type = item?.type || "unknown";
        const conf = item?.confidence != null ? ` (${Number(item.confidence).toFixed(2)})` : "";
        const detail = item?.description || item?.evidence || "";
        return `${type}${conf}${detail ? `: ${detail}` : ""}`;
      })
      .join("\n");
    behaviorPatterns.textContent = text;
  }

  const reviewTracks = [];
  for (const [trackId, entry] of Object.entries(memory || {})) {
    if (!entry || typeof entry !== "object") {
      continue;
    }
    if (entry.review_needed || entry.possible_identity_switch) {
      reviewTracks.push(`Track ${trackId}`);
    }
  }
  behaviorReview.textContent = reviewTracks.length > 0 ? reviewTracks.join(", ") : "None";

  const suggestions = Array.isArray(latest?.training_data_suggestions)
    ? latest.training_data_suggestions
    : [];
  if (suggestions.length === 0) {
    behaviorTraining.textContent = "None";
  } else {
    behaviorTraining.textContent = suggestions
      .slice(0, 6)
      .map((item) => {
        const track = item?.track_id != null ? `Track ${item.track_id}` : "Track ?";
        const hint = item?.label_hint || item?.type || "label_hint_missing";
        const reason = item?.reason ? ` (${item.reason})` : "";
        return `${track}: ${hint}${reason}`;
      })
      .join("\n");
  }
}

async function refreshBehaviorPanel() {
  try {
    const [latestResp, memoryResp] = await Promise.all([
      fetch("/api/behavior-agent/latest"),
      fetch("/api/behavior-agent/memory"),
    ]);
    const latest = latestResp.ok ? await latestResp.json() : {};
    const memory = memoryResp.ok ? await memoryResp.json() : {};
    renderBehaviorPanel(latest, memory);
  } catch (err) {
    behaviorSummary.textContent = "Behavior API unavailable.";
    behaviorMeta.textContent = "No data yet";
    behaviorPatterns.textContent = "None";
    behaviorReview.textContent = "None";
    behaviorTraining.textContent = "None";
  }
}

function renderJetsonPanel(payload) {
  if (!payload || payload.enabled !== true) {
    computeJetson.textContent = "Jetson exporter integration disabled in config.";
    return;
  }
  if (payload.available !== true) {
    computeJetson.textContent = `Exporter unavailable.\n${payload.error || "No response"}\nSource: ${payload.source_url || "unknown"}`;
    return;
  }
  computeJetson.textContent =
    `CPU: ${Number(payload.cpu_utilization_pct ?? 0).toFixed(1)}%\n` +
    `GPU: ${Number(payload.gpu_utilization_pct ?? 0).toFixed(1)}%\n` +
    `Memory: ${Number(payload.memory_utilization_pct ?? 0).toFixed(1)}% (${Number(payload.memory_used_mb ?? 0).toFixed(1)} / ${Number(payload.memory_total_mb ?? 0).toFixed(1)} MB)\n` +
    `Temp: ${Number(payload.temperature_c ?? 0).toFixed(1)} C\n` +
    `Power: ${Number(payload.power_w ?? 0).toFixed(2)} W\n` +
    `Fan PWM: ${Number(payload.fan_pwm_pct ?? 0).toFixed(1)}%\n` +
    `Source: ${payload.source_url || "unknown"}`;
}

async function refreshJetsonPanel() {
  try {
    const resp = await fetch("/api/jetson/stats");
    const payload = resp.ok ? await resp.json() : { enabled: false, available: false, error: "http_error" };
    renderJetsonPanel(payload);
  } catch (err) {
    computeJetson.textContent = "Jetson exporter API unavailable.";
  }
}

connect();
refreshBehaviorPanel();
refreshJetsonPanel();
setInterval(refreshBehaviorPanel, 5000);
setInterval(refreshJetsonPanel, 5000);
