const canvas = document.getElementById("overlayCanvas");
const ctx = canvas.getContext("2d");
const personsPanel = document.getElementById("personsPanel");
const alertsPanel = document.getElementById("alertsPanel");

const metricTracked = document.getElementById("metricTracked");
const metricViolations = document.getElementById("metricViolations");
const metricCompliance = document.getElementById("metricCompliance");
const metricFps = document.getElementById("metricFps");
const liveBadge = document.getElementById("liveBadge");

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
    ctx.strokeStyle = "#005eaa";
    ctx.lineWidth = 1;
    ctx.strokeRect(det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1);
    ctx.fillStyle = "#005eaa";
    ctx.font = "11px Segoe UI";
    ctx.fillText(`${det.label} ${det.conf.toFixed(2)}`, det.x1, Math.max(12, det.y1 - 3));
  }
}

function updatePanels(payload) {
  const metrics = payload.metrics || {};
  metricTracked.textContent = String(metrics.tracked_count ?? 0);
  metricViolations.textContent = String(metrics.active_violations ?? 0);
  metricCompliance.textContent = `${Number(metrics.compliance_rate ?? 0).toFixed(1)}%`;
  metricFps.textContent = String(metrics.fps ?? 0);

  personsPanel.innerHTML = "";
  for (const person of payload.persons || []) {
    const card = document.createElement("div");
    card.className = "person-card";
    card.innerHTML = `<div><strong>Person ${person.person_id}</strong> - ${person.overall_status}</div>`;

    const chipList = document.createElement("div");
    chipList.className = "chip-list";
    for (const [item, state] of Object.entries(person.per_item_state || {})) {
      const chip = document.createElement("span");
      const cls =
        state === "COMPLIANT" ? "ok" : state === "VIOLATION" ? "bad" : "warn";
      chip.className = `chip ${cls}`;
      chip.textContent = `${item}: ${state}`;
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

connect();
