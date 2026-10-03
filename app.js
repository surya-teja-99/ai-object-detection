/* VisionScope — real-time in-browser object detection.
 *
 * Pipeline: getUserMedia camera feed -> COCO-SSD (TensorFlow.js, on-device)
 * -> bounding boxes drawn on a canvas overlay + live detection list.
 * No video ever leaves the browser.
 */
"use strict";

const els = {
  video: document.getElementById("video"),
  canvas: document.getElementById("overlay"),
  banner: document.getElementById("banner"),
  aiToggle: document.getElementById("aiToggle"),
  fps: document.getElementById("fps"),
  fpsVal: document.getElementById("fpsVal"),
  confidence: document.getElementById("confidence"),
  confVal: document.getElementById("confVal"),
  flipCam: document.getElementById("flipCam"),
  snapshot: document.getElementById("snapshot"),
  detList: document.getElementById("detList"),
  detCount: document.getElementById("detCount"),
  fpsMeter: document.getElementById("fpsMeter"),
  modelStatus: document.getElementById("modelStatus"),
};

const state = {
  model: null,
  modelReady: false,
  aiEnabled: true,
  targetFps: 8,
  minScore: 0.5,
  facingMode: "environment",
  stream: null,
  lastPredictions: [],
  lastDetectAt: 0,
  detectCount: 0,
  detectWindowStart: performance.now(),
};

const COLORS = [
  "#4cc38a", "#5aa9ff", "#e5b567", "#ef6f6c", "#b388ff",
  "#ff9f6e", "#63d6d6", "#f2e394",
];

function showBanner(msg) {
  els.banner.textContent = msg;
  els.banner.classList.remove("hidden");
}

function hideBanner() {
  els.banner.classList.add("hidden");
}

function classColor(name) {
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) >>> 0;
  return COLORS[h % COLORS.length];
}

async function loadModel() {
  try {
    els.modelStatus.textContent = "loading model…";
    // cocoSsd is exposed by the @tensorflow-models/coco-ssd CDN bundle.
    state.model = await cocoSsd.load();
    state.modelReady = true;
    els.modelStatus.textContent = "model ready · COCO-SSD";
  } catch (err) {
    console.error(err);
    els.modelStatus.textContent = "model failed to load";
    showBanner("Could not load the detection model. Check your connection and reload.");
  }
}

async function startCamera() {
  if (state.stream) {
    state.stream.getTracks().forEach((t) => t.stop());
  }
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: state.facingMode },
    });
    els.video.srcObject = state.stream;
    await els.video.play();
    hideBanner();
    resizeCanvas();
  } catch (err) {
    console.error(err);
    showBanner(
      "Camera unavailable. Allow camera access and serve this page over localhost or HTTPS, then reload."
    );
  }
}

function resizeCanvas() {
  const v = els.video;
  if (v.videoWidth === 0) return;
  els.canvas.width = v.videoWidth;
  els.canvas.height = v.videoHeight;
}

function drawPredictions(predictions) {
  const ctx = els.canvas.getContext("2d");
  const v = els.video;
  ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);
  // Mirror the overlay when using the front camera so boxes track the preview.
  const mirror = state.facingMode === "user";
  for (const p of predictions) {
    let [x, y, w, h] = p.bbox;
    if (mirror) x = v.videoWidth - x - w;
    const color = classColor(p.class);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);
    const label = `${p.class} ${Math.round(p.score * 100)}%`;
    ctx.font = "14px system-ui, sans-serif";
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = color;
    ctx.fillRect(x, y - 20, tw + 12, 20);
    ctx.fillStyle = "#0e1116";
    ctx.fillText(label, x + 6, y - 6);
  }
}

function renderList(predictions) {
  els.detCount.textContent = predictions.length;
  if (predictions.length === 0) {
    els.detList.innerHTML =
      '<li class="empty">Nothing detected yet — point your camera at something.</li>';
    return;
  }
  const best = new Map();
  for (const p of predictions) {
    const cur = best.get(p.class);
    if (!cur || p.score > cur.score) best.set(p.class, p);
  }
  const items = [...best.values()].sort((a, b) => b.score - a.score);
  els.detList.innerHTML = items
    .map(
      (p) => `<li><div class="det-item"><span>${escapeHtml(p.class)}</span>` +
        `<span class="conf">${Math.round(p.score * 100)}%</span></div>` +
        `<div class="bar"><i style="width:${Math.round(p.score * 100)}%"></i></div></li>`
    )
    .join("");
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

async function detectLoop(now) {
  requestAnimationFrame(detectLoop);
  const interval = 1000 / state.targetFps;
  if (!state.aiEnabled || !state.modelReady) return;
  if (els.video.readyState < 2) return;
  if (now - state.lastDetectAt < interval) return;
  state.lastDetectAt = now;

  try {
    const predictions = await state.model.detect(els.video);
    const kept = predictions.filter((p) => p.score >= state.minScore);
    state.lastPredictions = kept;
    drawPredictions(kept);
    renderList(kept);

    // Rolling FPS meter over the detection calls.
    state.detectCount++;
    const elapsed = (now - state.detectWindowStart) / 1000;
    if (elapsed >= 2) {
      els.fpsMeter.textContent = `${(state.detectCount / elapsed).toFixed(1)} fps`;
      state.detectCount = 0;
      state.detectWindowStart = now;
    }
  } catch (err) {
    console.error("detection failed:", err);
  }
}

function clearOverlay() {
  const ctx = els.canvas.getContext("2d");
  ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);
  state.lastPredictions = [];
  renderList([]);
  els.fpsMeter.textContent = "— fps";
}

// --- Controls -------------------------------------------------------------

els.aiToggle.addEventListener("click", () => {
  state.aiEnabled = !state.aiEnabled;
  els.aiToggle.classList.toggle("on", state.aiEnabled);
  els.aiToggle.setAttribute("aria-checked", String(state.aiEnabled));
  if (!state.aiEnabled) clearOverlay();
});

els.fps.addEventListener("input", () => {
  state.targetFps = Number(els.fps.value);
  els.fpsVal.textContent = `${state.targetFps} fps`;
});

els.confidence.addEventListener("input", () => {
  state.minScore = Number(els.confidence.value) / 100;
  els.confVal.textContent = `${els.confidence.value}%`;
});

els.flipCam.addEventListener("click", async () => {
  state.facingMode = state.facingMode === "environment" ? "user" : "environment";
  await startCamera();
});

els.snapshot.addEventListener("click", () => {
  const v = els.video;
  if (v.videoWidth === 0) return;
  const c = document.createElement("canvas");
  c.width = v.videoWidth;
  c.height = v.videoHeight;
  const ctx = c.getContext("2d");
  ctx.drawImage(v, 0, 0);
  // Burn the current boxes into the snapshot.
  ctx.drawImage(els.canvas, 0, 0, c.width, c.height);
  const a = document.createElement("a");
  a.href = c.toDataURL("image/png");
  a.download = `ai-object-detection-${Date.now()}.png`;
  a.click();
});

window.addEventListener("resize", resizeCanvas);
els.video.addEventListener("loadedmetadata", resizeCanvas);

// --- Boot -----------------------------------------------------------------

(async function init() {
  if (!("mediaDevices" in navigator) || !navigator.mediaDevices.getUserMedia) {
    showBanner("This browser does not support camera access (getUserMedia).");
    return;
  }
  await loadModel();
  await startCamera();
  requestAnimationFrame(detectLoop);
})();
