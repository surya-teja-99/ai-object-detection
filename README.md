# ai-object-detection

Real-time AI object detection in your browser. Point your camera at something —
VisionScope draws bounding boxes and labels over the live feed, entirely
on-device. No video is uploaded anywhere; everything runs locally with
TensorFlow.js.

## Features

- **Live detection** — COCO-SSD model (80 everyday object classes) on your camera feed
- **AI toggle** — pause/resume detection without stopping the camera
- **Detection-rate slider** — 1–20 fps, with a live measured-fps meter
- **Confidence threshold** — filter weak detections (10–90%)
- **Detection panel** — live per-class list with confidence bars and counts
- **Flip camera** — switch front/back on mobile
- **Snapshot** — save the current frame with boxes burned in as PNG

## Run it

Any static file server works. From this folder:

```bash
python3 -m http.server 8000
```

then open **http://localhost:8000**. Camera access requires a secure context,
so use `localhost` or serve over HTTPS — `file://` will not work.

## How it works

```
camera (getUserMedia)
  → video element
  → COCO-SSD via TensorFlow.js (WebGL, on-device) @ throttled fps
  → canvas overlay: boxes + labels
  → sidebar: per-class counts + confidence bars
```

The detection loop runs on `requestAnimationFrame`, throttled to the
user-selected rate. The model (~27 MB) downloads once from a CDN and is then
cached by the browser.

## Browser support

Any modern browser with WebGL and camera access should work (Chrome, Edge,
Firefox, Safari 14.3+). Note: this was verified headless for clean page load
and script errors, but live camera detection wasn't tested on this machine —
if you hit an issue, please open one.

## Limitations

- Detects only the 80 [COCO classes](https://cocodataset.org/#explore) (person,
  car, dog, …) — anything outside that vocabulary is invisible to it.
- Accuracy and speed depend on your device; low-end phones may manage only a
  few fps.
- The model file downloads from a CDN on first load (~27 MB) — offline use
  needs it cached first.
- Front-camera preview is mirrored for usability; snapshots save unmirrored.

## Tech

- [TensorFlow.js](https://www.tensorflow.org/js) 4.17.0 (pinned)
- [COCO-SSD](https://github.com/tensorflow/tfjs-models/tree/master/coco-ssd) 2.2.3 (pinned)
- Plain HTML/CSS/JS — no build step, no framework
