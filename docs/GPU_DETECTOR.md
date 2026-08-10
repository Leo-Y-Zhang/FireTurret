# Learned detector & GPU acceleration

The fire detector is **injectable**: anything with
`detect(frame_bgr, camera_moving) -> list[FireBlob]` (and a no-op `reset()`) can
replace the classical HSV+flicker detector without touching the tracker, control,
ballistics, or rig layers.

```python
from fireturret.app import Pipeline
from fireturret.vision.onnx_detector import OnnxFireDetector

pipeline = Pipeline(cfg, detector=OnnxFireDetector("fire.onnx", cfg.detector))
```

or from the CLI:

```bash
pip install -e .[ml]                 # onnxruntime
python -m fireturret run --source 0 --detector onnx --model fire.onnx
```

## The ONNX detector

`OnnxFireDetector` runs a segmentation model that outputs a per-pixel
fire-probability map, thresholds it, and reuses the classical detector's
connected-components geometry (`blobs_from_mask`) to produce the same `FireBlob`
list the rest of the pipeline already consumes. So a trained model drops straight
in — the closed-loop visual servo, ballistics, and mission logic are unchanged.

Bring your own model. The class expects a single input (an NCHW RGB tensor,
normalised 0–1) and a single output (an HxW fire-probability map); adjust
`threshold` and `input_size` to your network.

## NVIDIA GPU / Jetson

Execution providers are selected in order of preference —
**TensorRT → CUDA → CPU** — so the *same code* runs accelerated on an NVIDIA GPU
and falls back to CPU elsewhere:

```python
select_providers(onnxruntime.get_available_providers())
# ["TensorrtExecutionProvider", "CUDAExecutionProvider", ...] when present
```

- **Desktop / server (NVIDIA GPU):** `pip install onnxruntime-gpu` (instead of the
  CPU `onnxruntime`); CUDA/TensorRT are picked automatically.
- **On-device (NVIDIA Jetson Orin / Nano):** a Jetson is the natural embedded
  target for this — it replaces the Raspberry Pi in
  [`BUILD_GUIDE.md`](BUILD_GUIDE.md) and runs the learned detector on-board with
  NVIDIA's Jetson `onnxruntime` build (JetPack + TensorRT). The turret firmware
  link is unchanged: the Jetson runs this software and talks to the Arduino/ESP32
  over the same serial protocol.

The classical detector needs no GPU and remains the zero-dependency default; the
learned path is opt-in for when accuracy on hard scenes matters more than
simplicity.
