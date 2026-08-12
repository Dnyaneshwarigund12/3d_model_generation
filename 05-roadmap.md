# Implementation Roadmap

Phased so that each phase produces something demoable/testable, and so measurement accuracy work (the genuinely hard part) doesn't get left until the end.

## Phase 0 — Research spike & model bake-off (1-2 weeks)

**Goal:** stop trusting blog-post benchmarks and validate model choice against your own target objects.

- Stand up minimal inference scripts (no API/queue yet) for: **SAM 3D Objects**, **TRELLIS-2**, **Hunyuan3D 2.1**, **TripoSR**.
- Build a small internal benchmark set: 20-30 photos across your actual target categories (e.g., furniture, packages, product photography, collectibles — whatever the app is really for), including some "hard" cases (reflective, thin structures, dark objects, cluttered backgrounds).
- Score each model on: mesh completeness/visual quality (subjective panel review is fine at this stage), inference time, VRAM footprint, failure rate (crashes/garbage output) on your hard cases.
- **Output of this phase:** a decision on primary + fallback generation model, and a documented go/no-go on whether SAM 3D Objects' "in the wild" strength actually holds for your specific object categories, or whether TRELLIS-2/Hunyuan3D's clean-background strength is the better match after all.
- Also spike the two metric-depth candidates (Metric3D v2, UniDepth v2) against a handful of tape-measure-verified real objects to get a feel for real-world Tier 3 error rates — this number should directly inform the UI copy in `docs/04-measurement-methodology.md` §5.

## Phase 1 — Backend inference MVP, no scale (2-3 weeks)

**Goal:** a single working endpoint: image in → mesh out, no measurement yet.

- FastAPI gateway with one synchronous-for-now endpoint (`POST /generate` → returns mesh URL); queue/async can come next phase once this proves the ML path end-to-end.
- Segmentation stage with `rembg` (simplest option, upgrade later).
- Primary 3D generation model from Phase 0, containerized, running on a single dev GPU box.
- Mesh export to GLB.
- **Demo:** curl an image at the endpoint, get back a GLB you can open in any glTF viewer.

## Phase 2 — Mobile capture MVP (2-3 weeks, can overlap Phase 1)

**Goal:** a working (rough) end-to-end app: take a photo → see a 3D model.

- Native iOS and/or Android app: plain camera capture (Tier 3 path only for now), upload to Phase 1's backend, render the returned GLB in a basic 3D viewer (`<model-viewer>` WebView is the fastest way to get this working on both platforms simultaneously; swap for native viewers later).
- No AR placement yet, no measurements yet — just "does the pipeline work as a product."

## Phase 3 — Async job architecture (1-2 weeks)

**Goal:** make the backend production-shaped before adding more model stages.

- Introduce Celery + Redis job queue; gateway becomes `POST /captures` (returns job id) + `GET /jobs/{id}` (poll).
- Add the TripoSR fast-preview path in parallel with the high-fidelity model, per the sequence diagram in `docs/02-architecture.md`.
- Mobile app updated to poll and progressively show preview → final mesh.

## Phase 4 — Tier 2 measurement: reference marker (2-3 weeks)

**Goal:** first *real* measurements, on any phone, no sensor dependency.

- Generate/print-ready ArUco marker asset; add on-device live marker detection for capture guidance (quality gate before upload).
- Backend `cv2.aruco` detection + homography scale-factor computation.
- Mesh-scaling step (§4 of measurement doc) using `trimesh`/`Open3D`: apply computed scale factor to the generated mesh.
- Bounding-box/volume extraction, `measurements.json` returned alongside the mesh.
- Mobile UI: measurement overlay + accuracy-tier badge.
- **Validate against ground truth:** literally tape-measure 15-20 real objects and compare — this is the acceptance criterion for this phase, not just "it runs."

## Phase 5 — Tier 1 measurement: depth sensor (2-4 weeks)

**Goal:** best-accuracy path on supported hardware.

- iOS: `ARSession` + `sceneDepth` capture flow (guided short motion capture), camera intrinsics extraction.
- Android: `ARCore` Depth API integration, same guided-capture UX.
- Backend: point-cloud back-projection, Umeyama/ICP alignment of the generated mesh to the metric point cloud (replacing the simpler Tier 2 uniform-scale approach where sensor data is available — better because it also gets orientation/pose alignment, not just scale).
- Re-run the tape-measure validation set from Phase 4 on Tier 1 devices — should show materially lower error.
- (Optional, evaluate cost/benefit) iOS-only: wire up Apple's native **Object Capture API** as an alternate "maximum accuracy" capture mode for LiDAR devices.

## Phase 6 — Tier 3 fallback: monocular metric depth (1-2 weeks)

**Goal:** graceful degradation on phones with neither a depth sensor nor a marker in frame.

- Integrate Metric3D v2 or UniDepth v2 as a backend worker.
- Wire into the same point-cloud back-projection + mesh-alignment code path already built for Tier 1 (same math, different depth source).
- Prominent "estimated" UI treatment per the measurement doc's UX section.
- Document observed real-world error rates from Phase 0's spike so support/marketing copy doesn't overpromise.

## Phase 7 — Production hardening (ongoing)

- GPU worker autoscaling (KEDA + queue depth), Kubernetes migration if traffic warrants it.
- Monitoring/alerting: queue latency, GPU utilization, per-model failure rate, measurement-tier distribution (are most users actually reaching Tier 1/2, or defaulting to rough Tier 3?).
- Data retention & privacy policy implementation (auto-delete raw captures, encryption at rest).
- Export polish: GLB/USDZ/OBJ/STL export, shareable links, a printable "measurement report."
- Cost monitoring per generation (GPU-seconds per job) — the two-speed preview/final pattern exists partly to control this; watch it as a real metric once there's traffic.
- Ongoing model refresh process: this field ships new open-weight checkpoints monthly — budget for periodic re-runs of the Phase 0 benchmark against new model releases (e.g., inevitable SAM 3D / TRELLIS / Hunyuan point updates) rather than treating the Phase 0 model choice as permanent.

## Key risks to watch

| Risk | Mitigation |
|---|---|
| Generative model output quality varies a lot by object category — what looks great in demos may fail on your actual users' photos | Phase 0's benchmark set should be built from *realistic* target photos, not cherry-picked product shots |
| GPU cost per generation adds up fast at scale (multiple models running per request) | Fast-preview/refine pattern, aggressive caching of loaded models in-process, consider making the high-fidelity pass opt-in/rate-limited per user tier |
| Tier 3 (no-sensor, no-marker) measurement error may be too high to be useful for some target use cases | Be honest about it in-product (tiered confidence UI); consider making Tier 3 "shape only, no measurement claim" for launch if validation numbers from Phase 0 are bad |
| Licensing drift — several of these models have licenses with revenue thresholds or attribution requirements (SPAR3D, Hunyuan3D) | Track license terms per model in a simple table (start from `docs/01-research-notes.md` §1) and re-check before any commercial launch, since terms can change between releases |
| Dependency/build fragility of research-grade repos (custom CUDA extensions, pinned exact package versions) | Docker image per model, pin everything, don't let workers share a Python environment |
