# Tech Stack

Organized by layer. Where there's a real fork in the road, both options are listed with a recommendation.

## 1. Mobile app

| Concern | iOS | Android | Notes |
|---|---|---|---|
| **Language/UI** | Swift + SwiftUI | Kotlin + Jetpack Compose | Recommended: **native per-platform** rather than cross-platform, specifically because of how deep the sensor/AR integration needs to go (see below). |
| **Camera capture** | `AVFoundation` | `CameraX` | Standard camera pipeline for plain-photo (Tier 3) capture. |
| **AR session / depth capture** | `ARKit` (`ARSession`, `ARFrame.sceneDepth`, `ARFrame.smoothedSceneDepth` on LiDAR devices) | `ARCore` (`Session`, `Frame.acquireDepthImage16Bits()`, Depth API) | This is the Tier 1 capture path. |
| **Guided object-capture flow (optional, high quality)** | `RealityKit` **Object Capture API** / `PhotogrammetrySession` | No direct Apple-equivalent; roll your own guided-capture UX on top of ARCore Depth API + CameraX burst capture | On iOS this can double as an *alternative* generation path for LiDAR devices — Apple's own photogrammetry pipeline can produce a scaled mesh directly from a guided multi-photo capture, which you could offer as a premium "high-accuracy" mode alongside the AI-generation pipeline. |
| **On-device marker detection (Tier 2 live guide)** | OpenCV iOS (via CocoaPods/SPM) or Vision framework's built-in barcode/rectangle detectors adapted for ArUco | OpenCV Android (AAR) | Purely for live UX feedback ("marker detected ✅") — final detection/measurement math still happens server-side on the uploaded image for consistency. |
| **On-device lightweight ML (optional)** | **Core ML** (convert PyTorch models via `coremltools`) | **TensorFlow Lite** or **ONNX Runtime Mobile**, or **MediaPipe** for turnkey segmentation/pose tasks | Only for real-time capture-quality feedback (e.g., "hold steady," rough live segmentation preview) — never the source of truth for the final mesh/measurements. |
| **3D/AR viewer** | `RealityKit` + **ARKit Quick Look** (native USDZ viewer, drag out to AR for free) | `Filament` (Google's real-time PBR renderer) or a `WebView` hosting `<model-viewer>` (Google's web component, GLB/GLTF + AR handoff via Scene Viewer) | `<model-viewer>` is the pragmatic cross-platform choice if you want one rendering codebase; native is smoother UX. |
| **Networking** | `URLSession` + `async/await`, or **Alamofire** | `Retrofit` + `OkHttp` | Standard REST/multipart upload + polling or WebSocket. |
| **Local persistence** | `SwiftData` / `Core Data` | `Room` | Cache job history, offline queue of pending uploads. |

**Cross-platform alternative:** Flutter (with `camera`, `arkit_plugin`/`ar_flutter_plugin`, and a `model_viewer_plus` widget) or React Native (with `react-native-vision-camera`, ARKit/ARCore native modules, and `@react-three/fiber` or a `model-viewer` WebView bridge) can work and will save real engineering time on non-AR screens (auth, history, settings, sharing). The tradeoff is that depth-sensor and photogrammetry-session access has to go through hand-written native modules either way, so you don't fully escape native code — evaluate based on team skill mix. If most of the app's value is the capture+AR experience, native-first is the safer bet.

## 2. Backend / API

| Concern | Choice | Alternative |
|---|---|---|
| **API framework** | **FastAPI** (Python) — natural fit since the ML stack is Python too, async support, auto OpenAPI docs | Node.js/Express or NestJS if the team is more JS-native (would need to shell out / gRPC to a separate Python ML service either way) |
| **Job queue** | **Celery + Redis** | **RQ** (simpler, fine for lower scale) or **Temporal** (if you want durable workflow orchestration with retries/observability built in — genuinely worth it once the pipeline has 4-5 chained steps) |
| **Database** | **PostgreSQL** | — |
| **Object storage** | **MinIO** (self-hosted, S3 API-compatible) for full self-hosting, or **AWS S3 / Cloudflare R2** if you're fine with a cloud storage dependency (this is *storage*, not a 3D-generation API, so it doesn't violate the "no external 3D API" constraint) | — |
| **Auth** | JWT-based sessions, or a managed provider (Auth0/Clerk/Supabase Auth) to avoid building it from scratch | — |
| **Inter-service communication** | REST between gateway and queue; internal gRPC or direct function calls between queue workers if colocated | — |

## 3. ML / 3D generation stack

| Concern | Choice |
|---|---|
| **Runtime** | **PyTorch** (all the candidate models — SAM 3D Objects, TRELLIS-2, Hunyuan3D 2.1, TripoSR, Metric3D v2, UniDepth v2, SAM2/SAM3, rembg — ship as PyTorch). |
| **Serving** | Start with plain FastAPI/Flask microservices wrapping each model (simplest to debug); graduate to **NVIDIA Triton Inference Server** if you need dynamic batching, multi-model management, and tighter GPU utilization at scale. |
| **Segmentation** | **rembg** (MVP) → **SAM2/SAM3** or **RMBG-2.0/BiRefNet** (quality upgrade). |
| **3D generation (primary)** | **Meta SAM 3D Objects** — https://github.com/facebookresearch/sam-3d-objects |
| **3D generation (quality alt.)** | **TRELLIS-2** (MIT) — https://github.com/microsoft (see repo linked from research notes) — or **Hunyuan3D 2.1** — https://github.com/tencent-hunyuan/hunyuan3d-2.1 |
| **3D generation (fast preview)** | **TripoSR** (MIT) |
| **Monocular metric depth (Tier 3 scale)** | **Metric3D v2** or **UniDepth v2** (open weight, actively maintained) |
| **Marker detection (Tier 2 scale)** | **OpenCV** `cv2.aruco` module |
| **Mesh math/cleanup** | **Open3D** (ICP/Umeyama alignment, point cloud fusion), **trimesh** (bounding box/volume/surface-area, format I/O), **PyMeshLab** (remeshing, hole filling, decimation) |
| **Optional heavy cleanup** | **Blender** headless (`bpy`) via CLI batch job, for advanced retopology/UV work if needed downstream (e.g., preparing for 3D printing) |
| **Model packaging** | Docker images per model family (dependency isolation — several of these repos have custom-compiled CUDA extensions that conflict across versions) |

## 4. Infrastructure / DevOps

| Concern | MVP | At scale |
|---|---|---|
| **Compute** | Single GPU box (e.g., one RTX 4090 or a cloud GPU instance like an A10G), `docker-compose` | Kubernetes with GPU node pools, separate pools per model family sized to their VRAM profile |
| **Autoscaling** | Manual / none | Queue-depth-based autoscaling (e.g., **KEDA** watching Redis/Celery queue length) |
| **Monitoring** | Basic logs + `nvidia-smi` dashboards | Prometheus + Grafana, GPU utilization/queue-latency dashboards, per-model error-rate tracking |
| **CI/CD** | GitHub Actions building mobile + backend + Docker images | Same, plus staged rollout for new model versions (quality regressions on generative models are easy to miss without a held-out benchmark set — see `docs/05-roadmap.md` Phase 0) |
| **Secrets/config** | `.env` files, platform secret store | Vault / cloud KMS |

## 5. Formats used throughout the pipeline

| Format | Where it's used |
|---|---|
| **GLB** | Primary interchange format for generated meshes — compact, single-file, includes PBR textures, broadly supported (three.js, `<model-viewer>`, Filament, most engines). |
| **USDZ** | iOS-native AR format — convert GLB → USDZ (Apple provides `usdz_converter` tooling) specifically for the ARKit Quick Look viewer/AR handoff. |
| **OBJ / STL** | Export option for users who want to 3D print the result or bring it into traditional CAD/DCC tools. |
| **JSON** | Measurement results (`{ "length_mm": …, "width_mm": …, "height_mm": …, "volume_cm3": …, "confidence_tier": … }`) delivered alongside the mesh. |
