# Research Notes — State of Open-Source Photo → 3D (as of Aug 2026)

This is the raw research behind the plan: what open-source models exist right now for each stage of the pipeline, how they compare, their licenses/hardware needs, and where scale/measurement information can realistically come from. Every claim below is sourced; links are at the bottom of each section and consolidated in `docs/reference-links.md`... (see the **References** section at the end of this file).

> **Note on currency of this research:** This field moves fast — new checkpoints ship monthly. Treat specific version numbers (e.g. "TRELLIS-2", "Hunyuan3D 2.1") as "the SOTA class of model as of mid-2026"; re-check the linked repos before committing engineering time, since a newer point-release is likely by the time you build.

---

## 1. Single-image → 3D mesh generation models

This is the most active area. The dominant technical pattern in 2026 is **multi-view diffusion + feed-forward reconstruction**: a diffusion model imagines several consistent novel views of the object, then a feed-forward network fuses those views into a mesh. This is what TRELLIS-2, Hunyuan3D, and InstantMesh do. A second pattern, **direct feed-forward single-image reconstruction** (TripoSR, Stable Fast 3D), skips the multi-view step entirely and trades some fidelity for extreme speed.

### Comparison table

| Model | Org | License | Input | Output | Hardware | Speed | Notes |
|---|---|---|---|---|---|---|---|
| **SAM 3D Objects** | Meta (Nov 2025) | Open weights + code (check current license terms at release repo before commercial use) | Single "in the wild" photo, incl. occluded/cluttered scenes | Full 3D geometry + texture + scene layout (multiple objects) | GPU, server-side (not practical for on-device/browser inference) | Seconds | Trained on ~8M images with a human+model-in-the-loop data engine; reported ≥5:1 human-preference win rate vs. prior single-image methods; explicitly designed for real, messy photos rather than clean product shots — the best fit for "a normal person's phone photo." |
| **TRELLIS-2** | Microsoft Research (Dec 2025) | **MIT** | Single image | Textured mesh + 3D Gaussians (GLB export) | Linux, **≥24GB VRAM**, CUDA 12.4 (community-optimized builds run on ~8GB in low-VRAM mode at reduced resolution) | Seconds | 4B-parameter sparse structured-latent model; widely cited as the best fully open-source model for raw quality in mid-2026; no text-to-3D, rigging, or retexturing built in. |
| **Hunyuan3D 2.1** | Tencent | Permissive, **commercial use allowed with attribution** ("Created with Hunyuan 3D-2.1") | Single or multi-view image | Textured mesh with production-ready PBR materials | 10GB VRAM (shape only) / 21GB (texture only) / 29GB (both); community "GP" (GPU-Poor) fork gets shape generation down to ~6GB | Tens of seconds | Strong PBR texture quality; two-stage pipeline (shape, then texture) means you can split it across separate workers/queues. |
| **TripoSR** | Stability AI + Tripo AI | **MIT** | Single image | Textured mesh | Low VRAM, works on a single consumer GPU | **~0.5s** on a strong GPU | Built on the Large Reconstruction Model (LRM) architecture, trained on Objaverse; best used as the **instant low-fidelity preview** while a higher-quality job runs in the background. |
| **Stable Fast 3D (SF3D) / SPAR3D** | Stability AI | Stability **Community License** — free for commercial use until the org passes $1M annual revenue, then requires an enterprise license | Single image | Textured, UV-unwrapped GLB; SPAR3D supports point-level editing | Low VRAM | <1s | Good speed/quality tradeoff; license has a revenue ceiling to be aware of if the app scales. |
| **InstantMesh** | Open source (multi-view diffusion + LRM-style reconstruction) | Open (check repo) | Single image | Mesh | Moderate VRAM | Seconds | Reasonable middle ground between TripoSR speed and TRELLIS/Hunyuan quality. |

### Recommendation

- **Primary model: Meta SAM 3D Objects.** It's specifically built and evaluated for exactly this use case — an uncontrolled, single photo taken by a normal person, not a staged product shot on a white background. That matches "picture from my mobile phone" more closely than the other models, which were mostly benchmarked on Objaverse-style clean renders/product photos.
- **Secondary/quality-comparison model: TRELLIS-2 or Hunyuan3D 2.1**, particularly for objects photographed cleanly (e.g., against a plain background) where their PBR texture quality edges out SAM 3D's. Worth A/B testing both against your own target use cases (furniture? packages? collectibles? people?) since "best" depends heavily on the object category.
- **Instant-preview model: TripoSR**, purely for perceived latency — show *something* in under a second, swap it for the high-fidelity result when the queued job finishes.
- If capturing **people** specifically becomes a use case, note that Meta also released **SAM 3D Body**, a separate model specialized for full-body human mesh recovery (with hand/foot detail) from a single image, using the open "Momentum Human Rig" format — worth a dedicated evaluation if body scanning is in scope.

**References:**
- SAM 3D Objects paper/code: https://arxiv.org/pdf/2511.16624 , https://github.com/facebookresearch/sam-3d-objects , https://ai.meta.com/blog/sam-3d/
- TRELLIS-2: https://trellis2.app/blog/comfyui-3d-model-generator-microsoft , https://www.meshy.ai/compare/meshy-vs-trellis-2
- Hunyuan3D 2.1: https://github.com/tencent-hunyuan/hunyuan3d-2.1 , GPU-poor fork: https://github.com/deepbeepmeep/Hunyuan3D-2GP
- TripoSR / Stable Fast 3D / SPAR3D / general 2026 landscape: https://www.pixazo.ai/blog/best-open-source-3d-model-generation-apis , https://www.triposrai.com/posts/open-source-3d-reconstruction-showdown/ , https://app.cinevva.com/guides/ai-3d-model-generators

---

## 2. Foreground segmentation / background removal

All of the generation models above work better on a subject cleanly separated from its background — and segmentation is also where you strip out any reference marker (see §4) before it confuses the 3D generator.

| Model | Notes |
|---|---|
| **SAM2 / SAM3 (Meta)** | State of the art, promptable (points/boxes/text). SAM3 adds text-phrase prompting ("segment the chair"), which is convenient for a mobile flow where the user taps or the app auto-detects the main subject. Heavier than dedicated matting nets. |
| **RMBG-2.0 (BRIA AI)** | Purpose-built background removal, good edge/hair detail, widely used in production cutout pipelines; check BRIA's commercial license terms. |
| **BiRefNet** | Strong dichotomous (salient-object) segmentation, handles fine detail, transparent/thin structures. |
| **rembg (U²-Net based)** | The simplest, most permissive, easiest to self-host option; lower fidelity on complex edges than the above but perfectly adequate for many objects and trivial to deploy. |

**Recommendation:** Start with **rembg** for MVP simplicity, upgrade the segmentation stage to **SAM2/SAM3 or RMBG-2.0/BiRefNet** once accuracy on real user photos (busy backgrounds, thin structures, reflective objects) demands it. This stage is swappable behind an internal interface — don't over-invest here early.

**References:** https://github.com/1038lab/ComfyUI-RMBG , https://www.bestaiweb.ai/topics/ai-background-removal/ , https://cran.r-project.org/web/packages/rembg/rembg.pdf

---

## 3. Monocular metric depth estimation (scale without a sensor or marker)

This is the "best effort" tier of measurement — used only when the phone has no depth sensor *and* the user didn't include a reference object. These models predict **absolute distance in meters per pixel** from a single RGB image (as opposed to plain "relative depth" models like the original MiDaS/Depth Anything, which only get relative ordering right, not real-world scale).

| Model | Approach | Notes |
|---|---|---|
| **Metric3D v2** | Discriminative, camera-aware canonicalization | Widely used zero-shot metric depth + surface normal foundation model; open code. |
| **UniDepth / UniDepth v2** | Self-promptable camera module, predicts metric 3D point clouds without needing camera intrinsics as input | Strong generalization across camera types, "universal" claim backed by benchmarks. |
| **Depth Pro (Apple)** | Estimates field-of-view from image features itself, produces sharp high-resolution metric depth in under a second | Good speed/quality; check Apple's research license terms for commercial use. |
| **MoGe / MoGe-2** | Monocular geometry estimation with optimal training supervision; MoGe-2 adds metric scale + sharper detail | CVPR'25 oral; actively developed. |
| **ZoeDepth, PatchFusion** | Earlier but still-used metric depth baselines | PatchFusion adds tile-based high-resolution inference. |

**Important caveat to design around:** all of these are *statistical priors* trained mostly on everyday scenes (indoor rooms, street scenes, common furniture/vehicles). They can generalize badly to atypical objects or unusual framings, and published zero-shot error rates for absolute scale are meaningfully non-trivial (commonly cited in the 5–20%+ range depending on domain and model, worse out-of-distribution). Treat this tier's output as a **labeled estimate**, not a certified measurement — this is reflected directly in the UX design in `docs/04-measurement-methodology.md`.

**References:** https://github.com/choyingw/Awesome-Monocular-Depth , https://arxiv.org/pdf/2501.11841 (Survey on Monocular Metric Depth Estimation) , https://arxiv.org/pdf/2509.14839 , https://arxiv.org/pdf/2512.12425

---

## 4. Getting real metric scale: sensors and markers

### 4a. Phone depth sensors (best accuracy, Tier 1)

- **iOS LiDAR (iPhone/iPad Pro):** Apple's **Object Capture API** (RealityKit/Photogrammetry) takes many photos of an object from multiple angles — optionally with LiDAR depth captured alongside — and produces a scaled, textured 3D mesh directly. Apple explicitly recommends the LiDAR-equipped capture flow "for best results and accurate scale of the object." Apple's related **RoomPlan** API (also LiDAR-powered) does the analogous thing for whole rooms, outputting parametric models with real dimensions, exportable as USD/USDZ.
- **Android ARCore Depth API:** Produces a per-pixel metric depth map even **without** a dedicated Time-of-Flight sensor, using a "depth-from-motion" algorithm that compares multiple frames as the user moves the phone slightly; it automatically fuses in ToF sensor data on the (growing) set of Android phones that have one. Depth is most accurate roughly 0.5–5 meters from the camera — a good match for "hold your phone up to an object" use cases.

**Product implication:** on supported hardware, the "single picture" the user thinks they're taking is best implemented as a **very short guided multi-frame capture** (a 1–3 second phone wiggle, exactly like Apple's Object Capture flow or ARCore's motion-based depth), not a literal single shutter press — this is what gets you real depth data instead of a guess.

**References:** https://developer.apple.com/videos/play/wwdc2022/10127/ , https://machinelearning.apple.com/research/roomplan , https://developer.vuforia.com/library/vuforia-engine/images-and-objects/model-targets/using-3d-scans/model-targets-apples-object-capture/ , https://developers.google.com/ar/develop/depth , https://developers.google.com/ar/develop/java/depth/quickstart

### 4b. Reference-object / marker-based scale (Tier 2, works on any phone, no sensor needed)

Long-established photogrammetry/computer-vision technique: place an object of **known real-world size** in the same photo (an ArUco fiducial marker printed on paper, a credit card — ISO/IEC 7810 ID-1 format is a standardized 85.60 × 53.98 mm — or a sheet of A4/Letter paper), detect it automatically with OpenCV's `cv2.aruco` module, and use its known size vs. its measured size in pixels to compute a pixel-to-millimeter conversion factor for the rest of the image.

- ArUco markers are purpose-built for this: a black-bordered square with an internal binary ID, designed for fast, robust, sub-pixel-accurate corner detection even under moderate lighting/angle variation.
- A plain credit card also works as a "reference standard," but has known limitations: proportions vary slightly across real cards, it doesn't scale well when the target object is much bigger or smaller than a card (error grows with scale mismatch and if the reference and object aren't in the same image plane), and the user needs to have one on hand.
- Best practice from the literature: prefer a **printed ArUco marker card** the app can generate/display, over relying on a credit card, for consistency and accuracy — while still supporting the credit-card fallback since it's more likely something a user already has nearby.

**References:** https://arshren.medium.com/measure-object-size-using-opencv-and-aruco-marker-fa8b2e3b0572 , https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/9696897 , https://www.researchgate.net/figure/Visualisation-function-left-Aruco-marker-detected-position-in-red-target-of_fig3_339838187

---

## 5. Mesh processing / measurement extraction libraries

Once we have (a) a generated mesh and (b) a real-world scale factor or metric point cloud, we need to combine them and pull out numbers.

| Library | Role |
|---|---|
| **Open3D** | Point cloud & mesh processing: ICP registration (aligning the generated mesh to a metric point cloud), Poisson surface reconstruction (if fusing raw LiDAR/ARCore depth points), mesh simplification, normal estimation. |
| **trimesh** | Lightweight mesh I/O and geometry queries: watertightness checks, axis-aligned/oriented bounding boxes, volume, surface area, format conversion (GLB/OBJ/STL/PLY). This is the primary library for the final "compute L×W×H / volume" step. |
| **PyMeshLab** | Python bindings to MeshLab's filters: hole filling, remeshing/retopology, decimation — useful for cleaning up generative-model output before it's presentable/printable. |
| **Blender (`bpy`, headless/CLI)** | Optional heavier-duty cleanup, UV work, or format conversions beyond what the above libraries cover; usable in a server-side batch job. |

**Umeyama alignment / similarity transform** (rotation + translation + **uniform scale**) is the standard algorithm for solving "what scale factor and pose turns my arbitrary-unit generated mesh into the metric point cloud I measured" — implemented in Open3D and easy to hand-roll (closed-form least-squares solution). Full detail in `docs/04-measurement-methodology.md`.

---

## 6. What we are deliberately *not* using

Per the project constraint, the plan excludes any hosted "image → 3D" product/API, including but not limited to: Meshy, Tripo (hosted product, as opposed to the open TripoSR *model* which we do use), Rodin AI / Hyper3D, Luma AI, CSM, Kaedim, Meshcapade, 3D AI Studio, Sloyd, and similar SaaS. All generation happens on model weights we download and run ourselves.

---

## References (consolidated)

- 3D AI Studio 2026 image-to-3D comparison: https://www.3daistudio.com/3d-generator-ai-comparison-alternatives-guide/best-image-to-3d-tools-2026
- Cinevva 2026 AI 3D model generator guide: https://app.cinevva.com/guides/ai-3d-model-generators
- TripoSR AI open-source reconstruction comparison: https://www.triposrai.com/posts/open-source-3d-reconstruction-showdown/
- Real3D (LRM scaling) paper: https://arxiv.org/pdf/2406.08479
- Pixazo open-source 3D generation APIs guide: https://www.pixazo.ai/blog/best-open-source-3d-model-generation-apis
- VirtualCoders open-source 3D AI models roundup: https://www.virtualcoders.net/blog/5-open-source-ai-models-that-generate-professional-3d-images-you-can-use-today/
- Awesome Monocular Depth list: https://github.com/choyingw/Awesome-Monocular-Depth
- Survey on Monocular Metric Depth Estimation: https://arxiv.org/pdf/2501.11841
- MapAnything (metric depth eval) paper: https://arxiv.org/pdf/2509.14839
- Ultralytics monocular depth docs: https://docs.ultralytics.com/tasks/depth
- UrbanVGGT (depth models survey section): https://arxiv.org/pdf/2603.22531
- Boosting Monocular Metric Depth via Bokeh Rendering: https://arxiv.org/pdf/2512.12425
- Apple RoomPlan (WWDC22): https://developer.apple.com/videos/play/wwdc2022/10127/
- Apple RoomPlan ML research: https://machinelearning.apple.com/research/roomplan
- Apple Object Capture via Vuforia docs: https://developer.vuforia.com/library/vuforia-engine/images-and-objects/model-targets/using-3d-scans/model-targets-apples-object-capture/
- it-jim iOS 3D reconstruction guide: https://www.it-jim.com/blog/3d-reconstruction-on-ios/
- Cultural-heritage Object Capture assessment paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11637407/
- Google ARCore Depth API overview: https://developers.google.com/ar/develop/depth
- ARCore Depth API quickstart: https://developers.google.com/ar/develop/java/depth/quickstart
- ARCore Raw Depth codelab: https://codelabs.developers.google.com/codelabs/arcore-rawdepthapi
- ArUco object-size measurement walkthrough: https://arshren.medium.com/measure-object-size-using-opencv-and-aruco-marker-fa8b2e3b0572
- Reference-standard measurement patent (credit-card limitations): https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/9696897
- Meshy vs TRELLIS-2 comparison (VRAM/license specifics): https://www.meshy.ai/compare/meshy-vs-trellis-2
- Hunyuan3D 2.1 GitHub: https://github.com/tencent-hunyuan/hunyuan3d-2.1
- Hunyuan3D-2GP (low-VRAM fork): https://github.com/deepbeepmeep/Hunyuan3D-2GP
- TRELLIS 2 ComfyUI guide (VRAM tiers): https://trellis2.app/blog/trellis-2-comfyui
- Meta SAM 3D announcement: https://ai.meta.com/blog/sam-3d/
- SAM 3D paper: https://arxiv.org/pdf/2511.16624
- SAM 3D GitHub: https://github.com/facebookresearch/sam-3d-objects
- Meta SAM 3D press release: https://about.fb.com/news/2025/11/new-sam-models-detect-objects-create-3d-reconstructions/
- ComfyUI-RMBG (segmentation model roundup): https://github.com/1038lab/ComfyUI-RMBG
- AI background removal (SAM2 vs RMBG) overview: https://www.bestaiweb.ai/topics/ai-background-removal/
- rembg package docs: https://cran.r-project.org/web/packages/rembg/rembg.pdf
