# Measurement Methodology — How We Actually Get Real-World Dimensions

This is the part of the system that determines whether "measurements" in the product means something trustworthy or just a plausible-looking number. Read this before designing any UI that shows a number with a unit on it.

## 1. Why this is hard: the fundamental scale ambiguity

A single 2D image is a projection of a 3D scene through a pinhole camera model. For any object in that image, there is a family of (size, distance) pairs that produce the **exact same pixels**: a 10cm toy held 30cm from the lens looks identical to a 100cm object 3m from the lens, if the focal length and framing are scaled to match. This is not a limitation of any particular AI model — it's a property of projective geometry. **No model can recover true metric scale from pixel content and a generic prior alone; it can only guess based on what similar-looking objects usually measure.**

This has a direct consequence for architecture: **scale must come from outside the pixels** — from a depth sensor, from a known-size reference object in the frame, or (as a fallback, with a wide error bar) from a model that has learned typical real-world sizes for common object categories. All three are implemented, and the system is explicit with the user about which one produced their result.

## 2. The three tiers

### Tier 1 — Depth sensor (best: typically low single-digit % error)

**When:** iPhone/iPad Pro with LiDAR, or Android device with ARCore Depth API support (works even without a dedicated ToF sensor, via depth-from-motion).

**How it works:**
1. During the guided capture (a brief phone movement, 1-3 seconds), `ARFrame.sceneDepth` (iOS) or `Frame.acquireDepthImage16Bits()` (Android) gives a per-pixel depth map `Z(u,v)` in meters, aligned to the RGB frame, along with the camera's **intrinsic matrix** `K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]` for that frame.
2. Back-project each foreground pixel (from the segmentation mask) into a **metric 3D point cloud**:
   ```
   X = (u - cx) * Z / fx
   Y = (v - cy) * Z / fy
   Z = Z(u,v)
   ```
3. This point cloud is *already in real-world units (meters)*. Its axis-aligned or oriented bounding box directly gives length/width/height; `trimesh`/`Open3D` convex hull volume gives an approximate volume.
4. **This metric point cloud is also the ground truth we scale the generated mesh to** (see §4) — because the AI-generated mesh has much better completeness (it hallucinates plausible geometry for occluded/back-facing surfaces that the camera never saw) but arbitrary scale, while the depth-sensor point cloud has correct scale but is incomplete (only covers the visible surface, and can be noisy/sparse for shiny, dark, transparent, or thin objects — a well-known LiDAR/ToF limitation flagged in Apple's own documentation).

**Practical note:** Apple's own **Object Capture API** can be used as an *alternative, higher-friction* path on LiDAR devices — feed it a real guided multi-photo capture and it produces an already-scaled mesh directly via traditional photogrammetry, without our own generative pipeline at all. This is worth offering as an optional "maximum accuracy" mode, at the cost of a slower, more deliberate capture flow (dozens of photos around the object vs. one quick AI-assisted shot).

### Tier 2 — Reference marker / known object (good: roughly 3-8% error, workable on any phone)

**When:** No depth sensor, but the user places a printed ArUco marker (or, as a lower-accuracy fallback, a standard credit card / A4 sheet) in the same photo, roughly coplanar with the object being measured.

**How it works:**
1. Detect the marker with OpenCV's `cv2.aruco` module — it returns the marker's four corner points in pixel coordinates, robust to moderate viewing angle and lighting variation.
2. Because we know the marker's **real-world side length** `S` (we control the printed size, or use the standardized credit-card dimension of 85.60mm × 53.98mm as a known constant), compute a homography `H` that maps the marker's canonical square (`[0,0], [S,0], [S,S], [0,S]`) to its detected pixel corners.
3. For any two points on the object that lie **approximately in the same plane as the marker**, unwarp their pixel coordinates through `H⁻¹` to get real-world distances directly. For points off that plane, this degrades (foreshortening error grows with the point's depth deviation from the marker's plane) — good enough for "how wide is this box" style measurements, not for arbitrary 3D shapes viewed at an angle.
4. For the 3D-mesh case specifically: use the marker's known size to compute a single **pixel-to-millimeter ratio** at the object's approximate depth, then apply that as a uniform scale factor to the generated mesh (see §4) — simpler and more robust than trying to unwarp full 3D geometry through a 2D homography.

**Error sources to design against:** marker not coplanar with the object, marker too small/far/blurry in frame (fewer pixels = more quantization error), lens distortion uncorrected at the image edges. Mitigate with: live on-device marker-quality feedback before upload (§3 of the architecture doc), and a minimum marker-size-in-frame requirement.

### Tier 3 — Monocular metric depth model (rough estimate: highly variable error, can exceed 20% out-of-distribution)

**When:** No depth sensor, no reference object — just a plain photo.

**How it works:**
1. Run a **metric** (not just relative) monocular depth model — **Metric3D v2** or **UniDepth v2** are the recommended open-weight choices — on the segmented image. Unlike relative-depth models (plain MiDaS/Depth Anything), these are specifically trained/calibrated to output depth in meters.
2. Back-project the predicted depth map into a point cloud exactly as in Tier 1 (§2, step 2), using either the phone's reported camera intrinsics (from EXIF or the camera API) or intrinsics estimated by the depth model itself (UniDepth's self-promptable camera module does this when true intrinsics aren't available).
3. Extract bounding-box dimensions from this point cloud, same as Tier 1.

**This is fundamentally a learned guess.** These models generalize well for object categories well-represented in their training data (furniture, people, vehicles, everyday indoor/outdoor scenes) and can be substantially wrong for unusual objects, extreme close-ups, or ambiguous framing. **Product requirement: always surface the accuracy tier to the user** (e.g., a small "Estimated ± " badge vs. a "Measured" badge for Tier 1/2), and consider actively prompting "add a reference card for a precise measurement" when the app detects it's about to fall back to Tier 3.

## 3. Confidence/tier metadata contract

Every measurement result the backend returns should include which tier produced it, so the client can render an honest confidence indicator rather than a bare number:

```json
{
  "length_mm": 412.3,
  "width_mm": 268.7,
  "height_mm": 190.1,
  "volume_cm3": 9840.2,
  "measurement_tier": "sensor_depth | reference_marker | monocular_estimate",
  "estimated_error_pct": 3.5
}
```

## 4. Bridging step: applying real-world scale to the AI-generated mesh

This is the piece that's easy to miss: **the generative 3D models (SAM 3D Objects, TRELLIS-2, Hunyuan3D 2.1, TripoSR) all output a mesh in an arbitrary canonical scale** — they were trained to reproduce *shape*, not real-world units, and typically normalize their output to fit inside a unit cube or similar. So the pipeline never asks the generator for a "metric mesh"; instead:

1. Generate the (unscaled) mesh from the segmented image.
2. Independently compute a real-world scale reference using whichever tier applies (§2): either a metric point cloud (Tier 1/3) or a scale factor (Tier 2).
3. **Align and rescale the generated mesh to match**, using one of:
   - **Umeyama algorithm** (closed-form least-squares similarity transform: rotation + translation + *uniform scale*) — the standard method when you have (even a partial, noisy) corresponding metric point cloud, e.g. from Tier 1 depth sensing. Fit the generated mesh's *visible-surface* vertices (the side facing the camera, which should roughly correspond to what the depth sensor saw) against the metric point cloud, solve for scale, apply that scale to the *entire* mesh (including the hallucinated back side).
   - **ICP (Iterative Closest Point)**, available in Open3D, to refine the alignment after an initial Umeyama estimate — useful when the correspondence between mesh vertices and depth points isn't already known/ordered.
   - **Direct scale-factor multiplication**, for the Tier 2 marker case — since the marker gives a single pixel-to-mm ratio rather than a full point cloud, the simplest robust approach is: measure a clear, unambiguous dimension of the object in the original 2D image (e.g., its bounding-box width in pixels), convert to mm using the marker ratio, then compute the scale factor needed to make the generated mesh's corresponding rendered-silhouette width match that mm value, and apply it uniformly.
4. Only *after* this alignment step do we run bounding-box/volume extraction (`trimesh`) on the now-correctly-scaled mesh for the final reported numbers.

## 5. UX implications (brief — full flows belong in a design doc, not here)

- Always show the **accuracy tier** next to any measurement, in plain language ("Measured with depth sensor," "Measured with reference card," "Estimated from photo — add a reference card for a precise measurement").
- Nudge, don't force: default to the best tier the hardware supports, but never block the Tier 3 fallback — a rough estimate is still useful for e.g. "will this roughly fit in my car," even if it's not good enough for e.g. shipping/manufacturing decisions.
- Consider a one-time onboarding explainer ("Why does adding a card make this more accurate?") — this is a legitimately non-obvious limitation of phone-only 3D scanning and users will ask.
