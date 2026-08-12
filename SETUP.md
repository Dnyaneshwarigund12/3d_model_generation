# Running the MVP

The core flow from the planning docs, cut down to one thing: **upload a photo, get a
3D model and its real-world dimensions.**

No mobile app, no job queue, no database, no object store, no Kubernetes. One Gradio
page, one synchronous function, files on disk. The planning documents in this folder
describe where it goes next; this file describes what exists now.

---

## What runs where

The models need a CUDA GPU, so the app is designed to run inside a **free Colab T4
notebook** and hand you a public URL.

The T4 matters for one decision: it is a Turing card with no bf16 and no
flash-attention, which is exactly what TRELLIS-2 expects. So the generation backends
here are **TripoSR** (fast, low quality, ~4 GB) and **Hunyuan3D 2.1** (slower, much
better, ~10 GB for shape), with TRELLIS-2 left for later.

| Backend | VRAM | Speed | Use it for |
|---|---|---|---|
| `silhouette` | none, CPU | instant | Testing the pipeline with no GPU. Extrudes the outline; **not a reconstruction.** |
| `triposr` | ~4 GB | ~2 s | Getting the flow working end to end. |
| `hunyuan3d` | ~10 GB shape, ~21 GB texture | 35-50 s | Actual quality. Keep texture off on a T4. |

## Quick start on Colab

Open `notebooks/colab_photo_to_3d.ipynb` in Colab, set **Runtime > Change runtime type >
T4 GPU**, and run the cells top to bottom. Steps 6 (Hunyuan3D) and 7 (UniDepth) are
optional on a first pass.

The notebook gets the code either by cloning your GitHub repo or by copying the folder
from Google Drive - set `CODE_SOURCE` in step 3. Step 2 caches model weights on Drive so
you download the multi-gigabyte checkpoints once rather than once per session.

The last cell prints a `gradio.live` URL that works for 72 hours, and only while the cell
is running. Open it on your phone to upload photos straight from the camera.

## Local install (development and tests only)

Everything except the GPU backends runs locally, which is enough to develop and test the
scale and measurement maths. Use a virtual environment - Gradio 6 wants
`huggingface-hub>=1.16`, which collides with older `transformers` installs:

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt

python -m pytest tests -q
python -m app.ui --generator silhouette
```

59 tests, no GPU and no model weights required. They pass on both OpenCV 4.6 with
numpy 1.26 and OpenCV 5.0 with numpy 2.4; ArUco's API changed at OpenCV 4.7 and
`app/scale.py` handles both spellings.

---

## Getting measurements that mean something

A single photo is scale-ambiguous. A toy car close up and a real car far away produce
identical pixels, so no model can recover true size from pixel content alone - it can
only guess from what similar-looking objects usually measure. Scale has to come from
outside the image.

Four sources, best first:

| Source | Error | What it needs |
|---|---|---|
| Printed ArUco marker | ~4% | One sheet of paper |
| Bank card | ~10% | A card, lying in the object's plane |
| A dimension you know | your own accuracy | You typed it in |
| Monocular depth model | 20%+, an estimate | Nothing |

### The marker workflow

```bash
python tools/make_marker.py --mm 50 --pdf --out assets/markers/marker_50mm.png
```

1. Print it at **100% scale**. Turn off "fit to page" - any rescaling silently changes
   the one constant the whole measurement chain divides by.
2. Measure the printed black square with a ruler. If it is not 50 mm, enter what it
   actually is.
3. Lay it flat beside the object, roughly in the same plane, all four corners visible
   and in focus.

The marker is detected and then **deleted from the subject mask** before generation, so
the 3D model is of your object and not of your object fused to a marker.

The dominant error in this path is depth mismatch: the millimetres-per-pixel ratio is
exact only in the marker's own plane and degrades as the object sits nearer or further
than it. Keeping the two roughly coplanar is what keeps the number honest.

---

## What you get back

Each run writes `outputs/<run_id>/`:

| File | Contents |
|---|---|
| `model.glb` | The scaled mesh. **In metres**, because that is glTF's unit, while every reported number is millimetres. |
| `measurements.json` | The dimensions, in the contract from `04-measurement-methodology.md` section 3. |
| `run.json` | Everything else: which model, which scale source, per-stage timings, warnings, settings. |
| `input.png`, `cutout.png`, `subject.png`, `reference_overlay.png` | What each stage saw. Check the overlay first when a number looks wrong. |

```json
{
  "length_mm": 202.4,
  "width_mm": 101.2,
  "height_mm": 41.1,
  "volume_cm3": 842.5,
  "watertight": true,
  "volume_basis": "mesh",
  "measurement_tier": "reference_marker",
  "estimated_error_pct": 5.0
}
```

Length, width and height are the oriented bounding box's extents sorted largest first.
They are not labelled by direction on purpose: the mesh has no notion of which way was
up, so calling the tallest extent "height" would be a guess.

Two things the output is deliberately explicit about:

- **`detail.inferred_depth_mm`** - depth away from the camera is invented by the 3D
  model. Nothing in the photo constrains it, so it is reported separately from the two
  dimensions the reference object actually determines.
- **`volume_basis`** - `mesh` when the mesh is watertight, `convex_hull` when it is not,
  which over-estimates anything concave.

---

## Accuracy: what is checked and what is not

**Checked**, by the test suite (54 tests, all on CPU):

- Marker detection recovers a known millimetres-per-pixel ratio to within 1% on
  synthetic images, and the generated printable marker round-trips to within 0.1% of its
  requested size.
- A marker tilted away from the camera by up to 55% foreshortening still measures within
  2%. This one needed work: averaging the marker's four edges, the obvious approach, gave
  up to **20% error** on a tilted marker, because perspective can only ever shorten an
  edge, so averaging pulls the estimate toward the foreshortened ones. Using the least
  foreshortened edge instead removes almost all of it. See `mm_per_px_from_quad` in
  `app/scale.py`.
- The mesh scaling and measurement maths are exact on known geometry, and invariant to
  the mesh's own units and to any rotation of it - generators differ in canonical pose,
  and the result must not.
- The whole chain, marker to reported millimetres, is within 3% on a synthetic scene.

Blur turns out to be a mild problem (1.4% bias even when heavily blurred, though it can
stop detection outright), and marker size only matters below about 30 px across.

**Not checked, and it needs you:** how the system behaves on real photographs of real
objects. The `estimated_error_pct` values the app reports start as the published ranges
from the research notes, not measurements of this pipeline.

To replace them with real numbers:

```bash
# fill in a CSV like tools/validation_manifest.example.csv first
python tools/validate.py --manifest tools/my_objects.csv --generator hunyuan3d
```

Tape-measure 10-15 objects covering the sizes and shapes you care about, including
awkward ones (dark, shiny, thin, cluttered background). The script prints measured error
per tier. **If measured error is worse than claimed, raise the base values in
`app/scale.py`** rather than leaving the badge optimistic.

---

## Code map

```
app/
  pipeline.py        the whole flow, stage by stage
  segment.py         rembg cutout, subject bbox, marker removal
  scale.py           marker / card / manual / estimate -> millimetres per pixel
  measure.py         mesh units -> millimetres, then dimensions
  depth.py           Tier 3 monocular metric depth
  ui.py              the Gradio page
  generators/        silhouette (CPU), triposr, hunyuan3d
tools/
  make_marker.py     printable marker at an exact size
  validate.py        measured error against tape-measured truth
notebooks/
  colab_photo_to_3d.ipynb
```

Adding a generation backend means one file in `app/generators/` and one line in
`_BACKENDS` in `generators/base.py`. Nothing downstream knows which model produced the
mesh: scale is solved independently and applied afterwards.

## Known limitations

- **Depth is a guess.** Only the two dimensions facing the camera are pinned by the
  reference object. A photo cannot show you the back of an object.
- **The marker must be roughly coplanar with the subject**, otherwise its
  millimetres-per-pixel ratio does not apply where the object is.
- **Card detection has no identity check.** It finds a rectangle of about the right
  proportions, so it can lock onto a book or a phone. Check the overlay.
- **Lens distortion is not corrected**, so a reference object near the frame edge of a
  wide-angle phone photo measures slightly wrong.
- **One object per photo.** The largest connected region wins.
- **Tier 1 from the measurement doc is absent**, because an uploaded photo carries no
  depth map. That path needs the phone-side capture app the planning docs describe.
