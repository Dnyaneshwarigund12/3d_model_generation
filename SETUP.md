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
T4 GPU**, and run the steps in order. There are only three that matter:

```
step 3   !python tools/colab_setup.py     install everything, once
step 4   Runtime > Restart session        then re-run step 2
step 5   !python tools/doctor.py          verify before spending GPU time
```

**Why it is shaped like this.** Colab ships numpy, Pillow, scipy, OpenCV and torch already
compiled against each other. Each separate `pip install` is a separate resolution, free to
move one of them to satisfy a new package's pin, and a partly applied downgrade leaves a
package's Python files and its compiled extension on different versions. The error that
follows names the wrong culprit — a broken numpy reports itself as "rembg is not installed" —
so fixing them one at a time never converges.

`tools/colab_setup.py` writes `constraints-colab.txt` from the versions the machine already
has, then installs the pipeline and both generation backends in a **single pip transaction**
bounded by that file. Anything that genuinely cannot live with the baseline fails at install
time, by name. Then restart once, because a compiled extension cannot be reloaded in place.

`tools/doctor.py` runs as a script, never imported into the kernel: a fresh process is the
only place a version number reflects what is on disk. It checks each library, then runs the
real pipeline on a synthetic scene of known size — a 300×150 px object beside a 150 px marker
that is really 50 mm, which must come back as 100×50 mm. That one check covers marker
detection, the millimetres-per-pixel maths, mesh scaling, the oriented bounding box and the
GLB export. It exits non-zero if anything required fails, and reports the GPU and the two
backends as optional.

Step 2 gets the code (clone your GitHub repo, or copy the folder from Drive — set
`CODE_SOURCE`) and caches both model weights and the pip cache on Drive, so the
multi-gigabyte checkpoints download once rather than once per session and later installs are
much faster. Re-run step 2 after the restart: it restores the cache paths and the working
directory.

Step 7 prints a `gradio.live` URL that works for 72 hours, and only while the cell is
running. Open it on your phone to upload photos straight from the camera.

**Neither TripoSR nor Hunyuan3D needs anything compiled.** TripoSR imports `torchmcubes`, a
CUDA extension that frequently fails to build; the adapter substitutes scikit-image's
marching cubes when it is missing, which is slower and can order the axes differently. That
rotates the model in the viewer and cannot change the measurements, because those come from
the oriented bounding box, which is rotation invariant. Hunyuan3D's two extensions belong to
its texture pass, which wants ~21 GB of VRAM and stays off on a T4.

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

81 tests, no GPU and no model weights required. They pass on both OpenCV 4.6 with
numpy 1.26 and OpenCV 5.0 with numpy 2.4; ArUco's API changed at OpenCV 4.7 and
`app/scale.py` handles both spellings.

`python tools/doctor.py` works locally too, and is the fastest way to see whether an
environment is sound: it reports the GPU and both generation backends as optional.

Both backends can be verified locally without a GPU: install CPU torch
(`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`), run
`python tools/colab_setup.py`, and the doctor then reports whether each one *imports*,
which is where they usually fail. Generating a mesh still needs CUDA.

**On Windows**, cloning Hunyuan3D prints `error: unable to create file ... Filename too
long` for two files under `hy3dshape/tools/mini_trainset/`. Their paths exceed the
260-character limit Windows enforces by default. Nothing here reads them, the packages
themselves arrive intact, and `colab_setup.py` says so rather than treating it as fatal.
Colab is Linux and never hits this.

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
  "length_mm": 133.4,
  "width_mm": 50.0,
  "height_mm": 20.8,
  "volume_cm3": 139.1,
  "surface_area_cm2": 209.9,
  "watertight": true,
  "volume_basis": "mesh",
  "measurement_tier": "reference_marker",
  "estimated_error_pct": 4.0,
  "obb_extents_mm": [133.4, 50.0, 20.8],
  "detail": { "scale": "...", "scale_solution": "...", "inferred_depth_mm": 20.8 }
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

**Checked**, by the test suite (81 tests, all on CPU):

- Marker detection recovers a known millimetres-per-pixel ratio to within 1% on
  synthetic images, and the generated printable marker round-trips to within 0.1% of its
  requested size.
- The **printable sheet itself**, ruler ticks and caption included, photographed at 120,
  150 and 240 px across, measures a known object back to within 0.2%. This is a separate
  test from the one above because the bias it guards against — black marks close enough
  to the marker that the detector traces them as part of it — cannot appear in a scene
  built from a bare ArUco square.
- The notebook's cells parse, call this project's functions with arguments they accept,
  and — for the session-setup cell, which step 4 tells you to re-run — do not delete the
  model repositories the install step cloned.
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
  colab_setup.py     freeze the baseline, then install everything in one transaction
  doctor.py          verify libraries and the whole pipeline, in a fresh process
  make_marker.py     printable marker at an exact size
  validate.py        measured error against tape-measured truth
  colab_launch.py    Colab-safe Gradio launch (no broken iframe)
notebooks/
  colab_photo_to_3d.ipynb
```

Adding a generation backend means one file in `app/generators/` and one line in
`_BACKENDS` in `generators/base.py`. Nothing downstream knows which model produced the
mesh: scale is solved independently and applied afterwards.

## When Colab breaks

First response to almost anything: **`!python tools/doctor.py`**. It names the layer that
failed instead of leaving you to infer it from a traceback that blames the wrong package.

**An ImportError from inside numpy or Pillow itself** — `cannot import name '_center' from
'numpy._core.umath'`, or `cannot import name '_Ink' from 'PIL._typing'`. Something installed
outside step 3 moved the package, and the move was applied only partly, so its Python files
and its compiled extension disagree. Colab's scipy, OpenCV, torch and rembg are all built
against the numpy and Pillow the image ships with, so they break together; `rembg` usually
reports it first, because it imports scipy.

Use the repair cell at the end of the notebook, then restart the runtime. Two traps make this
look unfixable if you do it by hand:

- **Reinstalling over the tree may not fix it.** pip records the new version and can leave
  stale files behind, so the metadata reports one version while the files are another. The
  repair cell deletes the package directory, its `*.dist-info` and its `.libs` first.
- **No version string in the running kernel can be trusted.** `__version__` reflects what was
  imported first, and `importlib.metadata.version` reads metadata that may disagree with the
  files. Check by importing in a separate process, which is what the repair cell and the
  doctor both do.

Installing through `tools/colab_setup.py` prevents this: the constraints file makes the
downgrade impossible in the first place, and the script reports anything that moved anyway.

**`No module named 'pymeshlab'`** when loading Hunyuan3D. `hy3dshape` imports it and the repo
does not declare it anywhere pip will find. `colab_setup.py` installs it.

**`compile_mesh_painter.sh: python3-config: command not found`.** Only reachable if you build
Hunyuan3D's texture extensions by hand. The texture pass wants ~21 GB of VRAM and stays off
on a T4, so nothing in the normal flow compiles them, and shape generation needs neither.

**`torchmcubes` will not build.** Ignore it. The TripoSR adapter falls back to scikit-image's
marching cubes, and `tools/doctor.py` reports which implementation is in use. Pass
`--with-torchmcubes` to `colab_setup.py` if you want the faster CUDA one.

**`FileNotFoundError: .../model.fp16.safetensors not found` for Hunyuan3D.** Hugging Face
only ships `model.fp16.ckpt` (~7 GB). This project loads `.ckpt` (not `.safetensors`).
Pull latest code, run notebook **step 6b** (`python tools/download_hunyuan.py`), wait
until it prints OK, then launch with `GENERATOR = "hunyuan3d"`. Step 2 sets
`HY3DGEN_MODELS` on Drive so the 7 GB file is not re-downloaded every session.
Until then, use `triposr`.

**`RuntimeError: Error(s) in loading state_dict for TSR` / Missing `q_proj` /
Unexpected `encoder.layer`.** TripoSR's weights were saved under transformers 4.35
ViT names; Colab's Gradio stack installs transformers 5.x, which renamed them. The
TripoSR adapter remaps the keys on load - pull the latest code (`git pull` in step 2)
and re-run. Do not pin `transformers==4.35.0`: it conflicts with Gradio 6.

**"Connection errored out. Failed to fetch"** in the Gradio page. This is a Gradio 6 +
Colab bug in the **embedded notebook preview**, not a failure of this pipeline. The page
HTML loads inside Colab's iframe, then every upload / Generate call goes to an internal
hostname the browser cannot reach.

Fix: re-run step 2 (so `git pull` picks up `tools/colab_launch.py`), then step 7. The
launcher turns the iframe off (`inline=False`) and prints working URLs:

1. Colab's own port-forward link
2. a `*.gradio.live` share link (after ensuring the tunnel binary is present)
3. a Cloudflare tunnel, if Gradio share still fails

Open **one of those printed URLs in a new browser tab**. Do not use the preview inside
the notebook. Leave the cell running.

**"Could not establish scale from this photo."** Not a bug — the app refusing to guess. One
photo carries no absolute scale, so it needs something of known size. Quickest route to a
number: choose **"I know one dimension already"** and type a measurement, which works on any
photo you have already taken. Best accuracy: print the marker from step 6 and re-shoot.

**UniDepth is not installed.** It only powers the `estimate` scale source, which is a
wide-error-bar guess. Install it with `python tools/colab_setup.py --with-unidepth`, inside
the constrained transaction so its dependencies cannot move Pillow. A marker costs one sheet
of paper and is several times more accurate.

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
