# occlubio — Model-by-Model Architectural Audit & Technical Blueprint

**Reverse-engineered from the source tree** (`occlubio/`, `configs/default.yaml`, `scripts/`).
Every AI/ML model and isolated inference module in the codebase is dissected against a
fixed 9-point framework, followed by storage/search internals, a scalability profile,
a failure-mode catalog, and an exhaustive Q&A.

> **What this system is.** A face-recognition surveillance/attendance platform that is
> *occlusion-robust* (masks, sunglasses, caps, scarves, side-profiles, low light, blur).
> It runs two flows on top of one recognition engine: **enrollment** (register a person →
> build a 512-d face template) and **identification** (scan a video → who appeared, when,
> for how long, plus unknown-person alerts).

> **How to read a module page.** Each module answers the same 9 questions:
> (1) strategic utility, (2) inner mechanics + classification-vs-regression + what it
> computes, (3) training data, (4) pipeline connectivity (series/parallel + DAG edges),
> (5) data representation & storage (shapes + dtypes), (6) search/matching logic,
> (7) boundary data types out, (8) scalability / Big-O, (9) failure modes.

---

## 0. Module inventory (what actually runs)

The engine is a **series pipeline with one parallel fan-out** (quality ∥ occlusion ∥
anti-spoof share the same aligned crop). Modules 1–11 are the runtime path; 12–14 are the
offline training path that *produces* the optional custom recognizer (module 7).

| # | Module | File | Kind | Learned? | Cls / Reg / Other |
|---|---|---|---|---|---|
| 1 | SCRFD face detector | `pipeline/face_analyzer.py` | CNN (ONNX) | yes (pretrained) | multi-task: classification + regression |
| 2 | Face aligner (Umeyama) | `pipeline/aligner.py` | geometric transform | no | deterministic (linear algebra) |
| 3 | Quality gate (FIQA proxy) | `pipeline/quality.py` | hand-crafted scorer | no | regression → threshold |
| 4 | Occlusion estimator | `pipeline/occlusion.py` | rule-based classifier | no | classification (+ ratio) |
| 5 | Anti-spoof (MiniFASNet) | `pipeline/antispoof.py` | CNN (ONNX) | yes | binary classification |
| 6 | ArcFace recognizer | `pipeline/face_analyzer.py` (via insightface) | CNN (ONNX) | yes (pretrained) | metric embedding |
| 7 | Custom recognizer | `pipeline/recognizer.py` | CNN (ONNX) | yes (you train it) | metric embedding |
| 8 | IoU tracker | `tracking/tracker.py` | combinatorial | no | assignment / matching |
| 9 | Track-embedding fusion | `pipeline/engine.py` | averaging | no | vector reduction |
| 10 | FAISS gallery (1:N) | `gallery/faiss_gallery.py` | vector index | no | nearest-neighbor retrieval |
| 11 | Identity-log analytics | `analytics/identity_log.py` | aggregation | no | voting + clustering |
| 12 | EmbeddingNet backbone | `training/train.py` | CNN (MobileNetV3) | trained here | feature extractor |
| 13 | AdaFace head | `training/losses.py` | margin softmax | trained here | classification (training-only) |
| 14 | ArcFace head | `training/losses.py` | margin softmax | trained here | classification (training-only) |

**Reading the "Cls / Reg" column:** a face *recognizer* is neither classification nor
regression at inference — it is a **metric-embedding** model. It is *trained* with a
classification-style margin loss (module 13/14) but at inference it emits a raw 512-d
vector and never a class probability. The detector is genuinely *both*: it classifies each
anchor (face vs. background) and regresses box + landmark coordinates.

---

## 1. The whole system as a Directed Acyclic Graph

### 1.1 Identification pipeline (video → report)

```
 CAMERA / VIDEO FILE / IMAGE        frame: BGR uint8  ndarray [H, W, 3]
        |
        v
 +-------------------------------+
 | (1) SCRFD DETECTOR            |  det_10g.onnx, fixed input 640x640
 |     multi-task CNN            |  OUT/face: bbox[4] + kps[5,2] + det_score
 +-------------------------------+
        |  list[FaceResult]  (N faces this frame)
        v
 +-------------------------------+
 | (8) IoU TRACKER (video only)  |  greedy IoU match -> track_id : int
 +-------------------------------+
        |  faces + track_id
        v                     ---- the rest runs PER FACE, in series ----
 +-------------------------------+
 | (2) ALIGNER (Umeyama)         |  5 kps -> 2x3 affine -> warp -> 112x112 crop
 +-------------------------------+
        |  aligned crop  [112, 112, 3] uint8
        +--------------------+--------------------+--------------------+
        v                    v                    v                    |
 +-------------+     +--------------+     +--------------+             | PARALLEL
 | (3) QUALITY |     | (4) OCCLUSION|     | (5) ANTISPOOF|             | fan-out
 | FIQA proxy  |     | skin-ratio   |     | MiniFASNet   |             | (same crop)
 | -> score f  |     | -> type,ratio|     | -> live,score|             |
 +-------------+     +--------------+     +--------------+             |
        \___________________ GATE: (quality passed) AND (live) ? _____/
                              |  yes                    |  no  -> face dropped, no ID
                              v
 +-------------------------------+
 | (6 or 7) RECOGNIZER           |  aligned 112x112 -> embedding [512] float32, L2-normed
 +-------------------------------+
        |  embedding [512]
        v
 +-------------------------------+
 | (9) TRACK FUSION (video)      |  mean of last <=8 embeddings on this track, renormalized
 +-------------------------------+
        |  fused embedding [512]
        v
 +-------------------------------+
 | (10) FAISS 1:N SEARCH         |  IndexFlatIP (cosine); top-1 vs match_threshold 0.45
 +-------------------------------+
        |  (identity : str, score : float)
        v
 +-------------------------------+
 | (11) IDENTITY LOG / ANALYTICS |  per-track majority vote, appearance windows,
 |                               |  unknown-cluster alerts, optional target match
 +-------------------------------+
        |  JSON report  +  annotated MP4
        v
   DB (Sighting, Alert rows)  +  Dashboard / API response
```

### 1.2 Enrollment pipeline (photos/webcam → template)

```
 Uploaded photos + webcam JPEG snapshots   ->  list of BGR frames
        |
        v   for EACH image:
 (1) SCRFD detect  ->  pick the LARGEST face (max bbox area)
        |
        v
 (2) Align -> 112x112
        |
        v
 (3) Quality gate  --fail-->  skip this image
        |  pass
        v
 (6/7) Embed  ->  512-d vector
        |   collect all accepted embeddings
        v
   mean(embeddings) -> L2-normalize  =  TEMPLATE  [512] float32
        |
        v
 (10) DUPLICATE CHECK: FAISS top-1 search of the template
        |   score >= dup_threshold(0.5) AND matches a DIFFERENT user -> HTTP 409, reject
        |  ok
        v
   DB: Enrollment(embedding=bytes, dim=512, n_images, quality)
        |
        v
   REBUILD FAISS IndexFlatIP from ALL enrollments  (DB is source of truth)
```

### 1.3 Series vs. parallel topology (explicit)

- **Series (sequential dependency):** 1 → 8 → 2 → {gate} → 6/7 → 9 → 10 → 11. Each stage
  consumes the previous stage's output; you cannot embed before you align, cannot search
  before you embed.
- **Parallel (concurrent branching, same input):** modules **3, 4, 5** all read the *same*
  aligned 112×112 crop independently. In the code they run one-after-another on a single
  thread (no threads are spawned), but they are **logically parallel** — none depends on
  another's output; they are three independent read-only analyses of one crop. Their
  results are combined by a boolean **gate**: recognition proceeds only if
  `quality.passed AND antispoof.live`. Occlusion is advisory metadata (it does not block).
- **Parallel across faces:** within a frame, every detected face is processed
  independently — this is data-parallel and is what a GPU batch would exploit.
- **Derived-not-inline:** the FAISS index (10) is a *derived* structure rebuilt from the
  relational DB; the DB — not FAISS — is the source of truth.

### 1.4 Per-edge data-type contract

| Edge (from → to) | Payload | Type / shape |
|---|---|---|
| camera → detector | frame | `ndarray[H,W,3]` uint8 (BGR) |
| detector → tracker | detections | `list[FaceResult]`; each: bbox `float32[4]`, kps `float32[5,2]`, det_score `float` |
| tracker → aligner | detection | same + `track_id: int` |
| aligner → gate | aligned crop | `ndarray[112,112,3]` uint8 |
| quality → gate | score | `float` in [0,1] + `bool` pass |
| occlusion → meta | class | `dict{type:str, occluded_ratio:float, use_periocular:bool}` |
| antispoof → gate | liveness | `dict{live:bool, score:float}` |
| recognizer → fusion | embedding | `ndarray[512]` float32, L2-norm = 1 |
| fusion → search | embedding | `ndarray[512]` float32, L2-norm = 1 |
| search → analytics | match | `(identity:str, score:float in [-1,1])` |
| analytics → DB/UI | report | JSON: people[], unknown_alerts[], target, processing |

---

## 2. Module-by-module audit

Every module below follows the same 9-point structure.

---

### 2.1 SCRFD Face Detector  (module 1)

**1. Strategic utility.** Nothing else in the pipeline can run until faces are *found and
landmarked*. SCRFD locates every face in a full frame and returns the 5 keypoints
(eyes, nose, mouth corners) that alignment (2) needs. It is the entry point and the
throughput governor: it runs on *every* frame, whereas the expensive recognizer runs only
on gated best-frames. Business value: converts raw pixels into a bounded set of
"here are the faces, here is where their eyes/nose/mouth are."

**2. Inner mechanics.** `det_10g.onnx` from the InsightFace **buffalo_l** pack — an
**SCRFD-10GF** detector (Sample-and-Computation-Redistribution, ICLR 2022). Backbone is a
ResNet-style CNN with an FPN neck and multi-level heads at strides 8/16/32. It is a
**multi-task, anchor-based** network:
  - *Classification branch* — per anchor, a **face/background** score (sigmoid-style
    confidence in [0,1]); thresholded at `det_thresh = 0.5`.
  - *Regression branch A* — bounding-box offsets (distance-to-4-sides), a **regression**.
  - *Regression branch B* — 5 landmark offsets (10 numbers), a **regression**.
  Post-processing: decode anchors → **Non-Max Suppression** to remove overlapping boxes.
So it is *both* classification (is this a face?) and regression (where exactly?).

**3. Training data.** Pretrained by InsightFace on **WIDER FACE** (the standard face-detection
benchmark: ~32k images, 393k faces across scale/pose/occlusion/illumination). It expects a
full BGR scene resized to 640×640; it is scale-robust because of the multi-stride heads.
(The architecture doc recommends fine-tuning on MAFA/masked imagery to harden heavily
occluded heads — a hook, not yet done.)

**4. Pipeline connectivity.** *First node, no upstream model.* **Feeds:** the tracker (8)
in video mode, otherwise straight to the aligner (2). It is in **series** with everything
downstream. Input edge: raw frame. Output edge: `list[FaceResult]`.

**5. Data representation.** Input is a dense image tensor internally
`[1, 3, 640, 640] float32` (letterboxed). Output per face is small structured data, not a
matrix: `bbox float32[4]`, `kps float32[5,2]`, `det_score float`. Faces with longest side
`< min_face(24 px)` are discarded in `analyze()`.

**6. Search/matching.** None — it is a dense predictor, not a retrieval module. Its internal
"matching" is anchor decoding + NMS (greedy IoU suppression of overlapping candidate boxes).

**7. Boundary data out.** A **list of structured coordinate records** (numbers): 4 box
floats, 10 landmark floats, 1 score float per face. No image is passed forward except the
original frame (the aligner re-crops from it).

**8. Scalability / Big-O.** Cost is fixed by the 640×640 input, so **O(1) per frame w.r.t.
the number of faces or gallery size** — one forward pass finds all faces at once. NMS is
`O(A log A)`..`O(A²)` in the number of surviving anchors `A`. Throughput: ~2–5 ms/frame on
Jetson Orin NX (INT8), tens of ms on CPU. Bottleneck: image resolution and CPU-only ONNX
runtime. Memory: fixed engine footprint, independent of scene content.

**9. Failure modes.** (a) Faces smaller than ~24 px or below `det_thresh 0.5` are never
emitted — **no detection ⇒ no recognition** (silent drop). (b) Extreme yaw/profile or a
fully helmeted head yields no box (the architecture doc's parallel body/gait modality is
the intended mitigation — a hook). (c) Bad or occluded landmarks corrupt alignment
downstream (this is the single largest silent-accuracy killer). (d) Motion blur / very low
light lowers confidence below threshold. (e) Heavy crowds increase NMS cost and can merge
/ suppress overlapping faces. Cascading effect: every downstream stage inherits detector
misses — a missed face is simply absent from the report.

---

### 2.2 Face Aligner — Umeyama Similarity Transform  (module 2)

**1. Strategic utility.** The recognizer was trained on faces in one *canonical* geometry
(eyes/nose/mouth at fixed pixel locations). Alignment warps each detected face into that
exact geometry so the embedding is comparable across images. **This is the highest-leverage
step for occluded FR**: a mis-aligned face produces a garbage embedding even if every other
stage is perfect.

**2. Inner mechanics.** Deterministic linear algebra — **no learned weights**. `estimate_norm`
solves an **Umeyama** least-squares **similarity transform** (rotation + uniform scale +
translation, *no shear*) mapping the 5 detected landmarks onto the fixed ArcFace canonical
template `ARCFACE_DST` (the standard insightface 112×112 5-point layout). It is neither
classification nor regression in the ML sense — it is a closed-form SVD solution. Then
`cv2.warpAffine` applies the resulting 2×3 matrix to produce a 112×112 crop.

**3. Training data.** None — parameter-free. The only "prior" is the hard-coded canonical
`ARCFACE_DST` point set, which *must* match the recognizer's training alignment exactly.

**4. Pipeline connectivity.** **Series.** Upstream: detector landmarks (1). Downstream: the
aligned crop feeds the parallel triplet (3/4/5) and the recognizer (6/7). Every per-face
branch reads *its* output.

**5. Data representation.** Input: `kps float32[5,2]`. Output: dense image
`ndarray[112,112,3] uint8`. Internal transform matrix `M float32[2,3]`.

**6. Search/matching.** None.

**7. Boundary data out.** A **112×112×3 image tensor** (pixels), the only place downstream a
full image is produced. The transform math is `O(1)` (a 2×2 SVD on a demeaned 5-point set).

**8. Scalability / Big-O.** **O(1) per face** (constant 5 points, fixed output size). The
`warpAffine` cost is `O(112·112)` = constant. Negligible vs. the CNNs. Fully CPU, no memory
growth.

**9. Failure modes.** (a) If an occluder displaces a landmark (e.g., sunglasses shift the
detected eye point), the similarity fit rotates/scales the whole face wrongly → embedding
collapse. (b) Near-profile faces where only 3 of 5 landmarks are reliable are fit poorly
(the architecture doc's partial-affine / pose-bucket fallback is a hook, not implemented —
current code always uses all 5). (c) A degenerate landmark set (rank-deficient) returns a
`NaN` matrix; `warpAffine` then yields a black/garbage crop. (d) **Train/inference alignment
mismatch** — if you swap in a custom model trained with a different canonical layout, cosine
scores silently degrade.

---

### 2.3 Face Quality Gate — FIQA proxy  (module 3)

**1. Strategic utility.** Cheap insurance: skip junk frames *before* paying for recognition,
and refuse to enroll a blurry/dark face as someone's permanent template. Saves compute and
prevents low-quality embeddings from polluting the gallery.

**2. Inner mechanics.** A **hand-crafted scalar scorer** (a stand-in for a learned CR-FIQA
model — see the `# HOOK`). It is **regression-like** (produces a continuous score in [0,1])
then **thresholded** to a boolean. The score is a fixed weighted sum of three sub-scores:

```
sharp_score  = clip( varLaplacian(gray) / (blur_min*3) , 0, 1)   # blur_min=40 -> /120
bright_score = 1.0                       if 30 <= mean(gray) <= 225
             = clip(1 - dist/60, 0, 1)   otherwise (dist to nearest bound)
det          = clip(det_score, 0, 1)
quality      = 0.5*det + 0.3*sharp_score + 0.2*bright_score
passes       = (quality >= min_score=0.35)
```

`varLaplacian` is the variance of the Laplacian — a classic image-sharpness statistic
(high = sharp, low = blurred). No probabilities are produced; the numbers are engineered
scores, not calibrated likelihoods.

**3. Training data.** None (rule-based). Thresholds (`blur_min 40`, brightness 30–225,
`min_score 0.35`) are hand-tuned constants in `configs/default.yaml`.

**4. Pipeline connectivity.** **Parallel** with occlusion (4) and anti-spoof (5) over the
same aligned crop. Its boolean feeds the **AND-gate** guarding the recognizer. Upstream:
aligner (2) + detector score. Downstream: gate → recognizer.

**5. Data representation.** Input: `[112,112,3]` crop. Internal: grayscale `[112,112]`.
Output: one `float` + one `bool`.

**6. Search/matching.** None.

**7. Boundary data out.** **A single number** (quality in [0,1]) and a **boolean** pass/fail.
The number is also stored on the enrollment row (`Enrollment.quality`) and averaged for the
UI.

**8. Scalability / Big-O.** **O(P)** in crop pixels `P = 112·112` = constant per face; a
Laplacian convolution + two reductions. Trivial cost, CPU only.

**9. Failure modes.** (a) The Laplacian sharpness statistic is fooled by *texture* — a busy
background patch or JPEG blocking can read as "sharp"; a smooth but perfectly in-focus face
can read as "blurry." (b) Brightness uses the whole-crop mean, so a half-lit face (one side
blown out, one side black) can average into the "OK" band. (c) It has **no notion of pose or
occlusion** — a sharp, well-lit *profile* passes and then embeds poorly. (d) Threshold too
high ⇒ everyone rejected at enrollment ("no usable face"); too low ⇒ junk templates. It is a
proxy; the intended CR-FIQA upgrade is a hook.

---

### 2.4 Occlusion Estimator  (module 4)

**1. Strategic utility.** Tells the rest of the system *what kind* of occlusion is present
(clear / lower-face / upper-face / heavy / cap) and raises a `use_periocular` flag when the
lower face is covered. Drives the on-video `[mask]`/`[lower_face]` overlay label and is the
intended routing signal for a future periocular sub-model.

**2. Inner mechanics.** A **rule-based classifier** — no weights. It measures **skin
coverage** in three fixed bands of the aligned 112×112 crop by converting to **YCrCb** and
counting pixels inside a skin chroma box (`135≤Cr≤180`, `85≤Cb≤135`):

```
eye band     = rows 40..64,  cols 16..96
lower band   = rows 70..112, cols 16..96
forehead     = rows  0..38,  cols 20..92

if lower_skin < 0.25:                    type = lower_face, ratio = 1-lower_skin
if eye_skin  < 0.20 and eye_gray < 70:   type = upper_face | heavy
if fore_skin < 0.15 and type == clear:   type = cap
use_periocular = type in {lower_face, heavy}
```

Output is a **discrete class** (classification) plus a continuous `occluded_ratio`
(regression-like). It does **not** gate recognition — it is advisory metadata.

**3. Training data.** None. The skin-chroma thresholds are hand-chosen for typical skin
tones — a known bias source (see failures).

**4. Pipeline connectivity.** **Parallel** with quality (3) and anti-spoof (5). Upstream:
aligned crop. Downstream: attaches `occlusion` dict to `FaceResult`; consumed by the drawing
overlay and (future) periocular routing. Not on the gate's critical path.

**5. Data representation.** Input `[112,112,3]`; internal YCrCb band slices; output
`dict{type:str, occluded_ratio:float, use_periocular:bool}`.

**6. Search/matching.** None.

**7. Boundary data out.** A **small dict** (one string label + one float ratio + one bool).

**8. Scalability / Big-O.** **O(P)** over the three band areas = constant per face. Cheap
color-convert + boolean-mask means. CPU only.

**9. Failure modes.** (a) **Skin-tone bias** — the fixed Cr/Cb box is calibrated to certain
skin tones; darker or very light skin, or strong color casts, misread as "occluded." (b)
Beards, heavy shadow, or backlight lower skin ratio → false "lower_face." (c) Skin-colored
masks/scarves read as "clear" (false negative). (d) It is purely 2D band statistics with no
semantic parsing (the BiSeNet/FaRL parser is the intended upgrade). Because it never blocks
recognition, its errors degrade *labels/routing*, not the identity decision directly.

---

### 2.5 Anti-Spoof — MiniFASNet  (module 5)

**1. Strategic utility.** Presentation-Attack Detection: block a printed photo or a phone
replay from being recognized (or from enrolling). In surveillance, a printed face is the
cheapest attack; liveness gates recognition. **Disabled by default** (`enabled: false`) —
it is wired in but waits for you to supply a MiniFASNet ONNX.

**2. Inner mechanics.** A small CNN (**MiniFASNet**, from Silent-Face-Anti-Spoofing) run via
ONNXRuntime. It is a **binary/multi-class classifier**. Preprocess: resize crop to
`input_size (80×80)`, scale `(x-127.5)/128`, transpose to `[1,3,80,80]`. It outputs raw
class logits; the code applies **softmax** and takes the **last class as the live
probability**: `live = softmax(out)[-1]`, `is_live = live >= min_score(0.5)`. This is the
one module that emits a true **calibrated-ish probability** via softmax.

**3. Training data.** (When you supply weights) Silent-Face MiniFASNet is trained on RGB
presentation-attack datasets (e.g., CASIA-SURF / CelebA-Spoof-style Fourier+patch crops). It
expects a tight, aligned RGB face crop. The architecture doc notes stronger 2024–2025
domain-generalization models (CFPL-FAS, BUDPT) and IR/depth sensors as upgrades.

**4. Pipeline connectivity.** **Parallel** with quality (3) and occlusion (4). Its boolean
`live` is ANDed with quality-pass to form the recognizer gate. If disabled, it returns
`{live:True, score:1.0}` (fail-open). Upstream: aligned crop. Downstream: the gate.

**5. Data representation.** Input `[1,3,80,80] float32`. Output: small logit vector →
softmax probability vector; the scalar `live_score float` is kept.

**6. Search/matching.** None (it is a classifier, not a retriever).

**7. Boundary data out.** A **boolean** (`live`) and a **probability float** in [0,1]. A
non-live face is dropped before recognition and drawn red as `SPOOF`.

**8. Scalability / Big-O.** **O(1) per face** (fixed 80×80). ~1–2 ms on Jetson. Runs only on
gated candidate faces. Bottleneck: adds a second ONNX session; on CPU it competes with the
detector/recognizer for cores.

**9. Failure modes.** (a) Disabled by default ⇒ **zero spoof protection** until weights are
supplied (fail-open is a security gap to note). (b) RGB-only PAD generalizes poorly across
unseen attack media/lighting (cross-dataset drift). (c) High-quality replays / 3D masks
defeat passive RGB (needs IR/depth). (d) A false "spoof" on a live person silently removes
them from the report (fail-closed on that face). (e) Softmax assumes the "live" class is the
last index — a differently-ordered ONNX head inverts the decision.

---

### 2.6 ArcFace Recognizer — `w600k_r50`  (module 6, default)

**1. Strategic utility.** **The core.** Turns an aligned face into a 512-d vector where the
*same person's* faces are close and *different people* are far — enabling 1:N identity search
against the gallery. This vector *is* the biometric template. Everything before it exists to
feed it a clean, live, well-aligned crop; everything after it searches or aggregates its
output.

**2. Inner mechanics.** The recognition model of InsightFace **buffalo_l**: `w600k_r50.onnx`,
an **IResNet-50** CNN trained with the **ArcFace** additive-angular-margin loss (partial-FC).
At **inference it is a metric-embedding model — neither classification nor regression**: it
emits a **512-d feature vector**, *not* a class or a probability. There is **no softmax at
inference**; the final layer is a linear embedding, then **L2-normalization** so the vector
lies on the unit hypersphere. The "probability" a user asks about is *not* produced here — it
is produced later as a **cosine similarity** in FAISS (module 10). (During training the
ArcFace head *was* a margin softmax classifier over identities; that head is discarded for
deployment.)

**3. Training data.** InsightFace trained `w600k_r50` on a large web-scraped face corpus
(the "**W600K**" cleaned WebFace/Glint-family set — hundreds of thousands of identities,
millions of images). It expects a **112×112 aligned BGR** face normalized to roughly
[-1,1]. This is why alignment (2) must match the training geometry exactly.

**4. Pipeline connectivity.** **Series**, gated. Upstream: aligner (2) + the AND-gate
(3∧5). In the *default* path insightface's `FaceAnalysis.get()` runs detection **and**
recognition together and hands back `f.embedding`, which the code L2-normalizes. Downstream:
track fusion (9) → FAISS search (10).

**5. Data representation.** Input `[1,3,112,112] float32`. Output **dense vector
`ndarray[512] float32`, unit L2 norm**. Stored per person as a `float32` blob in
`Enrollment.embedding` (2048 bytes = 512×4) and as one row of the FAISS matrix `[N,512]`.

**6. Search/matching.** The model does not search; its output *is the query* for the FAISS
cosine search (module 10). Similarity between two faces = **dot product of their L2-normed
embeddings = cosine similarity** ∈ [-1,1].

**7. Boundary data out.** A **512-number float32 vector** on the unit sphere. Not a label,
not a probability — an embedding. The *decision* (known/unknown) happens one stage later.

**8. Scalability / Big-O.** **O(1) per face** (fixed 112×112), so **O(F)** per frame for F
gated faces. ~1–3 ms/face FP16 on Jetson, ~30–100 ms on CPU (the platform's real
bottleneck — hence `--stride` on video). Independent of gallery size. Memory: one fixed
engine; embeddings are tiny (2 KB each). **Keep this model FP16, not INT8** — INT8 shifts
cosine distances and inflates false matches (architecture pitfall #2).

**9. Failure modes.** (a) Garbage-in from mis-alignment (2) ⇒ garbage embedding. (b) Heavy
occlusion (mask+sunglasses) pushes same-person cosine below `match_threshold 0.45` ⇒ the
person reads "unknown" (a **false reject**). (c) Look-alikes / identical twins can exceed
threshold ⇒ **false accept**. (d) Domain gap: trained on web photos, deployed on low-res
surveillance frames → distribution shift lowers separability (the occlusion-aware custom
model in 7 is the intended fix). (e) Demographic bias in TAR@FAR under occlusion/low-light
(must be measured). (f) An embedding is partially invertible back to a face ⇒ storing it
unprotected is a privacy failure (encrypt / IronMask hook).

---

### 2.7 Custom Recognizer — EdgeFace / MobileNetV3 + AdaFace  (module 7, optional)

**1. Strategic utility.** The upgrade path: replace the generic ArcFace with **your own
occlusion-aware** embedder, trained with heavy synthetic occlusion (masks/sunglasses/caps)
so masked/blurred faces still land near their clean template. Activated by setting
`recognition.custom_onnx` — the detector then runs **detection-only** and this model does the
embedding.

**2. Inner mechanics.** `CustomRecognizer` loads any ONNX embedder via ONNXRuntime. Same
**metric-embedding** class as 2.6 (no softmax at inference). Preprocess is explicit in code:
resize to 112×112 if needed, `(x-127.5)/128.0`, transpose to `[1,3,112,112]`, run, `ravel`,
**L2-normalize**. The reference model produced by `training/train.py` is a
**MobileNetV3-small** backbone + a Linear→BatchNorm head to 512-d (module 12), trained with
**AdaFace** (module 13). Output: 512-d unit vector.

**3. Training data.** *You* supply it: `root/<identity>/<aligned_112_img>` folders
(`AlignedFaceDataset`). During training every image is hit with `OcclusionAugmentor`
(mask/sunglasses/cap/scarf/random-erase at `occlude_prob 0.5`, plus motion-blur/low-light/
JPEG/downscale at `photometric_prob 0.5`) and random horizontal flip. So it is explicitly
*trained to see occlusion as normal*. It expects the exact same 112×112 aligned geometry as
the ArcFace path.

**4. Pipeline connectivity.** **Series**, drop-in replacement for 2.6. Upstream: aligner (2)
+ gate. Downstream: fusion (9) → FAISS (10). When present, `FaceAnalyzer` is constructed
with `detection_only=True`, so detection (1) and embedding (7) are cleanly separated.

**5. Data representation.** Input `[1,3,112,112] float32`. Output `ndarray[512] float32`
unit norm — identical contract to 2.6, so the gallery/DB need no changes.

**6. Search/matching.** Same as 2.6 — output is the FAISS query vector; cosine matching.

**7. Boundary data out.** **512-number float32 unit vector.**

**8. Scalability / Big-O.** **O(1) per face.** MobileNetV3-small is lighter than IResNet-50
(fewer FLOPs, edge-friendly), so *faster* than the default recognizer at some accuracy cost —
recovered via distillation from a big teacher (architecture §3). Bottleneck identical to 2.6.

**9. Failure modes.** (a) **Dimension mismatch** — if you train a 256-d head but leave
`embedding_dim 512`, the gallery `add()` raises `dim mismatch`. (b) **Alignment mismatch** —
if your training alignment differs from `norm_crop`, cosine scores silently collapse. (c)
Under-trained on too few identities ⇒ poor separability, worse than the pretrained default
(the README's "always keep the pretrained baseline" rule). (d) Same occlusion/twin/OOD
limits as any embedder. (e) INT8 export without QAT inflates false matches.

---

### 2.8 IoU Tracker  (module 8)

**1. Strategic utility.** Ties a person's face across consecutive frames into **one track**
so the system can (a) run recognition *once per track* not per frame, (b) **average
embeddings over a track** for robustness (module 9), and (c) report *appearances/durations*
instead of per-frame noise. It is what makes set-based recognition and the timeline report
possible.

**2. Inner mechanics.** A classic **greedy IoU-association** tracker — **no learning, no
motion model** (a lightweight stand-in for ByteTrack, noted as a hook). Each frame: age all
tracks; compute IoU between every detection box and every existing track box; sort candidate
pairs by IoU descending; greedily accept pairs with `IoU >= iou_thresh(0.25)`, one detection
per track; unmatched detections spawn new tracks; tracks unseen for `> max_age(30)` frames
are deleted. `min_hits(2)` marks when a track is "confirmed." It is a **combinatorial
assignment** problem solved greedily — neither classification nor regression.

**3. Training data.** None (parameter-free heuristic).

**4. Pipeline connectivity.** **Series**, immediately after the detector, **video-only**
(`recognize_image` disables it). Upstream: detector boxes. Downstream: writes `track_id` onto
each `FaceResult`, consumed by fusion (9) and analytics (11). Absent in single-image mode.

**5. Data representation.** State is a Python `list[_Track]`, each with `id:int`,
`bbox:float32[4]`, `age`, `hits`, `time_since_update`. No tensors.

**6. Search/matching.** Its matching *is* the IoU assignment: overlap area ÷ union area
between boxes, greedy max-first. No embedding is used (appearance-blind).

**7. Boundary data out.** A single **integer `track_id`** stamped on each detection.

**8. Scalability / Big-O.** **O(D·T)** per frame (D detections × T live tracks) to build the
IoU table, plus `O(DT·log DT)` to sort pairs. Fine for surveillance crowd sizes; quadratic
blow-up only with hundreds of simultaneous faces. Memory O(T). CPU only.

**9. Failure modes.** (a) **ID switches** — two people crossing/overlapping can swap IDs
because matching is appearance-blind (pure IoU). (b) Fast motion or high `--stride` moves a
box so far that IoU < 0.25 → the track breaks and the person is **re-IDed as a new track**
(fragmentation → possibly a spurious "unknown" alert). (c) Long occlusion beyond `max_age`
kills the track. (d) No motion prediction, so it cannot bridge gaps a Kalman filter would.
Fragmentation is partially healed downstream by the analytics merge (11) and unknown
clustering.

---

### 2.9 Track-Embedding Fusion  (module 9)

**1. Strategic utility.** **Set-based recognition.** A single occluded frame gives a noisy
embedding; averaging the last several embeddings on a track produces a far more stable vector
and is the practical core of "occlusion robustness at inference." One bad frame cannot flip
the identity.

**2. Inner mechanics.** No model. A per-track ring buffer (`deque(maxlen=per_track_buffer=8)`)
of recent embeddings; on each new frame it appends, takes the **mean over the stack**, and
**re-L2-normalizes**: `l2_normalize(mean(stack))`. A vector reduction — not classification or
regression.

**3. Training data.** None.

**4. Pipeline connectivity.** **Series**, between the recognizer (6/7) and search (10),
**video-only** (needs a `track_id`; in image mode the raw embedding is used). Buffers are
cleared per video job and on index rebuild.

**5. Data representation.** State: `dict[track_id → deque[ndarray[512]]]`, up to 8 vectors
each. Output: one `ndarray[512] float32` unit vector.

**6. Search/matching.** None — it *produces* the query that search consumes.

**7. Boundary data out.** A **fused 512-d unit vector** (still just 512 numbers).

**8. Scalability / Big-O.** **O(B·d)** per face, `B ≤ 8`, `d = 512` → effectively constant.
Memory O(active_tracks × 8 × 512 float32). Trivial.

**9. Failure modes.** (a) If the tracker (8) **ID-switches**, the buffer averages **two
different people** → a blended, wrong embedding (garbage query). (b) A long-lived track that
changes occlusion state mixes clean+occluded frames — usually helpful, occasionally dilutes a
good match. (c) Cold start: the first frame of a track is unaveraged (higher variance).

---

### 2.10 FAISS Gallery — 1:N Similarity Search  (module 10)

**1. Strategic utility.** The **identity decision** lives here. Given a query embedding, it
finds the nearest enrolled person and decides *known vs. unknown* by a threshold. It is also
the **duplicate-face guard** at enrollment (stops user B registering user A's face).

**2. Inner mechanics.** **FAISS `IndexFlatIP`** — an **exact, brute-force inner-product**
index over the gallery matrix. Because every vector (enrolled and query) is **L2-normalized**,
inner product **= cosine similarity**. `identify()` does a **top-1** search and applies the
decision rule `score >= match_threshold(0.45) ? label : "unknown"`. This is a
**nearest-neighbor retrieval**, not classification/regression; the "probability" the user
asks about is exactly this **cosine score ∈ [-1,1]** (≈1 same person, ≈0 unrelated).

**3. Training data.** None — it is an index, not a model. It is *populated* at runtime from
`Enrollment` rows (rebuilt on every enroll/change; DB is source of truth).

**4. Pipeline connectivity.** **Series**, terminal inference node. Upstream: fused embedding
(9) or template (enrollment). Downstream: `(identity, score)` → analytics (11) / duplicate
check / API. Swappable for Milvus/Qdrant behind the same `add/search` API.

**5. Data representation.** A **dense matrix `[N, 512] float32`** held in RAM (`index.faiss`)
plus a parallel Python `labels: list[str]` and `meta: list[dict]` (persisted as `meta.json`).
Row *i* of the matrix ↔ `labels[i]`. Enrolled templates also live as `float32` blobs in
SQLite (`Enrollment.embedding`). Not sparse — dense unit vectors.

**6. Search/matching.** **Cosine similarity via dot product** — for query `q` it computes
`scores = index · qᵀ` (an `[N]` vector of dot products), returns the top-k by score.
`IndexFlatIP` is **exact** (scans all N). The larger-scale ladder (HNSW graph / IVF-PQ
approximate ANN) is the documented upgrade, not the current code.

**7. Boundary data out.** A **(string label, float score)** pair — the first place a
**human-readable identity** and a **decision number** appear. Below threshold → the literal
string `"unknown"`.

**8. Scalability / Big-O.** Exact flat search is **O(N·d)** time per query and **O(N·d·4
bytes)** memory (N = enrolled users, d = 512). At 1e4 users ≈ 20 MB and sub-millisecond; at
1e6 ≈ 2 GB and still ms-scale on GPU but heavy on CPU → switch to **HNSW (≈O(log N) per
query)** or **IVF-PQ** (compressed, sub-linear) with an exact top-k **re-rank**. Rebuild is
**O(N)** and happens on every enrollment (fine ≤1e4; batch/incremental beyond).

**9. Failure modes.** (a) **Threshold economics** — `match_threshold 0.45` trades false
accepts vs. false rejects; occlusion compresses genuine scores toward the threshold, so a too-
high value rejects real people and a too-low value accepts look-alikes. (b) `dup_threshold
0.5` too low blocks legitimate distinct users as "duplicates"; too high lets one face
enroll twice. (c) Exact flat search degrades linearly — at millions of users CPU latency
grows and the per-enroll `O(N)` rebuild becomes a bottleneck. (d) The index is **derived**;
if it is out of sync with the DB (crash mid-rebuild) searches miss real users until rebuilt.
(e) Raw embeddings in the index/DB are a **privacy liability** (encryption/template
protection is a hook).

---

### 2.11 Identity-Log Analytics  (module 11)

**1. Strategic utility.** Converts thousands of noisy per-frame `(track, identity, score)`
tuples into the **human report**: who appeared, how many distinct appearances, first/last
seen, total visible seconds, max/avg confidence, appearance time-windows, and consolidated
**unknown-person alerts**. Also answers "was *this specific target* in the video?"

**2. Inner mechanics.** No model — pure aggregation. Per track it keeps a `TrackRecord`
(frame count, score sum/max, a **`Counter` of identity votes**, and the best-scoring
embedding). Final identity per track = **majority vote** (`votes.most_common(1)`). It then:
drops tracks with `< min_track_frames(3)`; groups tracks by identity; **merges appearance
windows** closer than `merge_gap_s(1.5 s)`; filters unknown blips shorter than
`unknown_min_seconds(0.6 s)`; and **clusters remaining unknown tracks** into distinct
"UNKNOWN-k" people by **cosine similarity ≥ unknown_merge_sim(0.45)** of their best
embeddings (a greedy single-pass clustering). `target_report` matches every track's best
embedding against a supplied target embedding by cosine ≥ threshold.

**3. Training data.** None. Behavior is set by the `analytics:` config block.

**4. Pipeline connectivity.** **Series**, terminal. Upstream: per-frame `FaceResult`s
(post-search). Downstream: the JSON report and, in the service layer, `Sighting`/`Alert` DB
rows and the dashboard.

**5. Data representation.** `dict[track_id → TrackRecord]`; each holds a `Counter`, floats,
and one `float32[512]` best-embedding. Output is nested JSON (lists of dicts with string
timestamps `MM:SS.ss` and float seconds).

**6. Search/matching.** Two cosine operations: (a) **unknown clustering** — dot products
between unknown tracks' best embeddings, greedy merge at ≥0.45; (b) **target search** — dot
product of each track's best embedding vs. the target, keep ≥ threshold.

**7. Boundary data out.** **Structured JSON**: `summary{known_people, unknown_alerts}`,
`people[]`, `unknown_alerts[]`, optional `target`, `processing{fps,frames,...}`. Numbers +
labels + time-windows — the final product.

**8. Scalability / Big-O.** Per-frame update is **O(faces)**. The report is **O(T)** in
tracks except unknown clustering, which is **O(U²·d)** in the number of unknown tracks `U`
(pairwise cosine). Fine for normal footage; a crowd of hundreds of unknowns makes clustering
the hot spot. Memory O(T·d) for stored best-embeddings.

**9. Failure modes.** (a) Inherits every tracker error — an **ID switch** can split one
person into two report entries or merge two people under one vote. (b) Majority vote can
pick the wrong identity if a track has many low-quality mismatches. (c) Unknown clustering at
0.45 can **merge two different strangers** into one alert or **split one stranger** across
occlusion changes. (d) `min_track_frames`/`unknown_min_seconds` filters can drop a real brief
appearance. (e) Timestamps assume a constant `fps`; variable-frame-rate video skews the
windows.

---

### 2.12 EmbeddingNet Backbone — MobileNetV3-small (training, module 12)

**1. Strategic utility.** The trainable body of the custom recognizer (7): a compact,
edge-friendly CNN that maps a 112×112 face to a 512-d embedding. Chosen for accuracy-per-FLOP
on Jetson-class hardware.

**2. Inner mechanics.** `timm.create_model("mobilenetv3_small_100", pretrained=True,
num_classes=0, global_pool="avg")` → a global-average-pooled feature vector → a head
`Linear(feat→512) + BatchNorm1d(512)`. `forward` returns **both** the L2-normalized embedding
**and its pre-norm magnitude** `norm` (the magnitude is the *quality signal* AdaFace
consumes). It is a **feature extractor**; classification only exists via the attached head
(13/14) during training.

**3. Training data.** Backbone weights start from **ImageNet** (transfer learning); the head
is trained from scratch on your `AlignedFaceDataset` (aligned 112×112 identity folders) with
heavy occlusion augmentation.

**4. Pipeline connectivity.** **Series → offline.** Feeds the AdaFace/ArcFace head during
training; after `export_onnx` it *becomes* module 7 in the inference graph (the head is
dropped; only the embedding output is exported).

**5. Data representation.** Input `[B,3,112,112] float32`; output embedding `[B,512]` +
norm `[B]`. Trained in mini-batches of 128.

**6. Search/matching.** None (training component).

**7. Boundary data out.** At train time: `(embedding[B,512], norm[B])`. At export time: a
single ONNX graph `input[b,3,112,112] → embedding[b,512]`.

**8. Scalability / Big-O.** Training is **O(epochs · N_images · forward_cost)**; forward is
**O(1)** per image (fixed 112²). MobileNetV3-small is deliberately low-FLOP for edge
inference. GPU-bound at train time (batch 128), CPU-capable but slow.

**9. Failure modes.** (a) Too few identities/images ⇒ overfitting, poor generalization. (b)
Wrong preprocessing at inference vs. training `(x-127.5)/128` mismatch ⇒ broken embeddings.
(c) BatchNorm with tiny batches is unstable. (d) Backbone name typo / timm version drift
changes `num_features`.

---

### 2.13 AdaFace Head (training, module 13)

**1. Strategic utility.** The loss that makes the embedding *occlusion/quality-aware*.
AdaFace **de-emphasizes** hard-to-identify (blurry/occluded) samples instead of forcing the
network to memorize them, which is empirically the best loss for low-quality faces — exactly
this project's domain.

**2. Inner mechanics.** A **margin-based softmax classifier** used **only during training**.
It normalizes the class kernel, computes `cosine = embeddings · kernelᵀ`, and applies an
**image-quality-adaptive angular margin**: it standardizes the batch's feature **norms**
(EMA-tracked `batch_mean`/`batch_std`, `t_alpha=0.01`) into a `margin_scaler`, then adjusts
both an angular margin (`g_angular`) and an additive cosine margin (`g_add`) per sample by
that scaler, and scales logits by `s=64`. High-norm (high-quality) samples get a stronger
margin; low-norm (occluded) samples get a gentler one. Params: `m=0.4, h=0.333, s=64`. It is
**classification** (produces logits fed to `CrossEntropyLoss`).

**3. Training data.** Same `AlignedFaceDataset` identities; `num_classes` = number of enrolled
identity folders. The feature **norm** it keys on is produced by the backbone (12).

**4. Pipeline connectivity.** **Series → offline, training-only.** Upstream: `(embedding,
norm, labels)` from the backbone. Downstream: logits → cross-entropy → backprop into the
backbone. **Discarded before ONNX export** (never in the inference graph).

**5. Data representation.** Kernel `Parameter[512, num_classes]`; per-batch `cosine
[B,num_classes]`; buffers `batch_mean[1]`, `batch_std[1]`. Output logits `[B,num_classes]`.

**6. Search/matching.** None.

**7. Boundary data out.** **Class logits** `[B, num_classes]` (a matrix of scores) → loss.

**8. Scalability / Big-O.** **O(B · num_classes · d)** per step (the `embeddings·kernel`
matmul). Cost grows with the number of identities — the classic large-class FR training cost
(partial-FC is the industrial fix; not implemented here). GPU memory scales with
`num_classes`.

**9. Failure modes.** (a) Very large `num_classes` blows up the kernel matmul/memory. (b)
Tiny/imbalanced per-identity image counts destabilize the norm-based margin. (c) `acos`
near ±1 is numerically touchy (guarded by clamps/eps). (d) It only helps if the augmentation
actually produces low-quality samples to down-weight.

---

### 2.14 ArcFace Head (training, module 14)

**1. Strategic utility.** The simpler, classic alternative margin loss (`--head arcface`) — a
strong baseline when you don't need AdaFace's quality adaptivity.

**2. Inner mechanics.** **Additive angular margin softmax**, training-only classification.
Normalizes the kernel, `cosine = embeddings · kernelᵀ`, adds a fixed angular margin `m=0.5`
to the *true-class* angle (`cos(θ + m)`), scales by `s=64` → logits → cross-entropy. Unlike
AdaFace the margin is **constant** (quality-agnostic).

**3. Training data.** Identical to 2.13 (`AlignedFaceDataset`).

**4. Pipeline connectivity.** **Series → offline, training-only**; interchangeable with 2.13.
Discarded before export.

**5. Data representation.** Kernel `Parameter[512, num_classes]`; logits `[B, num_classes]`.

**6. Search/matching.** None.

**7. Boundary data out.** **Class logits** `[B, num_classes]` → loss.

**8. Scalability / Big-O.** **O(B · num_classes · d)** per step, same large-class scaling as
AdaFace.

**9. Failure modes.** (a) Fixed margin can't protect low-quality samples → weaker on occluded
data than AdaFace (the whole reason AdaFace is default). (b) Same large-`num_classes` cost.
(c) Numerical care around `acos`.

---

## 3. Data representation & storage — master table

| Where | Structure | Shape | Dtype | Sparse/Dense |
|---|---|---|---|---|
| Camera frame | image tensor | `[H,W,3]` | uint8 (BGR) | dense |
| Detector input | image tensor | `[1,3,640,640]` | float32 | dense |
| Detector output | struct records | bbox`[4]`,kps`[5,2]`,score | float32 | n/a |
| Aligned crop | image tensor | `[112,112,3]` | uint8 | dense |
| Recognizer input | image tensor | `[1,3,112,112]` | float32 | dense |
| **Embedding** | **unit vector** | **`[512]`** | **float32** | **dense** |
| Track fusion buffer | vectors | `≤8 × [512]` | float32 | dense |
| FAISS index | **matrix** | `[N,512]` | float32 | dense |
| FAISS labels/meta | list | `N` | str / dict | n/a |
| DB `Enrollment.embedding` | byte blob | 2048 bytes (512×4) | float32 bytes | dense |
| DB `users/jobs/sightings/alerts` | relational rows | — | mixed | n/a |
| Report | nested JSON | — | str/float/int | n/a |

**Key facts:** the biometric unit of currency is a **dense 512-d float32 vector on the unit
hypersphere**. The gallery is a **dense [N,512] matrix**. Nothing is sparse. The relational
DB is the **source of truth**; the FAISS matrix is a **derived cache** rebuilt from it.

---

## 4. Vector search — the retrieval math in one place

- **Vectors are L2-normalized**, so `‖v‖ = 1`. Therefore
  **cosine(a,b) = a·b = 1 − ½‖a−b‖²** — cosine, dot product, and (squared) Euclidean are
  monotonically equivalent here. FAISS uses the **inner product** (`IndexFlatIP`).
- **Query:** one embedding `q[512]`. **Index:** matrix `G[N,512]`. **Scores:** `s = G·qᵀ`,
  an `[N]` vector; top-k by score. `IndexFlatIP` is **exact** (scans all N rows).
- **Decision rule (1:N):** `argmax_i s_i`; report the label if `s_max ≥ 0.45` else
  `"unknown"`. The score *is* the confidence.
- **Duplicate check (enroll):** same search; reject if `s_max ≥ 0.5` against a *different*
  user.
- **Scaling ladder** (documented, not all coded):
  - ≤10k users: `IndexFlatIP` exact — trivial, ms-scale. (current)
  - 10k–1M: **HNSW** graph (`≈O(log N)`/query) or **IVF-PQ** (compressed) + exact top-k
    **re-rank**.
  - \>1M / multi-node: **Milvus/Qdrant** sharded, GPU (cuVS), same `add/search` contract.

---

## 5. Scalability profile — Big-O and bottlenecks

| Module | Time / unit | Memory | Scales with | Bottleneck |
|---|---|---|---|---|
| 1 Detector | O(1)/frame (fixed 640²) | fixed engine | frame rate, resolution | CPU ONNX; NMS in crowds |
| 2 Aligner | O(1)/face | none | faces/frame | negligible |
| 3 Quality | O(P)/face | none | faces | negligible |
| 4 Occlusion | O(P)/face | none | faces | negligible |
| 5 Anti-spoof | O(1)/face | fixed engine | gated faces | 2nd ONNX session |
| 6/7 Recognizer | O(1)/face → O(F)/frame | fixed engine | gated faces | **CPU embed ~30–100 ms** |
| 8 Tracker | O(D·T)/frame | O(T) | faces × tracks | crowds |
| 9 Fusion | O(B·d)/face | O(tracks·8·512) | active tracks | negligible |
| 10 FAISS flat | **O(N·d)/query** | **O(N·d)** | gallery size N | linear scan at ~1M+ |
| 11 Analytics | O(T)+O(U²·d) | O(T·d) | tracks, unknowns | unknown clustering |
| 12 Backbone (train) | O(epochs·N_img) | GPU | dataset size | GPU |
| 13/14 Heads (train) | O(B·num_classes·d) | O(num_classes·d) | #identities | large-class matmul |

**Concurrency boundaries.** The video job is **single-threaded** and holds a service
`RLock` (`self._lock`) for the whole run — so **one video is processed at a time per process**.
Horizontal scale is by **process/worker** (the platform docs point to Celery/RQ + Redis and
GPU workers), and by moving embed/search to GPU (`onnxruntime-gpu`, `faiss-gpu`) or Triton.
The DB and vector store scale independently of the API.

**Primary end-to-end bottleneck today:** CPU embedding (6/7) at tens of ms/face — the reason
video uses `--stride` (process every k-th frame). Secondary: FAISS becomes the bottleneck
only past ~1M enrolled users.

---

## 6. Failure-mode catalog (where each model breaks)

| Failure | Trigger | Which module | Symptom | Mitigation (in repo / hook) |
|---|---|---|---|---|
| No detection | face <24 px, <0.5 conf, profile, helmet | 1 | person absent from report | fine-tune SCRFD / body+gait (hook) |
| Alignment collapse | occluder shifts a landmark; profile | 2 | garbage embedding, wrong ID | partial-affine/pose-bucket (hook) |
| False "blurry"/"sharp" | texture fools Laplacian | 3 | good frame dropped / junk kept | CR-FIQA model (hook) |
| Occlusion mislabel | skin-tone/chroma bias | 4 | wrong overlay/routing | BiSeNet/FaRL parser (hook) |
| No spoof protection | anti-spoof disabled | 5 | photo/replay accepted | supply MiniFASNet ONNX |
| False reject | heavy occlusion drops cosine <0.45 | 6/7,10 | real person = "unknown" | occlusion-aware model (7) + fusion (9) |
| False accept | look-alike/twin >0.45 | 6/7,10 | wrong identity | raise threshold; multimodal fusion |
| ID switch | two faces cross (IoU-only) | 8 | swapped/merged tracks | ByteTrack + appearance (hook) |
| Track fragmentation | fast motion, high stride | 8 | one person → many tracks | lower stride; analytics merge |
| Blended embedding | ID switch feeds fusion | 9 | averaged wrong person | fix tracker |
| Linear search slow | N ≳ 1M users | 10 | latency + O(N) rebuild | HNSW/IVF-PQ/Milvus |
| Duplicate mis-call | dup_threshold too low/high | 10 | block real user / dup enroll | tune `OCCLUBIO_DUP_THRESHOLD` |
| Unknown mis-cluster | strangers near 0.45 | 11 | merged/split alerts | tune `unknown_merge_sim` |
| Demographic bias | occlusion+low-light skew | 6/7 | uneven TAR@FAR | per-demographic eval (mandatory) |
| Template leak | raw embeddings stored | 6/7,10,DB | permanent biometric compromise | encrypt / IronMask (hook) |

**Cascading behavior.** Errors flow *downstream only* (it's a DAG). A detector miss →
nothing to align/embed → the person just never appears. A wrong alignment → wrong embedding →
wrong/unknown match → wrong report line. The two dampeners are **track fusion** (9, averages
out single-frame noise) and **analytics merging/clustering** (11, consolidates fragmented
tracks) — neither can fix a systematic upstream error, only random noise.

---

## 7. Comprehensive Q&A

**Pipeline & topology**

**Q1. What is the end-to-end pipeline?**
Detect (SCRFD) → track (IoU) → align (Umeyama 112×112) → parallel {quality, occlusion,
anti-spoof} → gate → embed (ArcFace/custom, 512-d) → fuse over track → FAISS 1:N cosine
search → analytics report. Enrollment is the same up to embed, then average → template →
duplicate-check → DB → rebuild index.

**Q2. Which models are in series and which in parallel?**
Series: detector → tracker → aligner → recognizer → fusion → search → analytics (each needs
the previous). Parallel: quality ∥ occlusion ∥ anti-spoof (three independent reads of the
same aligned crop), and all faces in a frame are processed data-parallel.

**Q3. Which stage makes the actual identity decision?**
The FAISS search (10): `argmax` cosine over the gallery, thresholded at 0.45. The recognizer
only produces the vector; the gallery turns it into a name-or-"unknown".

**Q4. What feeds each model and what consumes its output?**
See §1.4 (per-edge table) and every module's point 4. Example: the recognizer is fed the
aligned crop (from 2, gated by 3∧5) and its output is consumed by fusion (9) then search (10).

**Q5. Is the FAISS index the source of truth?**
No. The **SQLite/Postgres DB is the source of truth**; FAISS is a derived matrix rebuilt from
`Enrollment` rows on every change. You can swap FAISS→Milvus without touching business logic.

**Classification vs. regression**

**Q6. Is the detector classification or regression?**
Both — per-anchor face/background **classification** plus bbox and landmark **regression**
(multi-task).

**Q7. Is the face recognizer classification or regression?**
**Neither at inference** — it is a **metric-embedding** model that outputs a 512-d vector. It
is *trained* with a classification-style margin softmax (ArcFace/AdaFace), but that head is
thrown away for deployment. No class, no probability comes out of it.

**Q8. Which modules are classifiers?**
Anti-spoof (binary, softmax), occlusion estimator (rule-based multi-class), and the
training-only AdaFace/ArcFace heads. The detector's score branch is also classification.

**Q9. Which modules are regression-like?**
The quality gate (continuous [0,1] score), the detector's box/landmark branches, and the
occlusion `occluded_ratio`.

**Q10. Which modules learn nothing (deterministic/heuristic)?**
Aligner, quality gate, occlusion estimator, IoU tracker, fusion, FAISS index, analytics —
all parameter-free. Only the detector, recognizer(s), anti-spoof, and training backbone/heads
have learned weights.

**What each model computes / probabilities**

**Q11. Does any stage output a real probability?**
Yes — **anti-spoof** applies softmax and emits a live-probability in [0,1]. The detector's
confidence is a sigmoid-style score. The recognizer/gallery output a **cosine similarity**
in [-1,1] — a *similarity*, not a calibrated probability, but it is the number used as
"confidence."

**Q12. What exact activation/normalization does the recognizer end on?**
A linear embedding layer followed by **L2-normalization** (unit hypersphere). No softmax, no
sigmoid at inference.

**Q13. What does the quality score formula compute?**
`0.5·det + 0.3·sharpness + 0.2·brightness`, where sharpness = clip(varLaplacian/120,0,1) and
brightness is a piecewise band score; pass if ≥ 0.35.

**Q14. What "confidence" appears in the report?**
The **cosine similarity** between the (fused) query embedding and the matched gallery vector
(max and average per track). Also detector/quality scores internally.

**Data & storage**

**Q15. How is a face stored — matrix or vector?**
Each face/person is a **dense 512-d float32 vector** (unit norm). The gallery of N people is a
**dense [N,512] matrix**. DB stores each as a 2048-byte float32 blob. Nothing is sparse.

**Q16. What are the exact tensor shapes and dtypes?**
Frame `[H,W,3] uint8`; detector input `[1,3,640,640] f32`; aligned crop `[112,112,3] uint8`;
recognizer input `[1,3,112,112] f32`; embedding `[512] f32`; gallery `[N,512] f32`; anti-spoof
input `[1,3,80,80] f32`.

**Q17. Why 512 dimensions?**
It is the ArcFace/insightface `w600k_r50` output size and the configured `embedding_dim`. The
architecture doc notes 256-d is a valid storage/bandwidth optimization via a distilled head.

**Q18. Why L2-normalize everything?**
So inner product equals cosine similarity, making search metric-consistent and thresholds
stable, and so magnitude (lighting/contrast) doesn't distort identity distance.

**Search & matching**

**Q19. How does search work inside the matrix?**
FAISS computes `G · qᵀ` (dot products of the query against all N rows) and returns the top-k;
`identify` takes top-1 and thresholds at 0.45.

**Q20. What distance metric is used?**
**Cosine similarity** implemented as **inner product** on L2-normed vectors
(`IndexFlatIP`) — exact, brute-force. Equivalent to Euclidean here because vectors are unit
norm.

**Q21. Is it exact or approximate nearest neighbor?**
**Exact** today (`IndexFlatIP`, full scan). HNSW/IVF-PQ (approximate ANN) are the documented
scale-up, with an exact re-rank to recover accuracy.

**Q22. How does duplicate detection work at enrollment?**
Same cosine search of the new template against the gallery; if top-1 ≥ `dup_threshold 0.5`
and it's a *different* user, enrollment is rejected `409 duplicate_face`.

**Q23. How are unknown people grouped?**
Analytics clusters unknown tracks by cosine similarity of their best embeddings (≥ 0.45) into
`UNKNOWN-1, UNKNOWN-2, …`, then filters blips < 0.6 s.

**Outputs / boundary data**

**Q24. Is each stage's output a number, an image, or a structure?**
Detector → coordinate numbers; aligner → a 112×112 image; quality → a number+bool; occlusion
→ a small dict; anti-spoof → a bool+prob; recognizer → 512 numbers; search → (label, score);
analytics → JSON. See §1.4.

**Q25. What is the final system output?**
A **JSON report** (people with appearances/first-last-seen/durations/confidences, unknown
alerts, optional target match, processing stats) **plus an annotated MP4** with boxes, names,
track IDs, and occlusion tags; the service also writes `Sighting`/`Alert` DB rows.

**Q26. What does a "target identification" produce?**
`target_report`: found (bool), appearance count, max confidence, first/last seen, total
visible seconds, and time-windows — by matching every track's best embedding to the target
embedding (cosine ≥ threshold).

**Training & data provenance**

**Q27. What data is the detector trained on?**
WIDER FACE (InsightFace pretrained).

**Q28. What data is the default recognizer trained on?**
A cleaned large-scale web face corpus ("W600K", WebFace/Glint family) with ArcFace loss —
InsightFace's `w600k_r50`.

**Q29. What data would *your* custom recognizer train on?**
Your own aligned 112×112 identity folders, aggressively augmented with synthetic occlusion +
photometric corruption (mask/sunglasses/cap/scarf/erase + blur/low-light/JPEG/downscale).

**Q30. What input geometry do the recognizers expect?**
112×112 aligned to the ArcFace 5-point canonical layout, normalized to ≈[-1,1]. Alignment
must be identical between training and inference.

**Q31. What loss trains the custom model and why AdaFace?**
AdaFace (default) — a quality-adaptive angular-margin softmax that down-weights unidentifiable
(occluded/blurry) samples, which is empirically best for this project's low-quality domain;
ArcFace (fixed margin) is the simpler alternative.

**Scalability**

**Q32. What is the Big-O of 1:N search?**
`O(N·d)` time and `O(N·d)` memory for exact flat search (N users, d=512). HNSW ≈ `O(log N)`
per query; IVF-PQ sub-linear with compression.

**Q33. What is the Big-O of the recognizer?**
`O(1)` per face (fixed 112²) → `O(F)` per frame for F gated faces; independent of gallery
size.

**Q34. What is the Big-O of training the margin head?**
`O(B·num_classes·d)` per step — grows with the number of identities (partial-FC is the
industrial mitigation, not implemented).

**Q35. How many cameras/videos concurrently?**
The engine processes **one video at a time per process** (a service-wide lock). Scale out with
workers (Celery/RQ) and GPU serving; real-time multi-camera is the Jetson/DeepStream edge
path (a separate deployment).

**Q36. Where is the main bottleneck?**
CPU embedding (~30–100 ms/face) → mitigated with `--stride` and GPU/TensorRT. FAISS becomes
the bottleneck only past ~1M users.

**Failure / limits**

**Q37. When does the detector fail?**
Tiny faces (<24 px), confidence <0.5, extreme profile, full helmet, severe blur/low light.

**Q38. When does recognition fail?**
Mis-alignment, heavy multi-occlusion (mask+sunglasses) pushing genuine cosine below 0.45
(false reject), look-alikes/twins above threshold (false accept), and surveillance-domain
distribution shift.

**Q39. When does the tracker fail?**
Crossing/overlapping people (ID switches, it's appearance-blind), fast motion or high stride
(fragmentation), occlusion longer than `max_age(30)` frames.

**Q40. When does search fail or degrade?**
Threshold mis-set (false accept/reject trade-off), gallery in the millions (linear-scan
latency + O(N) rebuild), index out-of-sync with DB after a crash.

**Q41. What are the security/privacy failure modes?**
Anti-spoof disabled by default (photo/replay accepted); raw embeddings stored unprotected
(partially invertible → permanent biometric compromise) — encryption/IronMask is a hook.

**Q42. What are the endpoints (API) of the system?**
`POST /api/register`, `POST /api/users/{id}/enroll`, `POST /api/identify` (video job),
`GET /api/jobs/{id}` (status+report), `GET /api/jobs/{id}/video` (annotated MP4), plus
login/dashboard routes. These are the *service* endpoints; each model's "endpoint" is its
output edge in §1.4.

**Q43. How are cascading errors handled?**
They propagate downstream only (DAG). Random per-frame noise is damped by track fusion (9)
and analytics merging/clustering (11); systematic upstream errors (bad alignment, detector
miss) are *not* recoverable downstream — the person simply mis-reports or disappears.

**Q44. What's the difference between the two operating flows?**
Enrollment builds a stored template (average of accepted embeddings, duplicate-guarded);
identification searches live embeddings against that gallery and produces a timeline report.

**Q45. Why keep a pretrained baseline?**
Per the README's golden rule: always benchmark your custom occlusion-aware model against the
working pretrained ArcFace pipeline so you never ship a regression.

---

## 8. Glossary

- **Embedding** — a fixed-length vector (here 512-d) representing a face; distance ≈ identity
  difference.
- **L2-normalization** — scaling a vector to unit length so cosine = dot product.
- **Cosine similarity** — `a·b/(‖a‖‖b‖)`; here just `a·b` (unit vectors); ∈[-1,1].
- **ArcFace / AdaFace** — angular-margin softmax losses that make embeddings discriminative;
  AdaFace adapts the margin to sample quality.
- **FIQA** — Face Image Quality Assessment (here a hand-crafted proxy).
- **PAD / anti-spoof** — Presentation-Attack Detection (photo/replay/mask).
- **IoU** — Intersection-over-Union of two boxes; the tracker's association metric.
- **NMS** — Non-Max Suppression; removes overlapping duplicate detections.
- **1:N search** — match one query against N enrolled identities.
- **HNSW / IVF-PQ** — approximate nearest-neighbor indexes for large galleries.
- **DAG** — Directed Acyclic Graph; the pipeline's data-flow shape.
- **Metric-embedding model** — outputs a vector for distance comparison, not a class label.

---

## 9. Appendix — exact knobs, thresholds, file map

**Thresholds (from `configs/default.yaml`)**

| Knob | Value | Meaning |
|---|---|---|
| detection.model_name | buffalo_l | SCRFD-10G + w600k_r50 |
| detection.det_size | 640×640 | detector input |
| detection.score_thresh | 0.5 | min face confidence |
| detection.min_face | 24 px | min face size |
| recognition.embedding_dim | 512 | vector length |
| quality.min_score | 0.35 | gate threshold |
| quality.blur_min | 40.0 | sharpness floor (÷ used: 120) |
| quality.brightness | 30–225 | acceptable mean band |
| occlusion.enabled | true | heuristic estimator on |
| antispoof.enabled | false | PAD off until weights supplied |
| antispoof.input_size | 80×80 | MiniFASNet input |
| tracking.iou_thresh | 0.25 | association overlap |
| tracking.max_age | 30 | frames a track survives |
| tracking.min_hits | 2 | frames to confirm |
| gallery.match_threshold | 0.45 | known vs unknown cosine |
| gallery.per_track_buffer | 8 | fusion window |
| analytics.min_track_frames | 3 | drop tiny tracks |
| analytics.merge_gap_s | 1.5 | merge appearance windows |
| analytics.unknown_min_seconds | 0.6 | drop unknown blips |
| analytics.unknown_merge_sim | 0.45 | cluster unknowns |
| OCCLUBIO_DUP_THRESHOLD | 0.5 | duplicate-face cosine |

**File map**

| Concern | File |
|---|---|
| Detector + embed orchestration | `occlubio/pipeline/face_analyzer.py` |
| Alignment | `occlubio/pipeline/aligner.py` |
| Quality gate | `occlubio/pipeline/quality.py` |
| Occlusion | `occlubio/pipeline/occlusion.py` |
| Anti-spoof | `occlubio/pipeline/antispoof.py` |
| Custom recognizer | `occlubio/pipeline/recognizer.py` |
| Engine (series+gate+fusion) | `occlubio/pipeline/engine.py` |
| Tracker | `occlubio/tracking/tracker.py` |
| FAISS gallery | `occlubio/gallery/faiss_gallery.py` |
| Analytics | `occlubio/analytics/identity_log.py` |
| Service (enroll/identify/dup) | `occlubio/service/face_service.py` |
| DB models | `occlubio/db/models.py` |
| Training (backbone+export) | `occlubio/training/train.py` |
| Losses (AdaFace/ArcFace) | `occlubio/training/losses.py` |
| Occlusion augmentation | `occlubio/data/occlusion_aug.py` |
| Config | `configs/default.yaml` |

---

*Generated by auditing the source tree. Numbers, shapes, thresholds, and control flow are
taken directly from the code and `configs/default.yaml`; model provenance (WIDER FACE, W600K,
ArcFace/AdaFace) follows the InsightFace model cards and the project's own
`OCCLUSION_ROBUST_FR_ARCHITECTURE.md`.*
