# One Face Among Half a Million: Engineering and Benchmarking 1:N Face Recognition

Most face-recognition explainers stop at "the model says it's you." That's the easy
case — 1:1 verification, where a probe face is compared against a single claimed
identity. The harder, more interesting problem is **1:N identification**: a new face
arrives with no claim attached, and the system must search *all* N registered people
to answer "who is this — if anyone?" This post walks through how we built and
benchmarked a 1:N system, and what actually happens to latency, memory, and accuracy
as the gallery grows from 500 identities to 500,000 vectors.

## Why 1:N is harder than 1:1 — and why scale compounds it

A 1:1 verification is one distance computation against one enrolled template. Its
cost and error rate are constant no matter how many users you have.

1:N breaks both properties at once:

**Search cost grows with N.** Every probe must be compared — directly or through an
index — against the whole gallery. Exact search is O(N·d) per query in gallery size
N and embedding dimension d. At small N this is negligible; at large N it becomes
your latency budget.

**False matches accumulate with N.** Each enrolled identity is one more chance for a
random stranger to look "close enough." If a single comparison has false-match rate
*f*, a probe searched against N identities has roughly
1 − (1 − *f*)^N probability of at least one false match. A per-pair FMR
that is perfectly acceptable at N = 100 can make a system unusable at N = 100,000 —
*even if the model never changes*. Scale is not just an infrastructure problem; it is
an accuracy problem.

This is why a 1:N system must be measured on two axes simultaneously — systems
performance (latency, throughput, memory) *and* identification accuracy (Rank-k,
open-set error rates) — and why both must be swept across gallery size, not reported
at a single N.

## Architecture: two flows, one pipeline

The system has two entry points that share one face pipeline:

```
Enrollment:    photo → detect → align → embed → store (vector index + metadata DB)
Recognition:   probe → detect → align → embed → 1:N search → threshold → identity / "unknown"
```

[Figure: architecture diagram — enrollment and recognition sharing the
detection → alignment → embedding pipeline]

- **Detection** finds faces and 5 facial landmarks (eyes, nose, mouth corners) in the
  raw image (we use SCRFD; RetinaFace/MTCNN are equivalent choices).
- **Alignment** warps the face onto a canonical 112×112 geometry using a similarity
  transform fit to those landmarks. This matters more than it looks: the embedding
  model was *trained* on that exact geometry, and a misaligned crop silently degrades
  every downstream similarity score.
- **Embedding** maps the aligned crop to a 512-dimensional vector (ArcFace via
  InsightFace `buffalo_l`).
- **Search** compares that vector against the gallery in a FAISS index.

Enrollment and recognition literally call the same three stages — the only
difference is what happens to the embedding afterward: stored with a user ID, or
used as a query.

## Face embeddings in one paragraph

An embedding model is not a classifier: it outputs a point on a 512-dimensional unit
sphere, trained (with an angular-margin loss like ArcFace) so that images of the
same person land close together and different people land far apart. Because every
vector is L2-normalized, cosine similarity reduces to a plain dot product, and
cosine, dot product, and Euclidean distance become monotonically equivalent — so we
can use FAISS's inner-product indexes directly. The dimension is a capacity/cost
knob: 512 floats is 2 KB per identity, enough capacity to separate hundreds of
thousands of identities, while keeping a 1M-identity gallery at ~2 GB of RAM.

```python
from insightface.app import FaceAnalysis
import numpy as np

app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"])
app.prepare(ctx_id=0, det_size=(320, 320))

faces = app.get(image_bgr)                      # detect + align + embed
emb = faces[0].embedding.astype(np.float32)
emb /= np.linalg.norm(emb)                      # unit norm → cosine == dot product
```

## The index: exact vs approximate

**IndexFlatIP (exact).** A brute-force scan of the full [N × 512] matrix. It is
simple, exact (recall = 1.0 by definition), and memory-minimal — but O(N·d) per
query. Its latency grows linearly and eventually crosses whatever SLA you set.

**IndexIVFFlat (inverted file).** K-means partitions the gallery into `nlist` cells;
a query only scans the `nprobe` nearest cells. Speedup ≈ nlist/nprobe at the cost of
recall: the true nearest neighbor is sometimes in a cell you didn't probe. `nprobe`
is a runtime dial from "fast and lossy" (nprobe=1) to "nearly exact" (nprobe≥64).

**IndexHNSWFlat (graph).** A navigable small-world graph; queries greedily walk from
an entry point toward the neighborhood of the query, giving roughly logarithmic
scaling. Recall is tuned with `efSearch`. The costs are a slower one-time build and
extra memory for the graph links (~M×8 bytes/vector on top of the vectors).

```python
import faiss

d = 512
flat = faiss.IndexFlatIP(d)                     # exact baseline
ivf  = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, 1024, faiss.METRIC_INNER_PRODUCT)
hnsw = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)  # M=32

ivf.train(gallery)                              # k-means over gallery vectors
for idx in (flat, ivf, hnsw):
    idx.add(gallery)                            # [N, 512] float32, L2-normalized

ivf.nprobe = 16
scores, ids = ivf.search(probe[None, :], k=10)  # 1:N → top-10 (cosine, row ids)
```

## Scaling methodology: padding the gallery honestly

CelebA gives us ~202K images of **10,177 real identities** — enough to measure
accuracy up to a 9K-identity gallery (1,000 identities are held out as open-set
impostors), but nowhere near the gallery sizes where latency gets interesting.

The trick is to treat the two questions separately:

- **Accuracy scaling is bounded by real identities.** Rank-1/CMC/FPIR/FNIR sweeps
  run only on genuinely enrolled faces: gallery = one reference image per identity,
  mated probes = a *different* image of enrolled identities, non-mated probes =
  identities deliberately never enrolled.
- **Latency/memory scaling is not.** Search cost depends on the number and dimension
  of vectors, not on whether they came from faces. So we pad the real gallery with
  seeded, unit-normalized random vectors up to 50K–500K and query it with *real*
  probe embeddings. This isolates the systems question at sizes the dataset alone
  can't reach.

Everything is seeded and logged — the exact split, the padding generator, warm-up
queries (discarded), per-query timing via a nanosecond clock, and embedding caching
so reruns don't redo the CNN work. One caveat we accept: random padding vectors are
spread more uniformly than real faces, so padded galleries slightly *understate*
how much ANN recall degrades under real-face crowding — the accuracy sweep on real
identities is the corrective lens.

## The metrics that matter for 1:N

- **Rank-1 / Rank-k (CMC).** Among the ranked candidates returned for a mated probe,
  how often is the true identity first (Rank-1), or in the top k (the CMC curve)?
  This is the *closed-set* view: it assumes the person is enrolled.
- **FPIR / FNIR (open-set).** Real systems face people who are *not* enrolled. FPIR:
  how often does a never-enrolled probe score above the decision threshold (a false
  alarm)? FNIR: how often is an enrolled person rejected or misidentified at that
  threshold? Reported as FNIR at a fixed FPIR (e.g. 1%), or as a DET curve.
- **Why Rank-1 alone misleads.** Rank-1 can look excellent while the system fires
  constant false alarms on strangers, because Rank-1 never asks "should we have
  matched at all?" The threshold that controls false alarms *rises* as the gallery
  grows (more identities = more chances of a high impostor score), which converts
  directly into more false rejects. Open-set metrics capture this; Rank-1 cannot.
- **Latency percentiles and throughput.** p50 is the typical case; p95/p99 are what
  your SLA actually feels like. Single-query latency and batched throughput diverge
  wildly on vectorized search (batching amortizes memory traffic), so both are
  measured.

## Results

[Figure: search latency (p50/p95/p99) vs gallery size, log x-axis — exact grows
linearly; IVF and HNSW stay near-flat]

[Figure: throughput (QPS) vs gallery size, single-query and batched]

[Figure: memory — index size and peak RSS vs gallery size]

[Figure: Rank-1 vs number of real enrolled identities, with CMC curves at several N]

[Figure: open-set DET curve — FNIR vs FPIR by gallery size]

[Figure: Pareto plot — p95 latency vs recall@1 for FlatIP, IVF (nprobe sweep),
HNSW (efSearch sweep)]

The shape of the story is consistent with the theory: exact search cost tracks
O(N·d) almost perfectly once the gallery matrix outgrows cache; IVF trades recall
for speed along its nprobe dial; HNSW holds near-unity recall at millisecond-scale
latency but pays in build time and memory; and closed-set accuracy erodes slowly
with N while the open-set threshold climbs. (Exact figures, tables, and the
hardware/software manifest live in the benchmark report generated by the harness.)

## Takeaways

1. **Below ~50K vectors, don't bother with ANN.** Exact FlatIP is already
   sub-millisecond-to-few-ms there, is recall-perfect, and has zero tuning surface.
   Complexity buys nothing.
2. **Pick the index by which constraint binds.** Latency-bound at large N → HNSW
   (accept the memory and build cost, tune efSearch to your recall target).
   Memory-bound → IVF (or IVF-PQ beyond that), tune nprobe. Correctness-audit-bound
   → keep a Flat index on the side as ground truth, as this benchmark does.
3. **Find your maximum gallery size empirically, not by vibes.** Sweep N, plot p95
   against your SLA and Rank-1/FNIR against your accuracy floor, and take the
   largest N that satisfies *both*. The two curves bind at very different N — on our
   CPU-only test box, exact-search latency violated the 50 ms SLA long before
   closed-set accuracy fell meaningfully.
4. **Budget accuracy for the gallery you'll have, not the one you demo.** The same
   model that looks flawless at 1,000 identities will fire more false alarms at
   100,000 — plan thresholds (and expectations) against measured open-set curves at
   your target scale.
5. **The CNN, not the search, dominates end-to-end latency until N is large.**
   Detection + embedding costs tens of milliseconds per probe on CPU; search costs
   fractions of a millisecond until deep into six-figure galleries. Optimize the
   embedding path (batching, GPU, quantization) before micro-tuning the index.

*The benchmark harness (config-driven, seeded, one command end-to-end) and the full
report with hardware specs and raw results ship alongside this post.*
