# occlubio - Scalability Report

**System:** occlubio occlusion-robust face-recognition platform (FastAPI + SQLite + FAISS + InsightFace)
**Prepared:** 2026-07-06
**Scope:** Capacity, concurrency, and bottleneck analysis of the current single-machine deployment, with a scaling roadmap.

---

## Method

All figures marked *measured* were captured on the running instance on this machine
(single Uvicorn worker, CPU / `CPUExecutionProvider`, SQLite, ~10 users in the DB).
Load was generated with 200 requests per concurrency level against `GET /api/users`.
Figures marked *(est.)* are engineering estimates, not measured on this box.

---

## Executive summary

occlubio today is a **single-process, single-machine MVP**. It behaves in two very different regimes:

- **As a login / dashboard application** it comfortably serves roughly **50-150 concurrent active
  browsers** and **hundreds to low-thousands of registered users**, degrading *gracefully*
  (slower responses, not failure) beyond that point.
- **As a video-identification service** - its actual purpose - it processes **exactly one clip at a
  time, taking minutes per clip on CPU**. This is the binding limit, and it does **not** improve by
  adding traffic or by naively adding worker processes.

The clearest evidence is the throughput curve: raising request concurrency from 1 to 50 **did not
raise throughput** (it stayed near 47 req/s) and instead raised latency ~40x. That is the
signature of a serialized system.

---

## Measured capacity (single worker, CPU)

| Workload | Measurement | Practical ceiling |
|---|---|---|
| Light auth read (`/api/users`) | ~47 req/s, flat from concurrency 1->50; latency 25 ms -> 1,033 ms | ~100 concurrent browsers at p95 < 1 s |
| Login / register (PBKDF2 200k) | ~330 ms per operation | ~3/s per core; ~15-25/s on a 6-8 core box |
| Enroll a face | serialized by model lock; ~0.3-1 s + full index rebuild | 1 at a time |
| Identify a video | 1.84 processed-frames/s (12 frames in 6.5 s) | 1 concurrent job; ~3-8 min per 60 s clip |
| 1:N gallery search | exact `IndexFlatIP`, O(N x 512) | fine to ~10^4-10^5 identities *(est.)* |

Concurrency vs latency, `GET /api/users`, 200 requests per level (*measured*):

| Concurrency | Throughput | Median latency | p95 latency |
|---|---|---|---|
| 1 | 35.2 req/s | 24 ms | 47 ms |
| 4 | 40.0 req/s | 80 ms | 116 ms |
| 10 | 47.6 req/s | 214 ms | 256 ms |
| 25 | 47.4 req/s | 511 ms | 638 ms |
| 50 | 45.7 req/s | 1,033 ms | 1,622 ms |

A 60 s, 30 fps clip takes roughly **8 minutes at stride 2** or **3 minutes at stride 5**, and every
other enroll or identify request **waits behind it** on the same lock.

---

## Binding constraints, ranked

**1. Global inference lock - all heavy work is serialized (the primary ceiling).**
`FaceService._lock` (a re-entrant lock) wraps startup, embedding, duplicate-check, and the entire
`identify_video` frame loop, because the ONNX / InsightFace session is not thread-safe. Result:
**at most one concurrent enroll-or-identify operation**. On CPU that is ~1.8 frames/s of video.
This is by far the hardest limit for the surveillance workload.

**2. In-process FAISS index - the app is effectively single-process for correctness, not just speed.**
The gallery is an in-RAM exact index, rebuilt from the database on **every** enroll or removal.
If the server is run with multiple workers, **each worker holds its own copy of the index**; an
enrollment handled by worker 1 is invisible to worker 2 until that worker independently rebuilds.
So the current design cannot be horizontally scaled by adding web workers without producing
inconsistent enrollment and identification results. This is the single most important
architectural finding.

**3. Video jobs run inside the web process.**
Identification runs via FastAPI background tasks in the same process as the API. A multi-minute,
CPU-bound job saturates the machine and consumes a thread-pool thread that would otherwise serve
API requests, so the whole console becomes sluggish while a job runs.

**4. Password hashing cost (~330 ms per login).**
PBKDF2 with 200k iterations is a deliberate security cost, but it makes a login burst (for example,
a cohort signing in at the same time) the first user-visible stall. PBKDF2 releases the interpreter
lock, so it does scale across CPU cores, which softens this on multi-core hardware.

**5. SQLite without WAL - a single database-wide writer.**
The default rollback journal is in effect (no `journal_mode=WAL`). Concurrent registrations,
enrollments, and messages serialize and can raise "database is locked" under write bursts. Reads
are unaffected.

**6. Uploads are read fully into memory.**
Video and image uploads are read entirely into RAM before being written to disk. A 500 MB clip is a
~500 MB memory spike per upload; several concurrent large uploads risk out-of-memory.

**7. Minor items.** An N+1 lazy-load in the user listing (one query per user), web pages re-read from
disk on each request, and no rate-limiting or backpressure.

---

## How many users can it frequently handle?

- **Registered users (data volume):** thousands are fine. Gate-crossing history is bounded by design
  (one row per user, per camera, per day), and the exact index handles ~10^4-10^5 identities before
  rebuild cost and memory become concerns.
- **Concurrent active browsers (dashboard use):** ~50-100 comfortable, ~150 with visible latency,
  then graceful queuing. The system slows; it does not crash.
- **Concurrent enroll / identify operations:** one. This is the real product ceiling.
- **Continuous multi-camera surveillance:** not supported on this deployment - it is a batch,
  one-clip-at-a-time tool on CPU.

---

## What breaks first as load rises

1. Someone submits a long video, and the console is sluggish for minutes (job plus lock).
2. A login burst raises authentication latency (password hashing).
3. Two people enroll or identify at once, and the second waits (global lock).
4. Sustained write bursts produce occasional "database is locked".
5. A large upload (or several) causes a memory spike and possible out-of-memory.

---

## Scaling roadmap

**Quick wins (hours, same machine; roughly 2-5x on light traffic and safer writes)**

- Enable SQLite WAL (`journal_mode=WAL`) and a busy timeout: concurrent reads during writes and
  far fewer lock errors.
- Fix the N+1 in the user listing with eager loading of the enrollment relationship.
- Stream uploads to disk instead of reading whole files into memory (removes the RAM spikes).
- Add an upload size cap and basic rate limiting / backpressure.

**Medium (the real fix)**

- Move video jobs to a task queue (Celery or RQ with Redis) and dedicated workers, off the web
  process, so identification no longer blocks the API and scales by adding workers.
- Run one model instance per worker on GPU (`onnxruntime-gpu`): identification throughput moves from
  ~1.8 fps to an estimated 50-200+ fps per GPU; N GPU workers give N concurrent streams.
- Replace the in-process index with a shared vector store (Milvus or Qdrant): this removes the
  rebuild-on-every-enroll and unlocks true horizontal scaling, since all workers share one gallery.
- Move from SQLite to Postgres for real write concurrency, connection pooling, and replication.
- Use object storage (S3 or MinIO) for uploads and outputs; put Nginx and multiple Gunicorn workers
  in front.

**Production / multi-node**

- Sharded Milvus (optionally GPU-accelerated), Postgres, object storage, and an autoscaling GPU
  worker pool behind a load balancer. Throughput then scales approximately linearly with workers,
  and the database and vector store scale independently. Realistic targets at this tier are
  thousands of API requests/s and dozens of concurrent video streams *(est.)*.

This path matches the platform's own documented milestones (M1-M3) in PLATFORM.md.

---

## Caveats

- Numbers are order-of-magnitude, measured on one development machine with a small dataset; the
  absolute requests/s is endpoint-specific, but the shape (flat throughput, linearly rising latency)
  generalizes to the single-worker deployment.
- GPU and worker figures are estimates, not measured here (no GPU on this machine).
- The architecture is appropriately scoped as an MVP. This report is a map of where the ceilings are
  and which one to lift first, not a criticism of the design.
