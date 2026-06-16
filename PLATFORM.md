# occlubio Platform — End-User Face Recognition (registration → video identification)

The **application layer** on top of the recognition engine. Users self-register, enroll their
face, upload a video, and get back identified people with timestamps + an annotated video.

- Per-model research (detection/recognition/tracking/FIQA, papers, benchmarks, TensorRT/ONNX,
  Jetson/DeepStream/Triton): **[OCCLUSION_ROBUST_FR_ARCHITECTURE.md](OCCLUSION_ROBUST_FR_ARCHITECTURE.md)**.
- This doc covers the **platform**: API, database, workflows, scaling, deployment, roadmap.

---

## 1. System architecture

```
 Browser (web/index.html)
   │  register / login / enroll(images, webcam) / upload video / poll job
   ▼
 FastAPI  (occlubio/api/app.py)            ← REST API, request validation, background jobs
   │
   ▼
 FaceService (occlubio/service/face_service.py)   ← orchestration + business logic
   │            ├─ password hashing (PBKDF2)
   │            ├─ quality-gated enrollment + duplicate-face detection
   │            └─ video identification (engine + analytics)
   ▼                         ▼
 RecognitionEngine        Database (source of truth)
 (occlubio/pipeline)      ├─ SQLite/Postgres  (users, enrollments, jobs)  ← occlubio/db
   detect→track→align→    └─ FAISS index (DERIVED, rebuilt from DB)        ← occlubio/gallery
   quality→embed→search
```

**Design rule:** the relational DB is the **source of truth**; the FAISS vector index is a
*derived* structure rebuilt from the DB. This means you can swap FAISS → Milvus/Qdrant, or
SQLite → Postgres, without touching business logic.

---

## 2. Platform-layer component choices (the "new" research vs the model docs)

| Concern | MVP pick (in repo) | Production / scale | Why |
|---|---|---|---|
| Web API | **FastAPI + Uvicorn** | + Gunicorn workers, Nginx | async, auto OpenAPI docs at `/docs`, huge ecosystem |
| Frontend | single `index.html` (vanilla JS) | React/Next.js | zero build step for MVP |
| Auth | PBKDF2 password + user-id | **JWT (fastapi-users / Authlib)**, OAuth | stdlib now; tokens + RBAC later |
| Relational DB | **SQLite** | **PostgreSQL** (`+pgvector` optional) | set `OCCLUBIO_DB` env, no code change |
| Vector search | **FAISS FlatIP** (exact) | **Milvus** / **Qdrant** / FAISS IVF-PQ-GPU | exact ≤1M; ANN + sharding beyond |
| Embedding store | float32 blob in DB | **encrypted** blob / protected template (IronMask) | biometrics can't be reissued — encrypt |
| Job processing | FastAPI **BackgroundTasks** | **Celery / RQ + Redis**, GPU workers | decouple slow video jobs, scale workers |
| File storage | local `data/uploads`,`data/outputs` | **S3 / MinIO** | durable, multi-node |
| Realtime/edge | offline video jobs | **DeepStream + Triton** on Jetson | multi-camera, TensorRT FP16/INT8 |

GitHub refs: FastAPI `tiangolo/fastapi` · Milvus `milvus-io/milvus` · Qdrant `qdrant/qdrant`
· FAISS `facebookresearch/faiss` · Celery `celery/celery` · fastapi-users `fastapi-users/fastapi-users`.
The face models themselves are unchanged from the architecture doc (SCRFD, ArcFace/EdgeFace+AdaFace, CR-FIQA, …).

---

## 3. User registration workflow

```
POST /api/register {username,email,password}      -> creates User (PBKDF2 hash), returns id
POST /api/users/{id}/enroll  (multipart images)   -> for each image:
       detect (SCRFD) -> align -> FIQA quality gate -> embed
   accepted samples averaged -> 512-d template
   DUPLICATE CHECK: search index; if matches a different user >= dup_threshold -> 409
   upsert Enrollment(embedding bytes, quality) -> rebuild FAISS index
```
Enrollment accepts **uploaded photos and/or live webcam captures** (the UI snapshots frames to
JPEG and posts them as images). Low-quality samples are skipped; if none pass, enrollment is
rejected with the reason (override with `?force=true`).

**Verified:** alice enrolls (quality 0.92); bob trying to enroll alice's face is rejected
`409 duplicate_face (matches alice, score 1.0)`; bob with a different face succeeds.

---

## 4. Face database design

| Table | Key columns | Notes |
|---|---|---|
| `users` | id, username (unique), email (unique), password_hash, created_at | account |
| `enrollments` | user_id (unique FK), **embedding (LargeBinary)**, dim, n_images, quality | one template/user; encrypt `embedding` in prod |
| `jobs` | id, kind, status, input_path, output_video, report_path, message | async video jobs |

FAISS index: `IndexFlatIP` (cosine on L2-normalized vectors), **labelled by username**, rebuilt
from `enrollments` on every change. Swap to Milvus by replacing `FaissGallery` (same add/search API).

---

## 5. Video identification workflow

```
POST /api/identify (multipart video, stride)  -> creates Job(pending), saves upload, returns job_id
   BackgroundTask: per processed frame -> detect -> track -> embed -> 1:N match
       IdentityLog aggregates per track: majority-vote identity, best/avg confidence,
       first/last-seen timestamp, appearance windows
GET /api/jobs/{id}          -> status + JSON report (when done)
GET /api/jobs/{id}/video    -> annotated MP4 (boxes, names, track IDs, confidence)
```
Report fields per person: **username, user_id, appearances, first_seen, last_seen,
total_visible_s, max/avg confidence, appearance_windows**; plus **unknown-person alerts**.

---

## 6. Run it

```powershell
.venv\Scripts\Activate.ps1
pip install -e ".[api]"
uvicorn occlubio.api.app:app --host 0.0.0.0 --port 8000
# open http://localhost:8000   (UI)   and  http://localhost:8000/docs  (API explorer)
```
Env knobs: `OCCLUBIO_DB` (DB URL), `OCCLUBIO_DUP_THRESHOLD` (duplicate-face cosine, default 0.5).
Recognition/quality thresholds live in [configs/default.yaml](configs/default.yaml).

---

## 7. Performance & scalability

| Metric | This machine (CPU) | Jetson Orin NX (TRT FP16) | Server + GPU |
|---|---|---|---|
| Per-face recognition | ~30–100 ms | ~8–15 ms | ~3–6 ms |
| Video throughput | ~0.4 FPS (use `--stride`) | real-time, 4–8 cams | many streams |
| 1:N search latency | <1 ms (≤1M, FAISS flat) | <1 ms | <1 ms (IVF-PQ, billions) |

**Scaling 100 → 1M users:**
1. **≤10k:** SQLite + FAISS FlatIP (current). Exact, trivial.
2. **10k–1M:** Postgres + FAISS **IVF-PQ** (GPU) or Qdrant; add a top-k **exact re-rank**.
3. **>1M / multi-node:** **Milvus** (sharded, GPU via cuVS) + Postgres + S3/MinIO + Celery GPU workers + load-balanced API.
Throughput scales by adding **Celery workers** (CPU/GPU) behind the API; the DB/vector store scale independently.

---

## 8. Deployment strategy (MVP → production)

1. **Local MVP** (now): `uvicorn`, SQLite, FAISS, in-process jobs.
2. **Containerize:** Docker image for API; `docker-compose` adds Postgres + Redis + MinIO + a worker.
3. **GPU inference:** install `onnxruntime-gpu` + `faiss-gpu`; or serve models via **Triton** and have the worker call it.
4. **Edge / realtime:** port detection+recognition to **TensorRT engines** in a **DeepStream** pipeline on Jetson for live multi-camera (see `deepstream/`); the platform DB/API stays the same, fed by the edge node.

---

## 9. Beginner roadmap (MVP → production)

- **M1 (done):** accounts, quality-gated enrollment, duplicate-face detection, video identify, report + annotated video, web UI. ✅
- **M2:** JWT auth + per-user job history; Dockerize; move to Postgres.
- **M3:** Celery + Redis workers (so video jobs don't block); S3/MinIO for uploads/outputs.
- **M4:** swap recognizer to your **occlusion-aware EdgeFace+AdaFace** model (Phase 3); add anti-spoofing at enroll & identify.
- **M5:** Milvus for scale; **encrypt/protect templates**; GPU serving (Triton) or Jetson/DeepStream edge nodes.
- **M6:** monitoring, per-demographic bias eval, DPIA + legal review (EU AI Act / India DPDP Act).

---

## 10. Security & privacy (mandatory before real users)
- Encrypt the `embedding` column at rest; prefer **protected/cancelable templates** (architecture §1.12) — a leaked raw embedding is a permanent compromise.
- Add **face anti-spoofing** at enrollment and identification so photos/replays can't register or be identified.
- HTTPS, rate-limiting, real auth (JWT), audit logs, data-deletion ("right to erasure").
- Biometric data is legally sensitive (EU AI Act, India DPDP Act 2023) — get consent + a DPIA before deployment.
