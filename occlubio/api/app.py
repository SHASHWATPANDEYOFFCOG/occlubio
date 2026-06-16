"""End-user face-recognition platform API (FastAPI).

Run:  uvicorn occlubio.api.app:app --host 0.0.0.0 --port 8000
Open: http://localhost:8000   (self-registration + enrollment + video identification UI)

This is an MVP: auth returns a user id (no JWT), and video jobs run in-process via
BackgroundTasks. The production hardening path (JWT, Celery/RQ workers, object storage,
Milvus, template encryption) is documented in PLATFORM.md.
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import numpy as np
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from sqlalchemy.orm import Session

from occlubio.api.schemas import (EnrollResponse, JobCreated, JobOut, LoginRequest,
                                  RegisterRequest, UserOut)
from occlubio.db import SessionLocal, get_db, init_db
from occlubio.db.models import Enrollment, Job, User
from occlubio.service.face_service import (FaceService, decode_image, hash_password,
                                          verify_password)
from occlubio.utils import ensure_dir, get_logger

log = get_logger("api")
WEB_DIR = Path(__file__).resolve().parents[2] / "web"
UPLOAD_DIR = ensure_dir("data/uploads")
OUTPUT_DIR = ensure_dir("data/outputs")

service: FaceService | None = None  # set in lifespan


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    init_db()
    service = FaceService()
    with SessionLocal() as s:
        service.startup(s)
    log.info("platform ready")
    yield


app = FastAPI(title="occlubio face-recognition platform", lifespan=lifespan)


# ---------------------------------------------------------------- UI
@app.get("/", response_class=HTMLResponse)
def index():
    f = WEB_DIR / "index.html"
    return f.read_text(encoding="utf-8") if f.exists() else "<h1>occlubio</h1><p>UI missing.</p>"


# ---------------------------------------------------------------- accounts
@app.post("/api/register", response_model=UserOut)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter((User.username == req.username) | (User.email == req.email)).first():
        raise HTTPException(409, "username or email already taken")
    user = User(username=req.username, email=req.email, password_hash=hash_password(req.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return UserOut(id=user.id, username=user.username, email=user.email, enrolled=False)


@app.post("/api/login")
def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == req.username).first()
    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(401, "invalid credentials")
    return {"user_id": user.id, "username": user.username, "enrolled": user.enrollment is not None}


@app.get("/api/users", response_model=List[UserOut])
def list_users(db: Session = Depends(get_db)):
    return [UserOut(id=u.id, username=u.username, email=u.email, enrolled=u.enrollment is not None)
            for u in db.query(User).order_by(User.id).all()]


# ---------------------------------------------------------------- enrollment
@app.post("/api/users/{user_id}/enroll", response_model=EnrollResponse)
async def enroll(user_id: int, files: List[UploadFile] = File(...), force: bool = False,
                 db: Session = Depends(get_db)):
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(404, "user not found")

    images = [decode_image(await f.read()) for f in files]
    try:
        template, n_acc, n_total, quality = service.embed_images(images)
    except ValueError as e:
        raise HTTPException(422, str(e))

    dup = service.find_duplicate(template, exclude_username=user.username)
    if dup and not force:
        raise HTTPException(409, detail={"error": "duplicate_face", "matches": dup})

    enr = user.enrollment or Enrollment(user_id=user.id)
    enr.embedding = template.astype(np.float32).tobytes()
    enr.dim = int(template.shape[0])
    enr.n_images = n_acc
    enr.quality = quality
    db.add(enr)
    db.commit()
    service.rebuild_index(db)
    return EnrollResponse(user_id=user.id, username=user.username, n_accepted=n_acc,
                          n_total=n_total, quality=round(quality, 3), duplicate=dup)


# ---------------------------------------------------------------- identification
def _run_identify(job_id: int, video_path: str, stride: int):
    with SessionLocal() as s:
        job = s.get(Job, job_id)
        job.status = "running"
        s.commit()
        try:
            out = f"{OUTPUT_DIR}/job_{job_id}.mp4"
            rep = f"{OUTPUT_DIR}/job_{job_id}.json"
            service.identify_video(video_path, out, rep, session=s, stride=stride)
            job.output_video, job.report_path, job.status = out, rep, "done"
            s.commit()
        except Exception as e:  # noqa: BLE001
            job.status, job.message = "error", str(e)
            s.commit()
            log.exception("identify job %d failed", job_id)


@app.post("/api/identify", response_model=JobCreated)
async def identify(background: BackgroundTasks, file: UploadFile = File(...),
                   stride: int = Form(2), db: Session = Depends(get_db)):
    job = Job(kind="identify", status="pending")
    db.add(job)
    db.commit()
    db.refresh(job)
    dest = f"{UPLOAD_DIR}/job_{job.id}_{file.filename}"
    with open(dest, "wb") as out:
        out.write(await file.read())
    job.input_path = dest
    db.commit()
    background.add_task(_run_identify, job.id, dest, stride)
    return JobCreated(job_id=job.id, status="pending")


@app.get("/api/jobs/{job_id}", response_model=JobOut)
def job_status(job_id: int, db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(404, "job not found")
    report = None
    if job.report_path and Path(job.report_path).exists():
        report = json.loads(Path(job.report_path).read_text(encoding="utf-8"))
    return JobOut(id=job.id, status=job.status, message=job.message, report=report,
                  output_video=(f"/api/jobs/{job.id}/video" if job.output_video else None))


@app.get("/api/jobs/{job_id}/video")
def job_video(job_id: int, db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if not job or not job.output_video or not Path(job.output_video).exists():
        raise HTTPException(404, "video not ready")
    # filename=... sets Content-Disposition: attachment so the browser DOWNLOADS the file
    # (mp4v can't be previewed inline in most browsers — open the download in VLC/any player).
    return FileResponse(
        str(Path(job.output_video).resolve()),
        media_type="video/mp4",
        filename=f"identification_job_{job_id}.mp4",
    )
