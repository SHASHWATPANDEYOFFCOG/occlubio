"""End-user face-recognition platform API (FastAPI).

Run:  uvicorn occlubio.api.app:app --host 0.0.0.0 --port 8000
Open: http://localhost:8000   (self-registration + enrollment + video identification UI)

This is an MVP: auth returns a user id (no JWT), and video jobs run in-process via
BackgroundTasks. The production hardening path (JWT, Celery/RQ workers, object storage,
Milvus, template encryption) is documented in PLATFORM.md.
"""
from __future__ import annotations

import json
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import (BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException,
                     UploadFile)
from fastapi.responses import FileResponse, HTMLResponse
from sqlalchemy.orm import Session as DBSession

from occlubio.api.schemas import (AuthResponse, EnrollResponse, JobCreated, JobOut, LoginRequest,
                                  MessageCreate, MessageOut, RegisterRequest, UserOut)
from occlubio.db import SessionLocal, get_db, init_db
from occlubio.db.models import Enrollment, Job, Message, Session, User
from occlubio.service.face_service import (FaceService, decode_image, hash_password,
                                          verify_password)
from occlubio.utils import ensure_dir, get_logger

log = get_logger("api")
WEB_DIR = Path(__file__).resolve().parents[2] / "web"
UPLOAD_DIR = ensure_dir("data/uploads")
OUTPUT_DIR = ensure_dir("data/outputs")

# Shared secret that lets someone self-register as an "authority" (surveillance operator).
# Override in production; a real deployment would invite/provision operators instead.
AUTHORITY_CODE = os.environ.get("OCCLUBIO_AUTHORITY_CODE", "occlubio-authority")

service: FaceService | None = None  # set in lifespan


# ---------------------------------------------------------------- auth helpers
def _issue_token(db: DBSession, user: User) -> str:
    token = secrets.token_urlsafe(32)
    db.add(Session(token=token, user_id=user.id))
    db.commit()
    return token


def current_user(authorization: Optional[str] = Header(None),
                 db: DBSession = Depends(get_db)) -> User:
    """Resolve the bearer token to a User, or 401."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    sess = db.get(Session, token)
    if not sess:
        raise HTTPException(401, "invalid or expired session")
    user = db.get(User, sess.user_id)
    if not user:
        raise HTTPException(401, "invalid session")
    return user


def require_authority(user: User = Depends(current_user)) -> User:
    if user.role != "authority":
        raise HTTPException(403, "authority role required")
    return user


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
def _page(name: str) -> str:
    f = WEB_DIR / name
    return f.read_text(encoding="utf-8") if f.exists() else f"<h1>occlubio</h1><p>{name} missing.</p>"


@app.get("/", response_class=HTMLResponse)
def index():
    return _page("index.html")


@app.get("/login", response_class=HTMLResponse)
def login_page():
    return _page("login.html")


# ---------------------------------------------------------------- accounts
@app.post("/api/register", response_model=AuthResponse)
def register(req: RegisterRequest, db: DBSession = Depends(get_db)):
    role = (req.role or "user").lower()
    if role not in ("user", "authority"):
        raise HTTPException(422, "role must be 'user' or 'authority'")
    if role == "authority" and req.authority_code != AUTHORITY_CODE:
        raise HTTPException(403, "invalid authority access code")
    if db.query(User).filter((User.username == req.username) | (User.email == req.email)).first():
        raise HTTPException(409, "username or email already taken")
    user = User(username=req.username, email=req.email,
                password_hash=hash_password(req.password), role=role)
    db.add(user)
    db.commit()
    db.refresh(user)
    token = _issue_token(db, user)
    return AuthResponse(token=token, user_id=user.id, username=user.username,
                        role=user.role, enrolled=False)


@app.post("/api/login", response_model=AuthResponse)
def login(req: LoginRequest, db: DBSession = Depends(get_db)):
    user = db.query(User).filter(User.username == req.username).first()
    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(401, "invalid credentials")
    token = _issue_token(db, user)
    return AuthResponse(token=token, user_id=user.id, username=user.username,
                        role=user.role, enrolled=user.enrollment is not None)


@app.post("/api/logout")
def logout(user: User = Depends(current_user), authorization: str = Header(None),
           db: DBSession = Depends(get_db)):
    sess = db.get(Session, authorization.split(" ", 1)[1].strip())
    if sess:
        db.delete(sess)
        db.commit()
    return {"ok": True}


@app.get("/api/me", response_model=UserOut)
def me(user: User = Depends(current_user)):
    return UserOut(id=user.id, username=user.username, email=user.email,
                   role=user.role, enrolled=user.enrollment is not None)


@app.get("/api/users", response_model=List[UserOut])
def list_users(user: User = Depends(require_authority), db: DBSession = Depends(get_db)):
    return [UserOut(id=u.id, username=u.username, email=u.email, role=u.role,
                    enrolled=u.enrollment is not None)
            for u in db.query(User).order_by(User.id).all()]


@app.delete("/api/users/{user_id}")
def remove_participant(user_id: int, caller: User = Depends(require_authority),
                       db: DBSession = Depends(get_db)):
    """Authority removes a participant: deletes the account, their face template,
    sessions and messages, then rebuilds the search index."""
    if user_id == caller.id:
        raise HTTPException(400, "you cannot remove your own account")
    target = db.get(User, user_id)
    if not target:
        raise HTTPException(404, "user not found")
    username = target.username
    db.query(Message).filter((Message.sender_id == user_id) |
                             (Message.recipient_id == user_id)).delete(synchronize_session=False)
    db.query(Session).filter(Session.user_id == user_id).delete(synchronize_session=False)
    db.delete(target)            # cascade removes the Enrollment row
    db.commit()
    service.rebuild_index(db)    # face template gone from FAISS too
    log.info("authority %s removed participant %s (id=%d)", caller.username, username, user_id)
    return {"removed": user_id, "username": username}


# ---------------------------------------------------------------- messages (authority -> participant)
def _msg_out(m: Message, names: dict) -> MessageOut:
    return MessageOut(id=m.id, sender=names.get(m.sender_id, "?"),
                      recipient=("All participants" if m.recipient_id is None
                                 else names.get(m.recipient_id, "?")),
                      body=m.body, created_at=m.created_at.isoformat(timespec="seconds"))


@app.post("/api/messages", response_model=MessageOut)
def send_message(req: MessageCreate, caller: User = Depends(require_authority),
                 db: DBSession = Depends(get_db)):
    if not req.body.strip():
        raise HTTPException(422, "message body is empty")
    if req.recipient_id is not None and not db.get(User, req.recipient_id):
        raise HTTPException(404, "recipient not found")
    m = Message(sender_id=caller.id, recipient_id=req.recipient_id, body=req.body.strip())
    db.add(m)
    db.commit()
    db.refresh(m)
    names = {u.id: u.username for u in db.query(User).all()}
    return _msg_out(m, names)


@app.get("/api/messages", response_model=List[MessageOut])
def list_sent_messages(caller: User = Depends(require_authority), db: DBSession = Depends(get_db)):
    names = {u.id: u.username for u in db.query(User).all()}
    msgs = db.query(Message).order_by(Message.created_at.desc()).all()
    return [_msg_out(m, names) for m in msgs]


@app.get("/api/me/messages", response_model=List[MessageOut])
def my_messages(caller: User = Depends(current_user), db: DBSession = Depends(get_db)):
    """Notices addressed to me, plus broadcasts to all participants."""
    names = {u.id: u.username for u in db.query(User).all()}
    msgs = (db.query(Message)
            .filter((Message.recipient_id == caller.id) | (Message.recipient_id.is_(None)))
            .order_by(Message.created_at.desc()).all())
    return [_msg_out(m, names) for m in msgs]


# ---------------------------------------------------------------- enrollment
@app.post("/api/users/{user_id}/enroll", response_model=EnrollResponse)
async def enroll(user_id: int, files: List[UploadFile] = File(...), force: bool = False,
                 caller: User = Depends(current_user), db: DBSession = Depends(get_db)):
    # A regular user may only enroll themselves; an authority may enroll anyone.
    if caller.role != "authority" and caller.id != user_id:
        raise HTTPException(403, "you can only enroll your own face")
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
                   stride: int = Form(2), user: User = Depends(require_authority),
                   db: DBSession = Depends(get_db)):
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
def job_status(job_id: int, user: User = Depends(require_authority),
               db: DBSession = Depends(get_db)):
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(404, "job not found")
    report = None
    if job.report_path and Path(job.report_path).exists():
        report = json.loads(Path(job.report_path).read_text(encoding="utf-8"))
    return JobOut(id=job.id, status=job.status, message=job.message, report=report,
                  output_video=(f"/api/jobs/{job.id}/video" if job.output_video else None))


@app.get("/api/jobs/{job_id}/video")
def job_video(job_id: int, token: Optional[str] = None,
              authorization: Optional[str] = Header(None), db: DBSession = Depends(get_db)):
    # Browser <a download> links can't send headers, so accept the token as ?token= too.
    tok = token or (authorization.split(" ", 1)[1].strip()
                    if authorization and " " in authorization else None)
    sess = db.get(Session, tok) if tok else None
    caller = db.get(User, sess.user_id) if sess else None
    if not caller or caller.role != "authority":
        raise HTTPException(403, "authority role required")
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
