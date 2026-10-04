from __future__ import annotations

import json
import os
import re
import secrets
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import (BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException,
                     UploadFile)
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session as DBSession

from occlubio.api.schemas import (AlertList, AlertOut, AuthResponse, AvailabilityOut, EnrollResponse,
                                  JobCreated, JobOut, LoginRequest, MessageCreate, MessageOut,
                                  RegisterRequest, SightingOut, UserOut, UserSightings)
from occlubio.db import SessionLocal, get_db, init_db
from occlubio.db.models import Alert, Enrollment, Job, Message, Session, Sighting, User
from occlubio.platform_support import data_path, resource_dir
from occlubio.service.face_service import (FaceService, decode_image, hash_password,
                                          verify_password)
from occlubio.utils import ensure_dir, get_logger

log = get_logger("api")
WEB_DIR = resource_dir() / "web"
UPLOAD_DIR = ensure_dir(data_path("data/uploads"))
OUTPUT_DIR = ensure_dir(data_path("data/outputs"))

AUTHORITY_CODE_FILE = data_path("authority_code.txt")


def _load_authority_code() -> str:
    configured = os.environ.get("OCCLUBIO_AUTHORITY_CODE", "").strip()
    if configured:
        return configured
    if AUTHORITY_CODE_FILE.exists():
        stored = AUTHORITY_CODE_FILE.read_text(encoding="utf-8").strip()
        if stored:
            return stored
    code = secrets.token_urlsafe(12)
    AUTHORITY_CODE_FILE.parent.mkdir(parents=True, exist_ok=True)
    AUTHORITY_CODE_FILE.write_text(code + "\n", encoding="utf-8")
    try:
        AUTHORITY_CODE_FILE.chmod(0o600)
    except OSError:
        pass
    return code


AUTHORITY_CODE = _load_authority_code()

service: FaceService | None = None


def _safe_suffix(filename: Optional[str]) -> str:
    suffix = Path(filename or "").suffix.lower()
    return suffix if re.fullmatch(r"\.[a-z0-9]{1,5}", suffix) else ".mp4"


def _display(u: User) -> str:
    return (u.full_name or "").strip() or u.username


def _mmss(seconds: float) -> str:
    m = int(seconds // 60)
    return f"{m:02d}:{seconds - 60 * m:05.2f}"


def _issue_token(db: DBSession, user: User) -> str:
    token = secrets.token_urlsafe(32)
    db.add(Session(token=token, user_id=user.id))
    db.commit()
    return token


def current_user(authorization: Optional[str] = Header(None),
                 db: DBSession = Depends(get_db)) -> User:
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
    if os.environ.get("OCCLUBIO_AUTHORITY_CODE", "").strip():
        log.info("authority sign-up code: from OCCLUBIO_AUTHORITY_CODE")
    else:
        log.info("authority sign-up code: %s (stored in %s)", AUTHORITY_CODE, AUTHORITY_CODE_FILE.resolve())
    log.info("platform ready")
    yield


app = FastAPI(title="occlubio face-recognition platform", lifespan=lifespan)
if (WEB_DIR / "static").is_dir():
    app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")


def _page(name: str) -> str:
    f = WEB_DIR / name
    return f.read_text(encoding="utf-8") if f.exists() else f"<h1>occlubio</h1><p>{name} missing.</p>"


@app.get("/", response_class=HTMLResponse)
def index():
    return _page("index.html")


@app.get("/login", response_class=HTMLResponse)
def login_page():
    return _page("login.html")


def _user_out(u: User) -> UserOut:
    return UserOut(id=u.id, username=u.username, full_name=u.full_name or "",
                   roll_number=u.roll_number, email=u.email, role=u.role,
                   enrolled=u.enrollment is not None)


@app.post("/api/register", response_model=AuthResponse)
def register(req: RegisterRequest, db: DBSession = Depends(get_db)):
    role = (req.role or "user").lower()
    if role not in ("user", "authority"):
        raise HTTPException(422, "role must be 'user' or 'authority'")
    full_name = (req.full_name or "").strip()
    if not full_name:
        raise HTTPException(422, "full name is required")

    if role == "authority":
        if not secrets.compare_digest((req.authority_code or "").encode(), AUTHORITY_CODE.encode()):
            raise HTTPException(403, "invalid authority access code")
        username = (req.username or "").strip()
        if not username:
            raise HTTPException(422, "username is required for an authority")
        roll_number = None
    else:
        roll_number = (req.roll_number or "").strip()
        if not roll_number:
            raise HTTPException(422, "roll number is required for a student")
        username = roll_number

    field_label = "roll number" if role == "user" else "username"
    if db.query(User).filter(User.email == req.email).first():
        raise HTTPException(409, detail={"field": "email",
                                         "message": "that email is already registered"})
    if roll_number and db.query(User).filter(User.roll_number == roll_number).first():
        raise HTTPException(409, detail={"field": "roll_number",
                                         "message": "that roll number is already registered"})
    if db.query(User).filter(User.username == username).first():
        raise HTTPException(409, detail={"field": field_label.replace(" ", "_"),
                                         "message": f"that {field_label} is already taken"})

    user = User(username=username, full_name=full_name, roll_number=roll_number,
                email=req.email, password_hash=hash_password(req.password), role=role)
    db.add(user)
    db.commit()
    db.refresh(user)
    token = _issue_token(db, user)
    return AuthResponse(token=token, user_id=user.id, username=user.username,
                        full_name=user.full_name or "", roll_number=user.roll_number,
                        role=user.role, enrolled=False)


@app.get("/api/check-availability", response_model=AvailabilityOut)
def check_availability(field: str, value: str, db: DBSession = Depends(get_db)):
    field = (field or "").strip().lower()
    value = (value or "").strip()
    if field == "email":
        value = value.lower()
    if field not in ("username", "roll_number", "email"):
        raise HTTPException(422, "field must be username, roll_number or email")
    if not value:
        return AvailabilityOut(field=field, value=value, available=False, reason="required")
    column = {"username": User.username, "roll_number": User.roll_number, "email": User.email}[field]
    taken = db.query(User).filter(column == value).first() is not None
    return AvailabilityOut(field=field, value=value, available=not taken,
                           reason=("already taken" if taken else None))


@app.post("/api/login", response_model=AuthResponse)
def login(req: LoginRequest, db: DBSession = Depends(get_db)):
    ident = (req.username or "").strip()
    user = (db.query(User)
            .filter((User.username == ident) | (User.roll_number == ident))
            .first())
    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(401, "invalid credentials")
    token = _issue_token(db, user)
    return AuthResponse(token=token, user_id=user.id, username=user.username,
                        full_name=user.full_name or "", roll_number=user.roll_number,
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
    return _user_out(user)


@app.get("/api/users", response_model=List[UserOut])
def list_users(user: User = Depends(require_authority), db: DBSession = Depends(get_db)):
    return [_user_out(u) for u in db.query(User).order_by(User.id).all()]


@app.delete("/api/users/{user_id}")
def remove_participant(user_id: int, caller: User = Depends(require_authority),
                       db: DBSession = Depends(get_db)):
    if user_id == caller.id:
        raise HTTPException(400, "you cannot remove your own account")
    target = db.get(User, user_id)
    if not target:
        raise HTTPException(404, "user not found")
    username = target.username
    db.query(Message).filter((Message.sender_id == user_id) |
                             (Message.recipient_id == user_id)).delete(synchronize_session=False)
    db.query(Session).filter(Session.user_id == user_id).delete(synchronize_session=False)
    db.query(Sighting).filter(Sighting.user_id == user_id).delete(synchronize_session=False)
    db.delete(target)
    db.commit()
    service.rebuild_index(db)
    log.info("authority %s removed participant %s (id=%d)", caller.username, username, user_id)
    return {"removed": user_id, "username": username}


@app.get("/api/users/{user_id}/sightings", response_model=UserSightings)
def user_sightings(user_id: int, caller: User = Depends(current_user),
                   db: DBSession = Depends(get_db)):
    if caller.role != "authority" and caller.id != user_id:
        raise HTTPException(403, "you can only view your own gate activity")
    target = db.get(User, user_id)
    if not target:
        raise HTTPException(404, "user not found")
    rows = (db.query(Sighting).filter(Sighting.user_id == user_id)
            .order_by(Sighting.entered_at.desc()).all())
    by_camera: dict = {}
    for s in rows:
        by_camera[s.camera] = by_camera.get(s.camera, 0) + 1
    out = [SightingOut(id=s.id, camera=s.camera,
                       entered_at=s.entered_at.isoformat(timespec="seconds"),
                       exited_at=s.exited_at.isoformat(timespec="seconds"),
                       duration_s=round(s.duration_s, 2), appearances=s.appearances,
                       confidence=round(s.confidence, 3),
                       in_video=[s.in_video_start, s.in_video_end], job_id=s.job_id)
           for s in rows]
    return UserSightings(
        user_id=target.id, username=target.username, full_name=target.full_name or "",
        total_days=len(rows),
        first_seen=(min(s.entered_at for s in rows).isoformat(timespec="seconds") if rows else None),
        last_seen=(max(s.exited_at for s in rows).isoformat(timespec="seconds") if rows else None),
        total_visible_s=round(sum(s.duration_s for s in rows), 2),
        by_camera=by_camera, sightings=out,
    )


def _msg_out(m: Message, names: dict) -> MessageOut:
    return MessageOut(id=m.id, sender=names.get(m.sender_id, "?"),
                      recipient=("All participants" if m.recipient_id is None
                                 else names.get(m.recipient_id, "?")),
                      body=m.body, created_at=m.created_at.isoformat(timespec="seconds"))


@app.post("/api/messages", response_model=List[MessageOut])
def send_message(req: MessageCreate, caller: User = Depends(require_authority),
                 db: DBSession = Depends(get_db)):
    body = req.body.strip()
    if not body:
        raise HTTPException(422, "message body is empty")
    ids = [i for i in (req.recipient_ids or []) if i is not None]
    for rid in ids:
        if not db.get(User, rid):
            raise HTTPException(404, f"recipient {rid} not found")
    created: List[Message] = []
    if not ids:
        created.append(Message(sender_id=caller.id, recipient_id=None, body=body))
    else:
        for rid in ids:
            created.append(Message(sender_id=caller.id, recipient_id=rid, body=body))
    for m in created:
        db.add(m)
    db.commit()
    for m in created:
        db.refresh(m)
    names = {u.id: _display(u) for u in db.query(User).all()}
    return [_msg_out(m, names) for m in created]


@app.get("/api/messages", response_model=List[MessageOut])
def list_sent_messages(caller: User = Depends(require_authority), db: DBSession = Depends(get_db)):
    names = {u.id: _display(u) for u in db.query(User).all()}
    msgs = db.query(Message).order_by(Message.created_at.desc()).all()
    return [_msg_out(m, names) for m in msgs]


@app.get("/api/me/messages", response_model=List[MessageOut])
def my_messages(caller: User = Depends(current_user), db: DBSession = Depends(get_db)):
    names = {u.id: _display(u) for u in db.query(User).all()}
    msgs = (db.query(Message)
            .filter((Message.recipient_id == caller.id) | (Message.recipient_id.is_(None)))
            .order_by(Message.created_at.desc()).all())
    return [_msg_out(m, names) for m in msgs]


def _alert_out(a: Alert, has_video: bool = False) -> AlertOut:
    return AlertOut(id=a.id, job_id=a.job_id, camera=a.camera, label=a.label,
                    entered_at=a.entered_at.isoformat(timespec="seconds"),
                    exited_at=a.exited_at.isoformat(timespec="seconds"),
                    duration_s=round(a.duration_s, 2), appearances=a.appearances, seen=a.seen,
                    video_start_s=round(a.video_start_s or 0.0, 2),
                    video_end_s=round(a.video_end_s or 0.0, 2), has_video=has_video)


@app.get("/api/alerts", response_model=AlertList)
def list_alerts(user: User = Depends(require_authority), db: DBSession = Depends(get_db)):
    rows = db.query(Alert).order_by(Alert.created_at.desc()).all()
    unread = sum(1 for a in rows if not a.seen)
    playable: dict = {}
    for a in rows:
        if a.job_id not in playable:
            job = db.get(Job, a.job_id)
            playable[a.job_id] = bool(job and job.input_path and Path(job.input_path).exists())
    return AlertList(unread=unread, alerts=[_alert_out(a, playable.get(a.job_id, False)) for a in rows])


@app.post("/api/alerts/read")
def mark_alerts_read(user: User = Depends(require_authority), db: DBSession = Depends(get_db)):
    db.query(Alert).filter(Alert.seen == False).update({"seen": True})
    db.commit()
    return {"ok": True}


@app.post("/api/users/{user_id}/enroll", response_model=EnrollResponse)
async def enroll(user_id: int, files: List[UploadFile] = File(...),
                 caller: User = Depends(current_user), db: DBSession = Depends(get_db)):
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
    if dup:
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
                          n_total=n_total, quality=round(quality, 3), duplicate=None)


def _record_sightings(db: DBSession, job: Job, report: dict) -> int:
    base = job.captured_at or job.created_at
    n = 0
    for p in report.get("people", []):
        uid = p.get("user_id")
        if uid is None:
            continue
        windows_s = p.get("appearance_windows_s") or []
        if not windows_s:
            continue
        first_s = min(w[0] for w in windows_s)
        last_s = max(w[1] for w in windows_s)
        entered = base + timedelta(seconds=float(first_s))
        exited = base + timedelta(seconds=float(last_s))
        conf = round(float(p.get("max_confidence", 0.0)), 3)
        day0 = datetime(entered.year, entered.month, entered.day)
        row = (db.query(Sighting)
               .filter(Sighting.user_id == uid, Sighting.camera == job.camera,
                       Sighting.entered_at >= day0, Sighting.entered_at < day0 + timedelta(days=1))
               .first())
        if row:
            if entered < row.entered_at:
                row.entered_at = entered
                row.in_video_start = _mmss(float(first_s))
            if exited > row.exited_at:
                row.exited_at = exited
                row.in_video_end = _mmss(float(last_s))
            row.appearances += len(windows_s)
            row.duration_s = round((row.exited_at - row.entered_at).total_seconds(), 2)
            row.confidence = max(row.confidence, conf)
            row.job_id = job.id
        else:
            db.add(Sighting(
                user_id=uid, job_id=job.id, camera=job.camera,
                entered_at=entered, exited_at=exited,
                duration_s=round((exited - entered).total_seconds(), 2),
                appearances=len(windows_s), confidence=conf,
                in_video_start=_mmss(float(first_s)), in_video_end=_mmss(float(last_s)),
            ))
            n += 1
    db.commit()
    return n


def _record_alerts(db: DBSession, job: Job, report: dict) -> int:
    base = job.captured_at or job.created_at
    n = 0
    for a in report.get("unknown_alerts", []):
        start_s = float(a.get("first_seen_s", 0.0))
        end_s = float(a.get("last_seen_s", start_s))
        db.add(Alert(
            job_id=job.id, camera=job.camera, label=a.get("alert", "UNKNOWN"),
            entered_at=base + timedelta(seconds=start_s),
            exited_at=base + timedelta(seconds=end_s),
            duration_s=float(a.get("visible_s", 0.0)),
            appearances=int(a.get("appearances", 1)), seen=False,
            video_start_s=start_s, video_end_s=end_s,
        ))
        n += 1
    db.commit()
    return n


def _run_identify(job_id: int, video_path: str, stride: int,
                  target_embedding: Optional[np.ndarray], target_label: str):
    with SessionLocal() as s:
        job = s.get(Job, job_id)
        job.status = "running"
        s.commit()
        try:
            out = (OUTPUT_DIR / f"job_{job_id}.mp4").as_posix()
            rep = (OUTPUT_DIR / f"job_{job_id}.json").as_posix()
            report = service.identify_video(video_path, out, rep, session=s, stride=stride,
                                            target_embedding=target_embedding,
                                            target_label=target_label)
            job.output_video, job.report_path, job.status = out, rep, "done"
            s.commit()
            n_s = _record_sightings(s, job, report)
            n_a = _record_alerts(s, job, report)
            log.info("job %d: %d daily sighting rows, %d unknown alert(s) on camera '%s'",
                     job_id, n_s, n_a, job.camera)
        except Exception as e:
            job.status, job.message = "error", str(e)
            s.commit()
            log.exception("identify job %d failed", job_id)


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(422, "time must be ISO-8601 (e.g. 2026-07-01T13:00:00)")
    return parsed.astimezone().replace(tzinfo=None) if parsed.tzinfo else parsed


@app.post("/api/identify", response_model=JobCreated)
async def identify(background: BackgroundTasks, file: UploadFile = File(...),
                   stride: int = Form(2), camera: str = Form("main-gate"),
                   target_user_id: Optional[int] = Form(None),
                   target_file: Optional[UploadFile] = File(None),
                   window_start: Optional[str] = Form(None),
                   window_end: Optional[str] = Form(None),
                   user: User = Depends(require_authority),
                   db: DBSession = Depends(get_db)):
    start = _parse_dt(window_start) or datetime.now()
    end = _parse_dt(window_end)
    if end and end < start:
        raise HTTPException(422, "window end is before window start")

    target_emb: Optional[np.ndarray] = None
    target_label = "target"
    if target_user_id is not None:
        tmpl = service.template_for_user(db, target_user_id)
        if tmpl is None:
            raise HTTPException(422, "that subject has no enrolled face to search for")
        target_emb = tmpl
        tu = db.get(User, target_user_id)
        target_label = _display(tu) if tu else "target"
    elif target_file is not None:
        img = decode_image(await target_file.read())
        if img is None:
            raise HTTPException(422, "could not read the target photo")
        try:
            tmpl, *_ = service.embed_images([img])
        except ValueError as e:
            raise HTTPException(422, f"no usable face in target photo: {e}")
        target_emb = tmpl
        target_label = "uploaded photo"
    else:
        raise HTTPException(422, "provide a subject to search for: an enrolled ID or a photo")

    job = Job(kind="identify", status="pending", camera=(camera or "main-gate").strip(),
              captured_at=start, window_end=end)
    db.add(job)
    db.commit()
    db.refresh(job)
    dest = (UPLOAD_DIR / f"job_{job.id}{_safe_suffix(file.filename)}").as_posix()
    with open(dest, "wb") as out:
        out.write(await file.read())
    job.input_path = dest
    db.commit()
    background.add_task(_run_identify, job.id, dest, stride, target_emb, target_label)
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


def _authority_from_token(token: Optional[str], authorization: Optional[str],
                          db: DBSession) -> User:
    tok = token or (authorization.split(" ", 1)[1].strip()
                    if authorization and " " in authorization else None)
    sess = db.get(Session, tok) if tok else None
    caller = db.get(User, sess.user_id) if sess else None
    if not caller or caller.role != "authority":
        raise HTTPException(403, "authority role required")
    return caller


@app.get("/api/jobs/{job_id}/video")
def job_video(job_id: int, token: Optional[str] = None,
              authorization: Optional[str] = Header(None), db: DBSession = Depends(get_db)):
    _authority_from_token(token, authorization, db)
    job = db.get(Job, job_id)
    if not job or not job.output_video or not Path(job.output_video).exists():
        raise HTTPException(404, "video not ready")
    return FileResponse(
        str(Path(job.output_video).resolve()),
        media_type="video/mp4",
        filename=f"identification_job_{job_id}.mp4",
    )


@app.get("/api/jobs/{job_id}/source")
def job_source(job_id: int, token: Optional[str] = None,
               authorization: Optional[str] = Header(None), db: DBSession = Depends(get_db)):
    _authority_from_token(token, authorization, db)
    job = db.get(Job, job_id)
    if not job or not job.input_path or not Path(job.input_path).exists():
        raise HTTPException(404, "source clip not available")
    return FileResponse(
        str(Path(job.input_path).resolve()),
        media_type="video/mp4",
        filename=f"clip_job_{job_id}.mp4",
    )
