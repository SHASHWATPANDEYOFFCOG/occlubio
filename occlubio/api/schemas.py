from __future__ import annotations

import re
from typing import List, Optional

from pydantic import BaseModel, field_validator

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.]{3,32}$")
ROLL_RE = re.compile(r"^[A-Za-z0-9-]{3,20}$")
PASSWORD_MIN = 8


class RegisterRequest(BaseModel):
    email: str
    password: str
    full_name: str
    role: str = "user"
    roll_number: Optional[str] = None
    username: Optional[str] = None
    authority_code: Optional[str] = None

    @field_validator("email")
    @classmethod
    def _email_ok(cls, v: str) -> str:
        v = (v or "").strip().lower()
        if not EMAIL_RE.match(v):
            raise ValueError("enter a valid email address")
        return v

    @field_validator("password")
    @classmethod
    def _password_ok(cls, v: str) -> str:
        if len(v or "") < PASSWORD_MIN:
            raise ValueError(f"password must be at least {PASSWORD_MIN} characters")
        if (v or "").strip() != v:
            raise ValueError("password cannot start or end with a space")
        return v

    @field_validator("full_name")
    @classmethod
    def _name_ok(cls, v: str) -> str:
        v = (v or "").strip()
        if len(v) < 2:
            raise ValueError("full name is required")
        return v

    @field_validator("roll_number")
    @classmethod
    def _roll_ok(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.strip()
        if v and not ROLL_RE.match(v):
            raise ValueError("roll number may use 3-20 letters, digits or hyphens")
        return v

    @field_validator("username")
    @classmethod
    def _username_ok(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.strip()
        if v and not USERNAME_RE.match(v):
            raise ValueError("username may use 3-32 letters, digits, '.' or '_'")
        return v


class LoginRequest(BaseModel):
    username: str
    password: str

    @field_validator("username")
    @classmethod
    def _ident_ok(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("enter your roll number or username")
        return v


class AvailabilityOut(BaseModel):
    field: str
    value: str
    available: bool
    reason: Optional[str] = None


class UserOut(BaseModel):
    id: int
    username: str
    full_name: str = ""
    roll_number: Optional[str] = None
    email: str
    role: str = "user"
    enrolled: bool


class AuthResponse(BaseModel):
    token: str
    user_id: int
    username: str
    full_name: str = ""
    roll_number: Optional[str] = None
    role: str
    enrolled: bool


class MessageCreate(BaseModel):
    body: str
    recipient_ids: Optional[List[int]] = None


class MessageOut(BaseModel):
    id: int
    sender: str
    recipient: str
    body: str
    created_at: str


class EnrollResponse(BaseModel):
    user_id: int
    username: str
    n_accepted: int
    n_total: int
    quality: float
    duplicate: Optional[dict] = None


class SightingOut(BaseModel):
    id: int
    camera: str
    entered_at: str
    exited_at: str
    duration_s: float
    appearances: int = 1
    confidence: float
    in_video: List[str]
    job_id: int


class UserSightings(BaseModel):
    user_id: int
    username: str
    full_name: str = ""
    total_days: int
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    total_visible_s: float = 0.0
    by_camera: dict = {}
    sightings: List[SightingOut] = []


class AlertOut(BaseModel):
    id: int
    job_id: int
    camera: str
    label: str
    entered_at: str
    exited_at: str
    duration_s: float
    appearances: int
    seen: bool
    video_start_s: float = 0.0
    video_end_s: float = 0.0
    has_video: bool = False


class AlertList(BaseModel):
    unread: int
    alerts: List[AlertOut] = []


class JobCreated(BaseModel):
    job_id: int
    status: str


class JobOut(BaseModel):
    id: int
    status: str
    message: str = ""
    report: Optional[dict] = None
    output_video: Optional[str] = None
