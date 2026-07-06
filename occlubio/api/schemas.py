from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class RegisterRequest(BaseModel):
    email: str
    password: str
    full_name: str
    role: str = "user"
    roll_number: Optional[str] = None
    username: Optional[str] = None
    authority_code: Optional[str] = None


class LoginRequest(BaseModel):
    username: str
    password: str


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
