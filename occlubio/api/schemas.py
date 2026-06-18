from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str
    role: str = "user"               # "user" or "authority"
    authority_code: Optional[str] = None  # required when role == "authority"


class LoginRequest(BaseModel):
    username: str
    password: str


class UserOut(BaseModel):
    id: int
    username: str
    email: str
    role: str = "user"
    enrolled: bool


class AuthResponse(BaseModel):
    token: str
    user_id: int
    username: str
    role: str
    enrolled: bool


class MessageCreate(BaseModel):
    body: str
    recipient_id: Optional[int] = None  # None => broadcast to all participants


class MessageOut(BaseModel):
    id: int
    sender: str
    recipient: str           # username, or "All participants" for a broadcast
    body: str
    created_at: str


class EnrollResponse(BaseModel):
    user_id: int
    username: str
    n_accepted: int
    n_total: int
    quality: float
    duplicate: Optional[dict] = None


class JobCreated(BaseModel):
    job_id: int
    status: str


class JobOut(BaseModel):
    id: int
    status: str
    message: str = ""
    report: Optional[dict] = None
    output_video: Optional[str] = None
