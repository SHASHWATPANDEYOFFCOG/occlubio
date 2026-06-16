from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class UserOut(BaseModel):
    id: int
    username: str
    email: str
    enrolled: bool


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
