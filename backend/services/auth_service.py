"""
auth_service.py — ลงทะเบียนหน่วยบริการและยืนยันตัวตน

ตามคำตอบข้อ 1 อสม. และเจ้าหน้าที่สาธารณสุขมีสิทธิ์เท่ากัน จึงใช้ role เดียวกัน
ตามคำตอบข้อ 2 ผู้ป่วยไม่เข้าถึงระบบโดยตรง ต้องผ่านเจ้าหน้าที่เสมอ
                 จึงไม่มี role ผู้ป่วยในระบบเลย
ตามคำตอบข้อ 3 ต้องเลือกหน่วยบริการก่อนใช้งาน

คำเตือนด้านความปลอดภัยสำหรับ Prototype:
บัญชีใน config.DEMO_USERS เป็น placeholder สำหรับทดสอบ flow เท่านั้น
ระบบจริงต้องเปลี่ยนไปใช้รหัสหน่วยบริการที่ออกโดยหน่วยงานต้นสังกัด
พร้อมการเก็บรหัสผ่านแบบ hash และนโยบายรหัสผ่านที่เหมาะสม
"""
import secrets
from typing import Dict, Optional

from config import DEMO_USERS, HEALTH_UNITS
from database import get_conn

# เก็บ session ในหน่วยความจำ เพียงพอสำหรับ Prototype ที่รันเครื่องเดียว
_SESSIONS: Dict[str, Dict] = {}


def list_units() -> list:
    with get_conn() as conn:
        rows = conn.execute("SELECT code, name FROM health_units ORDER BY code").fetchall()
    return [dict(r) for r in rows] or HEALTH_UNITS


def login(username: str, password: str, unit_code: str) -> Optional[Dict]:
    user = DEMO_USERS.get((username or "").strip())
    if not user or user["password"] != (password or "").strip():
        return None

    units = {u["code"]: u["name"] for u in list_units()}
    if unit_code not in units:
        return None

    token = secrets.token_urlsafe(24)
    session = {
        "token": token,
        "username": username,
        "role": user["role"],
        "display": user["display"],
        "unit_code": unit_code,
        "unit_name": units[unit_code],
    }
    _SESSIONS[token] = session
    return session


def get_session(token: str) -> Optional[Dict]:
    return _SESSIONS.get((token or "").strip())


def logout(token: str) -> bool:
    return _SESSIONS.pop((token or "").strip(), None) is not None


def require_session(token: str) -> Dict:
    session = get_session(token)
    if not session:
        raise PermissionError("ยังไม่ได้เข้าสู่ระบบ หรือ session หมดอายุแล้ว")
    return session
