"""
consent_service.py — การขอความยินยอมและบันทึกประวัติการสนทนา

ตามคำตอบข้อ 24 ตั้งใจไม่เก็บข้อมูลที่ระบุตัวตนผู้ป่วยเลย
ตามคำตอบข้อ 25 ถามผู้ใช้ทุกครั้งว่าจะบันทึกได้หรือไม่
                 โดยผู้ใช้ต้องไปถามผู้เข้ารับบริการอีกทอดหนึ่ง

ข้อความขอความยินยอมกำหนดไว้ตายตัวในไฟล์นี้ ไม่ให้ AI แต่งขึ้นเองทุกครั้ง
เพื่อให้ถ้อยคำทางกฎหมายสม่ำเสมอและตรวจสอบย้อนหลังได้
"""
import json
import re
import uuid
from typing import Dict, List, Optional

from database import get_conn

CONSENT_TEXT = (
    "ก่อนบันทึกข้อมูล กรุณาสอบถามผู้เข้ารับบริการก่อนว่า\n\n"
    "\"ขออนุญาตบันทึกข้อมูลอาการและคำแนะนำจากการปรึกษาครั้งนี้ไว้ในระบบ "
    "เพื่อใช้ติดตามการดูแลและปรับปรุงคุณภาพบริการ โดยจะไม่บันทึกชื่อ เลขบัตรประชาชน "
    "หรือข้อมูลที่ระบุตัวตนใดๆ ท่านยินยอมหรือไม่\"\n\n"
    "ถ้าผู้เข้ารับบริการไม่ยินยอม ระบบจะไม่บันทึกข้อมูลใดๆ "
    "และยังใช้งานได้ตามปกติทุกประการ"
)

CONSENT_VERSION = "1.0"

# รูปแบบข้อมูลที่ระบุตัวตนซึ่งต้องกรองออกก่อนบันทึกเสมอ
_PII_PATTERNS = [
    (re.compile(r"\b\d{13}\b"), "[เลขบัตรประชาชนถูกลบ]"),
    (re.compile(r"\b0\d{8,9}\b"), "[เบอร์โทรถูกลบ]"),
    (re.compile(r"[\w\.-]+@[\w\.-]+\.\w+"), "[อีเมลถูกลบ]"),
    (re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b"), "[วันเดือนปีถูกลบ]"),
]


def new_session_id() -> str:
    return uuid.uuid4().hex[:16]


def consent_prompt() -> Dict:
    return {"text": CONSENT_TEXT, "version": CONSENT_VERSION,
            "options": ["ผู้เข้ารับบริการยินยอมให้บันทึก", "ไม่ยินยอม ไม่ต้องบันทึก"]}


def scrub_pii(text: str) -> str:
    """กรองข้อมูลระบุตัวตนที่อาจหลุดเข้ามาในข้อความก่อนบันทึกลงฐานข้อมูล"""
    cleaned = text or ""
    for pattern, replacement in _PII_PATTERNS:
        cleaned = pattern.sub(replacement, cleaned)
    return cleaned


def save_log(session_id: str, unit_code: str, mode: str, consent_given: bool,
             transcript: List[Dict], outcome: str = "", referred: bool = False,
             red_flag_ids: List[str] = None) -> Optional[int]:
    """
    บันทึกประวัติ เฉพาะเมื่อได้รับความยินยอมเท่านั้น
    ถ้าไม่ยินยอม ฟังก์ชันนี้จะไม่เขียนอะไรลงฐานข้อมูลเลยและคืนค่า None
    """
    if not consent_given:
        return None

    safe_transcript = [
        {"role": m.get("role", ""), "content": scrub_pii(m.get("content", ""))}
        for m in (transcript or [])
    ]

    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO session_logs
               (session_id, unit_code, mode, consent_given, transcript, outcome, referred, red_flag_ids)
               VALUES (?,?,?,?,?,?,?,?)""",
            (session_id, unit_code, mode, 1,
             json.dumps(safe_transcript, ensure_ascii=False),
             scrub_pii(outcome), 1 if referred else 0,
             ",".join(red_flag_ids or [])),
        )
        return cur.lastrowid


def list_logs(unit_code: str, limit: int = 50) -> List[Dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT id, session_id, mode, outcome, referred, red_flag_ids, created_at
               FROM session_logs WHERE unit_code = ?
               ORDER BY id DESC LIMIT ?""",
            (unit_code, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def get_log(log_id: int) -> Optional[Dict]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM session_logs WHERE id = ?", (log_id,)).fetchone()
    if not row:
        return None
    data = dict(row)
    data["transcript"] = json.loads(data["transcript"])
    return data
