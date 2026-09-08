"""
database.py — จัดการ SQLite ทั้งหมด
ตามคำตอบข้อ 23 เก็บข้อมูลแยกเฉพาะเครื่อง/หน่วยบริการ ไม่ซิงก์ไป server กลาง
ตามคำตอบข้อ 24 ไม่มีคอลัมน์ใดเก็บข้อมูลที่ระบุตัวตนผู้ป่วย
"""
import json
import sqlite3
from contextlib import contextmanager

from config import DB_PATH, DATA_DIR, HEALTH_UNITS

SCHEMA = """
CREATE TABLE IF NOT EXISTS health_units (
    code        TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    registered_at TEXT DEFAULT (datetime('now', 'localtime'))
);

-- บัญชียาแยกตามหน่วยบริการ ตามคำตอบข้อ 11 และ 12
-- เจ้าหน้าที่กรอกชื่อยาและปรับจำนวนเอง อัปเดตเป็นรอบ ไม่ใช่เรียลไทม์
CREATE TABLE IF NOT EXISTS medicines (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    unit_code   TEXT NOT NULL,
    name        TEXT NOT NULL,
    generic     TEXT DEFAULT '',
    form        TEXT DEFAULT '',
    category    TEXT DEFAULT '',
    indications TEXT DEFAULT '',
    quantity    INTEGER NOT NULL DEFAULT 0,
    unit        TEXT DEFAULT 'หน่วย',
    is_custom   INTEGER NOT NULL DEFAULT 0,
    updated_at  TEXT DEFAULT (datetime('now', 'localtime')),
    UNIQUE(unit_code, name)
);

-- บันทึกการสนทนา เก็บเฉพาะเมื่อผู้ใช้ยินยอม ตามคำตอบข้อ 25
-- ไม่มีฟิลด์ชื่อ เลขบัตร หรือข้อมูลระบุตัวตนใดๆ ตามคำตอบข้อ 24
CREATE TABLE IF NOT EXISTS session_logs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL,
    unit_code     TEXT NOT NULL,
    mode          TEXT NOT NULL,
    consent_given INTEGER NOT NULL DEFAULT 0,
    transcript    TEXT NOT NULL,
    outcome       TEXT DEFAULT '',
    referred      INTEGER NOT NULL DEFAULT 0,
    red_flag_ids  TEXT DEFAULT '',
    created_at    TEXT DEFAULT (datetime('now', 'localtime'))
);

CREATE TABLE IF NOT EXISTS pubmed_cache (
    query      TEXT PRIMARY KEY,
    payload    TEXT NOT NULL,
    fetched_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_med_unit ON medicines(unit_code);
CREATE INDEX IF NOT EXISTS idx_log_unit ON session_logs(unit_code);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """สร้างตารางและใส่ข้อมูลตั้งต้นถ้ายังไม่มี"""
    with get_conn() as conn:
        conn.executescript(SCHEMA)

        for unit in HEALTH_UNITS:
            conn.execute(
                "INSERT OR IGNORE INTO health_units (code, name) VALUES (?, ?)",
                (unit["code"], unit["name"]),
            )

        # ใส่ยาตั้งต้นให้แต่ละหน่วยบริการ เฉพาะครั้งแรกที่ตารางยังว่าง
        count = conn.execute("SELECT COUNT(*) AS c FROM medicines").fetchone()["c"]
        if count == 0:
            seed_path = DATA_DIR / "seed_medicines.json"
            if seed_path.exists():
                seed = json.loads(seed_path.read_text(encoding="utf-8"))
                for unit in HEALTH_UNITS:
                    for i, med in enumerate(seed.get("medicines", [])):
                        # ให้แต่ละหน่วยบริการมีสต็อกต่างกันเล็กน้อย เพื่อสาธิตเรื่องยาทดแทน
                        qty = med["quantity"]
                        if unit["code"] == "HU-002" and med["generic"] in ("amoxicillin", "mupirocin"):
                            qty = 0  # จำลองกรณียาหมด เพื่อทดสอบการแนะนำยาทดแทน
                        if unit["code"] == "HU-003" and i % 5 == 0:
                            qty = max(0, qty // 4)
                        conn.execute(
                            """INSERT OR IGNORE INTO medicines
                               (unit_code, name, generic, form, category, indications, quantity, unit)
                               VALUES (?,?,?,?,?,?,?,?)""",
                            (unit["code"], med["name"], med["generic"], med["form"],
                             med["category"], med["indications"], qty, med["unit"]),
                        )


def reset_db():
    """ล้างข้อมูลทั้งหมดแล้วสร้างใหม่ ใช้สำหรับเดโมเท่านั้น"""
    if DB_PATH.exists():
        DB_PATH.unlink()
    init_db()
