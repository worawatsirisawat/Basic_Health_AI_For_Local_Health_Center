"""
config.py — ค่าตั้งต้นของระบบทั้งหมด อ่านจากไฟล์ .env ถ้ามี
แยกออกมาเป็นไฟล์เดียวเพื่อให้แก้ค่าได้โดยไม่ต้องแตะโค้ดส่วนอื่น
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = BASE_DIR / "uploads"
FRONTEND_DIR = BASE_DIR / "frontend"

DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)


def _load_dotenv():
    """อ่านไฟล์ .env แบบง่าย ไม่ต้องพึ่ง library เพิ่ม"""
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

# ── ฐานข้อมูล ────────────────────────────────────────────────────────────────
# ตามคำตอบข้อ 23 เก็บข้อมูลบัญชียาแยกเฉพาะเครื่อง/หน่วยบริการ ไม่ใช้ server กลาง
DB_PATH = DATA_DIR / "local_health.db"

# ── Typhoon LLM ──────────────────────────────────────────────────────────────
TYPHOON_BASE_URL = os.getenv("TYPHOON_BASE_URL", "https://api.opentyphoon.ai/v1")
TYPHOON_API_KEY = os.getenv("TYPHOON_API_KEY", "").strip()
TYPHOON_MODEL = os.getenv("TYPHOON_LLM_MODEL", "typhoon-v2.5-30b-a3b-instruct")
TYPHOON_TIMEOUT = int(os.getenv("TYPHOON_TIMEOUT", "60"))

# ถ้าไม่มี API key ระบบจะทำงานใน Mock mode อัตโนมัติ ทำให้ Prototype รันได้ทันที
USE_MOCK_LLM = not bool(TYPHOON_API_KEY)

# ── PubMed / NCBI E-utilities ────────────────────────────────────────────────
# ตามคำตอบข้อ 21 เรียกแบบเรียลไทม์ แต่มี cache กันช้าและกันชน rate limit
# NCBI จำกัด 3 ครั้ง/วินาที ถ้าไม่มี key และ 10 ครั้ง/วินาที ถ้ามี key
PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_API_KEY = os.getenv("PUBMED_API_KEY", "").strip()
PUBMED_TOOL_NAME = os.getenv("PUBMED_TOOL_NAME", "local_health_ai")
PUBMED_EMAIL = os.getenv("PUBMED_EMAIL", "").strip()
PUBMED_TIMEOUT = int(os.getenv("PUBMED_TIMEOUT", "12"))
PUBMED_MAX_RESULTS = int(os.getenv("PUBMED_MAX_RESULTS", "3"))
PUBMED_CACHE_TTL = int(os.getenv("PUBMED_CACHE_TTL", "86400"))  # 24 ชั่วโมง
PUBMED_ENABLED = os.getenv("PUBMED_ENABLED", "true").lower() == "true"

# ── บัญชีสำหรับ Prototype ────────────────────────────────────────────────────
# ตามคำตอบข้อ 3 ระบบต้องลงทะเบียนหน่วยบริการก่อนใช้งาน
# หมายเหตุสำคัญ: บัญชีนี้เป็น placeholder สำหรับทดสอบ flow เท่านั้น
# ระบบจริงต้องใช้รหัสหน่วยบริการที่ออกโดยหน่วยงานต้นสังกัด ไม่ใช่รหัสผ่านคงที่
DEMO_USERS = {
    "local_health_care_user": {"password": "1234", "role": "staff", "display": "เจ้าหน้าที่สาธารณสุข"},
    "admin_xyz_health_care": {"password": "1234", "role": "staff", "display": "ผู้ดูแลหน่วยบริการ"},
}

# ตามคำตอบข้อ 1 อสม. และเจ้าหน้าที่สาธารณสุขมีสิทธิ์เท่ากัน จึงใช้ role เดียว
HEALTH_UNITS = [
    {"code": "HU-001", "name": "รพ.สต. บ้านหนองบัว"},
    {"code": "HU-002", "name": "รพ.สต. บ้านโคกสว่าง"},
    {"code": "HU-003", "name": "รพ.สต. บ้านห้วยทราย"},
    {"code": "HU-004", "name": "ศูนย์สุขภาพชุมชนบ้านนาดี"},
]

# ── ตรรกะการซักประวัติ ───────────────────────────────────────────────────────
# ตามคำตอบข้อ 7 ซัก 3-5 คำถาม โดยถามต่อยอดจากคำตอบก่อนหน้าได้
MIN_QUESTIONS = int(os.getenv("MIN_QUESTIONS", "3"))
MAX_QUESTIONS = int(os.getenv("MAX_QUESTIONS", "5"))

APP_NAME = "Local Health AI"
APP_VERSION = "0.1.0-prototype"
