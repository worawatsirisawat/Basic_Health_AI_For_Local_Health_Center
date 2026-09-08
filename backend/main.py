"""
main.py — FastAPI application และ routes ทั้งหมด

โครงสร้าง endpoint แยกตามฟังก์ชันการทำงาน:
  /api/auth/*      ลงทะเบียนหน่วยบริการและเข้าสู่ระบบ
  /api/medicine/*  Medicine Mode และการจัดการบัญชียา
  /api/wound/*     Wounded Mode
  /api/consent/*   ความยินยอมและบันทึกประวัติ
  /api/system/*    สถานะระบบและกฎความปลอดภัย

รัน: python backend/main.py  แล้วเปิด http://127.0.0.1:8000
"""
import base64
import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import (APP_NAME, APP_VERSION, FRONTEND_DIR, MAX_QUESTIONS,
                    MIN_QUESTIONS, UPLOAD_DIR)
from database import init_db
from models import (ConsentSaveRequest, LoginRequest, MedicineAddRequest,
                    MedicineChatRequest, QuantityUpdateRequest,
                    WoundAnalyzeRequest)
from services import (auth_service, consent_service, llm_service,
                      medicine_service, pubmed_service, triage_service,
                      wound_service)

@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    print(f"\n  {APP_NAME} v{APP_VERSION}")
    print(f"  LLM mode : {llm_service.status()['mode']}  ({llm_service.status()['model']})")
    print(f"  PubMed   : {'เปิดใช้งาน' if pubmed_service.status()['enabled'] else 'ปิด'}"
          f" — {pubmed_service.status()['rate_limit']}")
    print(f"  เปิดใช้งานที่ http://127.0.0.1:8000\n")
    yield


app = FastAPI(title=APP_NAME, version=APP_VERSION, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


def _session(token: str):
    try:
        return auth_service.require_session(token)
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Auth — การลงทะเบียนหน่วยบริการ (ข้อ 1-3)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/auth/units")
def get_units():
    return {"units": auth_service.list_units()}


@app.post("/api/auth/login")
def login(req: LoginRequest):
    session = auth_service.login(req.username, req.password, req.unit_code)
    if not session:
        raise HTTPException(status_code=401,
                            detail="ชื่อผู้ใช้ รหัสผ่าน หรือหน่วยบริการไม่ถูกต้อง")
    return session


@app.post("/api/auth/logout")
def logout(x_token: str = Header(default="")):
    return {"ok": auth_service.logout(x_token)}


@app.get("/api/auth/me")
def me(x_token: str = Header(default="")):
    return _session(x_token)


# ─────────────────────────────────────────────────────────────────────────────
# Medicine Mode — บทสนทนา (ข้อ 6-10, 20, 21)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/medicine/chat")
def medicine_chat(req: MedicineChatRequest, x_token: str = Header(default="")):
    session = _session(x_token)
    unit_code = session["unit_code"]

    history = [{"role": m.role, "content": m.content} for m in req.history]
    if req.message:
        history.append({"role": "user", "content": req.message})

    conversation_text = " ".join(m["content"] for m in history if m["role"] == "user")

    # ── ชั้นที่ 1: กฎตายตัว ทำงานก่อน AI เสมอ ────────────────────────────────
    red_flag = triage_service.check_symptoms(conversation_text)
    if red_flag["triggered"]:
        return {
            "phase": "refer_immediately",
            "red_flag": red_flag,
            "message": triage_service.build_referral_message(red_flag["matches"]),
            "history": history,
            "layer": "hard_rule",
            "note": "การแจ้งเตือนนี้มาจากกฎตายตัวของระบบ ไม่ได้มาจากการประเมินของ AI",
        }

    # ── ชั้นที่ 2: ให้ AI ประเมินแบบยืดหยุ่น ─────────────────────────────────
    evidence = []
    if req.use_pubmed and conversation_text:
        evidence = pubmed_service.search_for_symptoms(conversation_text)

    stock = medicine_service.list_medicines(unit_code, in_stock_only=True)
    stock_text = medicine_service.format_for_prompt(stock)

    result = llm_service.medicine_turn(history, stock_text, session["unit_name"], evidence)

    if result.get("question"):
        history.append({"role": "assistant", "content": result["question"]})

    result["history"] = history
    result["evidence"] = evidence
    result["questions_asked"] = sum(1 for m in history if m["role"] == "assistant")
    result["max_questions"] = MAX_QUESTIONS
    result["disclaimer"] = llm_service.DISCLAIMER
    result["stock_summary"] = medicine_service.stock_summary(unit_code)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Medicine Mode — บัญชียา (ข้อ 11, 12)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/medicine/stock")
def get_stock(in_stock_only: bool = False, x_token: str = Header(default="")):
    session = _session(x_token)
    return {
        "unit": {"code": session["unit_code"], "name": session["unit_name"]},
        "medicines": medicine_service.list_medicines(session["unit_code"], in_stock_only),
        "summary": medicine_service.stock_summary(session["unit_code"]),
    }


@app.patch("/api/medicine/stock/{med_id}")
def update_stock(med_id: int, req: QuantityUpdateRequest, x_token: str = Header(default="")):
    _session(x_token)
    med = medicine_service.update_quantity(med_id, req.quantity)
    if not med:
        raise HTTPException(status_code=404, detail="ไม่พบรายการยานี้")
    return med


@app.post("/api/medicine/stock")
def add_stock(req: MedicineAddRequest, x_token: str = Header(default="")):
    session = _session(x_token)
    try:
        return medicine_service.add_medicine(
            session["unit_code"], req.name, generic=req.generic, form=req.form,
            category=req.category, indications=req.indications,
            quantity=req.quantity, unit=req.unit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.delete("/api/medicine/stock/{med_id}")
def remove_stock(med_id: int, x_token: str = Header(default="")):
    _session(x_token)
    if not medicine_service.delete_medicine(med_id):
        raise HTTPException(status_code=404, detail="ไม่พบรายการยานี้")
    return {"ok": True}


@app.get("/api/medicine/autocomplete")
def autocomplete(q: str = "", x_token: str = Header(default="")):
    _session(x_token)
    return {"suggestions": medicine_service.autocomplete_names(q)}


# ─────────────────────────────────────────────────────────────────────────────
# Wounded Mode (ข้อ 13-18)
# ─────────────────────────────────────────────────────────────────────────────

def _run_wound(image_bytes: bytes, bbox: dict, user_note: str,
               reference_cm, use_pubmed: bool, session: dict):
    analysis = wound_service.analyze(image_bytes, bbox, user_note, reference_cm)

    # ── ชั้นที่ 1: กฎตายตัว ─────────────────────────────────────────────────
    red_flag = triage_service.check_wound(analysis)

    evidence = []
    if use_pubmed:
        hint = "wound infection" if analysis["infection"]["score"] >= 2 else "wound care"
        evidence = pubmed_service.search_for_wound(user_note, hint)

    stock = medicine_service.find_available(
        session["unit_code"], category="เวชภัณฑ์ทำแผล"
    ) or medicine_service.list_medicines(session["unit_code"], in_stock_only=True)
    stock_text = medicine_service.format_for_prompt(stock)

    advice = llm_service.wound_advice(analysis, user_note, stock_text,
                                      session["unit_name"], evidence)

    # ตามคำตอบข้อ 18 ถ้าเข้าเกณฑ์อันตราย ให้ frontend แสดง popup สองทางเลือก
    popup = None
    if red_flag["triggered"]:
        popup = {
            "show": True,
            "title": "ตรวจพบสัญญาณที่ควรส่งต่อโรงพยาบาลทันที",
            "message": triage_service.build_referral_message(red_flag["matches"]),
            "options": [
                {"key": "refer", "label": "ส่งต่อโรงพยาบาลทันที", "style": "danger"},
                {"key": "continue", "label": "ขอถามต่อกับระบบ", "style": "secondary"},
            ],
        }

    return {
        "analysis": analysis,
        "red_flag": red_flag,
        "popup": popup,
        "advice": advice,
        "evidence": evidence,
        "image_url": f"/uploads/{analysis['image_file']}",
        "disclaimer": llm_service.DISCLAIMER,
    }


@app.post("/api/wound/analyze")
async def wound_analyze(
    file: UploadFile = File(...),
    bbox_x: float = Form(0.0), bbox_y: float = Form(0.0),
    bbox_w: float = Form(1.0), bbox_h: float = Form(1.0),
    user_note: str = Form(""), reference_cm: str = Form(""),
    use_pubmed: bool = Form(True),
    x_token: str = Header(default=""),
):
    session = _session(x_token)
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="ไม่พบไฟล์ภาพ")

    ref = None
    try:
        ref = float(reference_cm) if reference_cm else None
    except ValueError:
        ref = None

    bbox = {"x": bbox_x, "y": bbox_y, "w": bbox_w, "h": bbox_h}
    return _run_wound(image_bytes, bbox, user_note, ref, use_pubmed, session)


@app.post("/api/wound/analyze-base64")
def wound_analyze_b64(req: WoundAnalyzeRequest, x_token: str = Header(default="")):
    session = _session(x_token)
    raw = req.image_base64
    if "," in raw:
        raw = raw.split(",", 1)[1]
    try:
        image_bytes = base64.b64decode(raw)
    except Exception:
        raise HTTPException(status_code=400, detail="ข้อมูลภาพไม่ถูกต้อง")

    return _run_wound(image_bytes, req.bbox.model_dump(), req.user_note,
                      req.reference_cm, req.use_pubmed, session)


# ─────────────────────────────────────────────────────────────────────────────
# Consent และบันทึกประวัติ (ข้อ 24, 25)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/consent/prompt")
def get_consent_prompt():
    return consent_service.consent_prompt()


@app.post("/api/consent/session")
def new_session():
    return {"session_id": consent_service.new_session_id()}


@app.post("/api/consent/save")
def save_session(req: ConsentSaveRequest, x_token: str = Header(default="")):
    session = _session(x_token)
    log_id = consent_service.save_log(
        req.session_id, session["unit_code"], req.mode, req.consent_given,
        [m.model_dump() for m in req.transcript],
        req.outcome, req.referred, req.red_flag_ids,
    )
    if log_id is None:
        return {"saved": False,
                "message": "ไม่ได้รับความยินยอม ระบบไม่บันทึกข้อมูลใดๆ"}
    return {"saved": True, "log_id": log_id,
            "message": "บันทึกแล้ว โดยกรองข้อมูลที่อาจระบุตัวตนออกก่อนเสมอ"}


@app.get("/api/consent/logs")
def get_logs(x_token: str = Header(default="")):
    session = _session(x_token)
    return {"logs": consent_service.list_logs(session["unit_code"])}


@app.get("/api/consent/logs/{log_id}")
def get_log_detail(log_id: int, x_token: str = Header(default="")):
    _session(x_token)
    log = consent_service.get_log(log_id)
    if not log:
        raise HTTPException(status_code=404, detail="ไม่พบบันทึกนี้")
    return log


# ─────────────────────────────────────────────────────────────────────────────
# System
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/system/status")
def system_status():
    rules = triage_service.load_rules()
    return {
        "app": APP_NAME, "version": APP_VERSION,
        "llm": llm_service.status(),
        "pubmed": pubmed_service.status(),
        "safety_rules": {
            "symptom_rules": len(rules.get("symptom_red_flags", [])),
            "wound_rules": len(rules.get("wound_red_flags", [])),
            "last_reviewed_by": rules.get("last_reviewed_by"),
        },
        "question_range": {"min": MIN_QUESTIONS, "max": MAX_QUESTIONS},
    }


@app.get("/api/system/rules")
def get_rules():
    return triage_service.load_rules()


@app.post("/api/system/rules/reload")
def reload_rules(x_token: str = Header(default="")):
    _session(x_token)
    rules = triage_service.reload_rules()
    return {"ok": True,
            "symptom_rules": len(rules.get("symptom_red_flags", [])),
            "wound_rules": len(rules.get("wound_red_flags", []))}


# ─────────────────────────────────────────────────────────────────────────────
# Static files
# ─────────────────────────────────────────────────────────────────────────────

app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")


@app.get("/")
def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
