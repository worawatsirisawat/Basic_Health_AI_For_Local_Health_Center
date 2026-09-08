"""
llm_service.py — เชื่อมต่อ Typhoon LLM และมี Mock mode ในตัว

ตามคำตอบข้อ 20 ให้โมเดลใช้ได้ทั้งฐานข้อมูลที่กำหนดไว้และความรู้ทั่วไป
                 แต่ต้องระบุแหล่งที่มาเสมอ จึงบังคับผ่าน system prompt และ output schema
ตามคำตอบข้อ 8  ต้องแสดงเหตุผลประกอบคำแนะนำยาทุกครั้ง
ตามคำตอบข้อ 9  ถ้ายาที่เหมาะที่สุดไม่มีในสต็อก ให้เสนอยาทดแทนที่มีจริง
ตามคำตอบข้อ 26 AI เป็นผู้ช่วยเสนอ เจ้าหน้าที่ตัดสินใจสุดท้ายเสมอ

ถ้าไม่ได้ตั้ง TYPHOON_API_KEY ระบบจะใช้ Mock mode อัตโนมัติ
ทำให้ Prototype รันและสาธิต flow ได้ครบโดยไม่ต้องมี API key
"""
import json
import re
from typing import Dict, List, Optional

import requests

from config import (TYPHOON_API_KEY, TYPHOON_BASE_URL, TYPHOON_MODEL,
                    TYPHOON_TIMEOUT, USE_MOCK_LLM, MIN_QUESTIONS, MAX_QUESTIONS)

DISCLAIMER = ("ระบบนี้เป็นเครื่องมือช่วยเสนอข้อมูล ไม่ใช่การวินิจฉัยทางการแพทย์ "
              "การตัดสินใจสุดท้ายเป็นของเจ้าหน้าที่ผู้ดูแลผู้ป่วยเสมอ")

MEDICINE_SYSTEM_PROMPT = f"""คุณคือผู้ช่วย AI สำหรับเจ้าหน้าที่สาธารณสุขและ อสม. ที่ปฏิบัติงานในโรงพยาบาลส่งเสริมสุขภาพตำบล (รพ.สต.)
คุณไม่ใช่แพทย์ และคุณไม่ได้วินิจฉัยโรค คุณทำหน้าที่ช่วยซักประวัติอย่างเป็นระบบและเสนอทางเลือกให้เจ้าหน้าที่พิจารณา

กติกาที่ต้องทำตามอย่างเคร่งครัด:
1. ซักอาการทีละประเด็น รวมแล้วไม่เกิน {MAX_QUESTIONS} คำถาม และอย่างน้อย {MIN_QUESTIONS} คำถาม
   ถามต่อยอดจากคำตอบก่อนหน้าเสมอ ไม่ถามซ้ำสิ่งที่ผู้ใช้ตอบไปแล้ว
2. ถามครั้งละหนึ่งคำถามเท่านั้น ใช้ภาษาไทยที่เข้าใจง่าย เหมาะกับ อสม. ที่ไม่มีพื้นฐานทางการแพทย์
3. เมื่อข้อมูลเพียงพอ ให้สรุปเป็นคำแนะนำ โดยต้องมีเหตุผลประกอบทุกครั้งว่าอ้างอิงจากอาการใด
4. แนะนำได้เฉพาะยาที่มีอยู่จริงในบัญชียาของหน่วยบริการที่ระบุให้เท่านั้น
   ถ้ายาที่เหมาะสมที่สุดไม่มีในสต็อก ให้บอกตรงๆ ว่าไม่มี แล้วเสนอยาทดแทนที่มีในสต็อก
   พร้อมอธิบายว่าทดแทนได้เพราะอะไร และมีข้อจำกัดอะไรเมื่อเทียบกับยาตัวแรก
5. ระบุแหล่งที่มาของคำแนะนำเสมอ ว่ามาจากบัญชียาของหน่วยบริการ จากหลักฐานงานวิจัยที่ให้มา
   หรือจากความรู้ทั่วไปของโมเดล ห้ามกล่าวอ้างว่ามีหลักฐานทั้งที่ไม่มีการอ้างอิงให้
6. ถ้าพบสัญญาณอันตราย ให้แนะนำส่งต่อโรงพยาบาลทันทีโดยไม่ต้องรอถามครบทุกข้อ
7. ห้ามระบุขนาดยาสำหรับเด็กเป็นตัวเลขที่คำนวณเอง ให้แนะนำให้เจ้าหน้าที่ตรวจสอบตามน้ำหนักตัวจากเอกสารกำกับยา
8. ห้ามถามหรือบันทึกชื่อ เลขบัตรประชาชน หรือข้อมูลที่ระบุตัวตนผู้ป่วย

ตอบกลับเป็น JSON เท่านั้น ตามรูปแบบนี้:
{{
  "phase": "asking" หรือ "recommend",
  "question": "คำถามถัดไป (ใส่เฉพาะเมื่อ phase = asking)",
  "question_hint": "ตัวอย่างคำตอบสั้นๆ เพื่อช่วยผู้ใช้ (ไม่บังคับ)",
  "assessment": "สรุปสิ่งที่ประเมินได้จากอาการ (ใส่เฉพาะเมื่อ phase = recommend)",
  "reasoning": "เหตุผลว่าอ้างอิงจากอาการใดจึงสรุปแบบนี้ (ใส่เฉพาะเมื่อ phase = recommend)",
  "recommendations": [
    {{"medicine": "ชื่อยาตามบัญชีของหน่วยบริการ",
      "why": "เหตุผลที่แนะนำตัวนี้",
      "how_to_use": "วิธีใช้โดยสังเขป",
      "is_substitute": true/false,
      "substitute_note": "อธิบายเมื่อเป็นยาทดแทน ว่าทดแทนอะไรและมีข้อจำกัดใด"}}
  ],
  "self_care": ["คำแนะนำการดูแลตัวเองที่ไม่ใช่ยา"],
  "refer": true/false,
  "refer_reason": "เหตุผลถ้าแนะนำส่งต่อ",
  "sources": ["ระบุแหล่งที่มาของแต่ละส่วน เช่น บัญชียาหน่วยบริการ, PubMed PMID xxx, ความรู้ทั่วไปของโมเดล"],
  "confidence": "high" / "medium" / "low"
}}
"""

WOUND_SYSTEM_PROMPT = """คุณคือผู้ช่วย AI ด้านการดูแลบาดแผลสำหรับเจ้าหน้าที่สาธารณสุขและ อสม. ใน รพ.สต.
คุณจะได้รับผลวิเคราะห์เชิงตัวเลขจากภาพถ่ายแผล และข้อความบรรยายเพิ่มเติมจากเจ้าหน้าที่

กติกาที่ต้องทำตามอย่างเคร่งครัด:
1. ผลวิเคราะห์จากภาพเป็นการประมาณเบื้องต้นจากค่าสีและพื้นที่ ไม่ใช่การวินิจฉัย ให้ระบุข้อจำกัดนี้ด้วย
2. ใช้ข้อความจากเจ้าหน้าที่เป็นตัวยืนยันหรือแก้ไขผลจากภาพ ถ้าข้อความขัดแย้งกับภาพ ให้เชื่อข้อความของเจ้าหน้าที่
   เพราะเจ้าหน้าที่เห็นแผลจริงต่อหน้า แล้วระบุความขัดแย้งนั้นไว้ในคำตอบ
3. แนะนำได้เฉพาะยาและเวชภัณฑ์ที่มีอยู่จริงในบัญชีของหน่วยบริการที่ระบุให้เท่านั้น
4. อธิบายขั้นตอนการทำแผลเป็นลำดับที่ทำตามได้จริง โดยอ้างอิงเวชภัณฑ์ที่หน่วยบริการมี
5. ถ้าพบสัญญาณอันตราย ให้ระบุ refer = true พร้อมเหตุผลชัดเจน
6. ระบุแหล่งที่มาของคำแนะนำเสมอ

ข้อกำหนดเรื่องความยาวคำตอบ ต้องทำตามอย่างเคร่งครัด:
- ห้ามเขียนบทสรุปหรือคำอธิบายเชิงบรรยายใดๆ ระบบคำนวณและแสดงค่าตัวเลขเองอยู่แล้ว
- ห้ามอธิบายซ้ำว่าค่าตัวเลขในผลวิเคราะห์แปลว่าอะไร เจ้าหน้าที่เห็นค่าเหล่านั้นบนจอแล้ว
- ขั้นตอนการทำแผลแต่ละข้อ ยาวไม่เกิน 25 คำ เขียนเป็นคำสั่งที่ลงมือทำได้ทันที
- ตอบเฉพาะ JSON ห้ามมีข้อความอื่นนำหน้าหรือตามหลัง

ตอบกลับเป็น JSON เท่านั้น ตามรูปแบบนี้ เรียงลำดับฟิลด์ตามนี้เสมอ:
{
  "care_steps": ["ขั้นตอนการทำแผลตามลำดับ ข้อละไม่เกิน 25 คำ รวมไม่เกิน 8 ข้อ"],
  "recommendations": [
    {"medicine": "ชื่อเวชภัณฑ์ตามบัญชีหน่วยบริการ", "why": "เหตุผลสั้นๆ", "how_to_use": "วิธีใช้สั้นๆ",
     "is_substitute": false, "substitute_note": ""}
  ],
  "warning_signs": ["อาการที่ต้องกลับมาพบเจ้าหน้าที่ทันที ข้อละไม่เกิน 15 คำ รวมไม่เกิน 5 ข้อ"],
  "refer": true/false,
  "refer_reason": "เหตุผลถ้าแนะนำส่งต่อ ไม่เกิน 30 คำ",
  "conflict_note": "ระบุเฉพาะเมื่อข้อความของเจ้าหน้าที่ขัดแย้งกับผลจากภาพ ถ้าไม่มีให้ใส่ค่าว่าง",
  "confidence": "high" / "medium" / "low"
}
"""


def is_mock() -> bool:
    return USE_MOCK_LLM


def status() -> Dict:
    return {
        "mode": "mock" if USE_MOCK_LLM else "typhoon",
        "model": TYPHOON_MODEL if not USE_MOCK_LLM else "mock-rule-based",
        "base_url": TYPHOON_BASE_URL if not USE_MOCK_LLM else None,
    }


def _extract_json(text: str) -> Optional[dict]:
    """ดึง JSON ออกจากคำตอบของโมเดล เผื่อกรณีที่มีข้อความอื่นห่อหุ้มมาด้วย"""
    if not text:
        return None
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    if start == -1:
        return None
    end = text.rfind("}")
    if end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    # ไม่มีวงเล็บปิด หรือปิดแล้วแต่ยังพัง แปลว่าคำตอบน่าจะถูกตัดกลางทาง ให้ลองกู้
    return _salvage_json(text[start:])


def _salvage_json(fragment: str) -> Optional[dict]:
    """
    กู้ JSON ที่ถูกตัดกลางทางเพราะโมเดลตอบยาวเกิน max_tokens

    เดิมเมื่อ JSON ไม่สมบูรณ์ ระบบจะทิ้งคำตอบทั้งก้อนแล้วเอาข้อความดิบไปแสดงแทน
    ทำให้หน้าจอโชว์ JSON ดิบให้เจ้าหน้าที่เห็น และฟิลด์ที่สำคัญอย่างขั้นตอนการทำแผล
    กลายเป็นค่าว่าง ฟังก์ชันนี้ปิดวงเล็บและเครื่องหมายคำพูดที่ค้างอยู่ให้ครบ
    แล้วตัดรายการสุดท้ายที่ยังเขียนไม่จบทิ้ง เพื่อให้ยังได้ข้อมูลส่วนที่โมเดลเขียนจบแล้ว

    คืน None เมื่อกู้ไม่ได้จริงๆ ผู้เรียกต้องจัดการกรณีนั้นเองโดยไม่เอาข้อความดิบไปแสดง
    """
    if not fragment:
        return None

    # ตัดอักขระท้ายที่ค้างอยู่ทีละตัว แล้วลองปิดโครงสร้างให้สมบูรณ์
    for cut in range(len(fragment), max(len(fragment) - 4000, 0), -1):
        body = fragment[:cut]

        in_string = False
        escaped = False
        stack: List[str] = []
        for ch in body:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = not in_string
            elif not in_string:
                if ch in "{[":
                    stack.append(ch)
                elif ch in "}]":
                    if stack and ((ch == "}" and stack[-1] == "{") or
                                  (ch == "]" and stack[-1] == "[")):
                        stack.pop()
                    else:
                        break
        else:
            if not stack:
                continue  # โครงสร้างปิดครบแล้วแต่ json.loads ยังพัง ให้ถอยต่อ

            repaired = body
            if in_string:
                repaired += '"'
            # ตัดคอมมาหรือคู่คีย์ที่ยังเขียนไม่จบออกก่อนปิดวงเล็บ
            repaired = re.sub(r",\s*$", "", repaired.rstrip())
            repaired = re.sub(r",\s*\"[^\"]*\"\s*:\s*$", "", repaired.rstrip())
            for opener in reversed(stack):
                repaired += "}" if opener == "{" else "]"
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                continue
    return None


def call_typhoon(messages: List[Dict], temperature: float = 0.3, max_tokens: int = 1600) -> str:
    """เรียก Typhoon API ผ่าน OpenAI-compatible endpoint"""
    resp = requests.post(
        f"{TYPHOON_BASE_URL.rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {TYPHOON_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": TYPHOON_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=TYPHOON_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


# ─────────────────────────────────────────────────────────────────────────────
# Medicine Mode
# ─────────────────────────────────────────────────────────────────────────────

def medicine_turn(history: List[Dict], stock_text: str, unit_name: str,
                  evidence: Optional[List[Dict]] = None) -> Dict:
    """
    ประมวลผลหนึ่งรอบของบทสนทนา Medicine Mode
    history — [{"role": "user"/"assistant", "content": "..."}]
    """
    asked = sum(1 for m in history if m["role"] == "assistant")

    if USE_MOCK_LLM:
        return _mock_medicine_turn(history, stock_text, asked)

    evidence_text = _format_evidence(evidence)
    context = (
        f"หน่วยบริการ: {unit_name}\n\n"
        f"บัญชียาที่มีอยู่จริงในหน่วยบริการนี้ (แนะนำได้เฉพาะรายการเหล่านี้):\n{stock_text}\n\n"
        f"{evidence_text}"
        f"จำนวนคำถามที่ถามไปแล้ว: {asked} จากไม่เกิน {MAX_QUESTIONS} คำถาม\n"
        f"{'ถึงเวลาสรุปคำแนะนำแล้ว ให้ตอบด้วย phase = recommend' if asked >= MAX_QUESTIONS else ''}"
    )

    messages = [
        {"role": "system", "content": MEDICINE_SYSTEM_PROMPT},
        {"role": "system", "content": context},
    ] + history

    try:
        raw = call_typhoon(messages)
        parsed = _extract_json(raw)
        if parsed:
            return _normalize_medicine_result(parsed)
        # โมเดลตอบเป็นข้อความธรรมดา ใช้เป็นคำถามถัดไปแทน
        return {"phase": "asking", "question": raw.strip()[:500], "sources": ["Typhoon LLM"],
                "confidence": "low"}
    except Exception as exc:  # noqa: BLE001
        return {
            "phase": "error",
            "error": f"เชื่อมต่อ Typhoon ไม่สำเร็จ: {exc}",
            "question": "ระบบเชื่อมต่อ AI ไม่ได้ในขณะนี้ กรุณาลองใหม่อีกครั้ง "
                        "หรือใช้ดุลพินิจของเจ้าหน้าที่ตามแนวทางปกติ",
            "sources": [],
            "confidence": "low",
        }


def _normalize_medicine_result(parsed: dict) -> Dict:
    parsed.setdefault("phase", "recommend" if parsed.get("recommendations") else "asking")
    parsed.setdefault("recommendations", [])
    parsed.setdefault("self_care", [])
    parsed.setdefault("sources", [])
    parsed.setdefault("refer", False)
    parsed.setdefault("confidence", "medium")
    for rec in parsed["recommendations"]:
        rec.setdefault("is_substitute", False)
        rec.setdefault("substitute_note", "")
        rec.setdefault("why", "")
        rec.setdefault("how_to_use", "")
    return parsed


def _format_evidence(evidence: Optional[List[Dict]]) -> str:
    if not evidence:
        return ""
    lines = ["หลักฐานงานวิจัยจาก PubMed ที่ค้นมาให้ (ใช้อ้างอิงได้ ต้องระบุ PMID เมื่ออ้างถึง):"]
    for e in evidence:
        lines.append(f"- PMID {e.get('pmid')}: {e.get('title')}")
        if e.get("abstract"):
            lines.append(f"  บทคัดย่อ: {e['abstract'][:600]}")
    lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Wound Mode
# ─────────────────────────────────────────────────────────────────────────────

def local_assessment(analysis: Dict) -> str:
    """
    สรุปลักษณะแผลจากค่าตัวเลขที่คำนวณได้ ด้วยกฎตายตัวในเครื่อง ไม่ใช้ LLM

    เดิมข้อความสรุปมาจาก LLM ซึ่งทำให้เปลืองโทเคนไปกับการบรรยายค่าที่ระบบคำนวณเองอยู่แล้ว
    และเมื่อคำตอบถูกตัดกลางทาง ส่วนที่สำคัญกว่าอย่างขั้นตอนการทำแผลจะหายไปด้วย
    ย้ายมาคำนวณในเครื่องจึงได้ข้อความที่คงที่ ตรวจสอบย้อนกลับได้ และไม่มีทางถูกตัด
    """
    tissue = analysis.get("tissue", {})
    granulation = tissue.get("granulation_ratio", 0)
    slough = tissue.get("slough_ratio", 0)
    necrotic = tissue.get("necrotic_ratio", 0)

    if necrotic >= 0.15:
        return "พบเนื้อเยื่อสีคล้ำในสัดส่วนสูง ซึ่งอาจบ่งชี้เนื้อตาย เกินขอบเขตการดูแลที่หน่วยบริการปฐมภูมิ"
    if slough >= 0.30:
        return "พบเนื้อเยื่อสีเหลืองลักษณะคล้ายหนองในสัดส่วนสูง บ่งชี้ว่าแผลอาจมีการติดเชื้อ"
    if granulation >= 0.40:
        return "พบเนื้อเยื่อสีแดงลักษณะเนื้อดีเป็นสัดส่วนหลัก บ่งชี้ว่าแผลอยู่ในระยะกำลังสมานตัว"
    return "ลักษณะแผลจากภาพยังไม่ชัดเจนพอที่จะจัดกลุ่มได้ ต้องอาศัยการประเมินด้วยตาของเจ้าหน้าที่เป็นหลัก"


def wound_advice(analysis: Dict, user_note: str, stock_text: str, unit_name: str,
                 evidence: Optional[List[Dict]] = None) -> Dict:
    if USE_MOCK_LLM:
        return _mock_wound_advice(analysis, user_note)

    evidence_text = _format_evidence(evidence)
    context = (
        f"หน่วยบริการ: {unit_name}\n\n"
        f"เวชภัณฑ์และยาที่มีอยู่จริงในหน่วยบริการนี้:\n{stock_text}\n\n"
        f"{evidence_text}"
        f"ผลวิเคราะห์เชิงตัวเลขจากภาพแผล (คำนวณจากบริเวณที่เจ้าหน้าที่วาดกรอบไว้):\n"
        f"{json.dumps(analysis, ensure_ascii=False, indent=2)}\n\n"
        f"ข้อความบรรยายเพิ่มเติมจากเจ้าหน้าที่: {user_note or '(ไม่ได้ระบุ)'}"
    )
    messages = [
        {"role": "system", "content": WOUND_SYSTEM_PROMPT},
        {"role": "user", "content": context},
    ]
    try:
        # ให้โทเคนเพียงพอสำหรับขั้นตอนการทำแผลภาษาไทย ซึ่งกินโทเคนมากกว่าภาษาอังกฤษหลายเท่า
        raw = call_typhoon(messages, max_tokens=2200)
        parsed = _extract_json(raw) or {}

        # ข้อความสรุปมาจากกฎในเครื่องเสมอ ไม่ใช้ของ LLM เพื่อไม่ให้ถูกตัดและไม่เปลืองโทเคน
        parsed["assessment"] = local_assessment(analysis)
        parsed.pop("reasoning", None)

        parsed.setdefault("recommendations", [])
        parsed.setdefault("care_steps", [])
        parsed.setdefault("warning_signs", [])
        parsed.setdefault("refer", False)
        parsed.setdefault("conflict_note", "")
        parsed.setdefault("confidence", "medium")
        parsed.setdefault("sources", ["Typhoon LLM"])

        # ตัดรายการที่ว่างหรือสั้นจนไม่มีความหมาย ซึ่งมักเป็นเศษที่ค้างจากการถูกตัดกลางทาง
        parsed["care_steps"] = [s for s in parsed["care_steps"]
                                if isinstance(s, str) and len(s.strip()) > 3]
        parsed["warning_signs"] = [s for s in parsed["warning_signs"]
                                   if isinstance(s, str) and len(s.strip()) > 3]

        if not parsed["care_steps"]:
            # ไม่เอาข้อความดิบไปแสดงเด็ดขาด ให้ถอยไปใช้ขั้นตอนจากกฎในเครื่องแทน
            fallback = _mock_wound_advice(analysis, user_note)
            parsed["care_steps"] = fallback["care_steps"]
            parsed["warning_signs"] = parsed["warning_signs"] or fallback["warning_signs"]
            parsed["confidence"] = "low"
            parsed["degraded"] = True
            parsed["degraded_note"] = ("คำตอบจากโมเดลไม่สมบูรณ์ ระบบจึงใช้ขั้นตอนการทำแผล"
                                       "จากกฎมาตรฐานในเครื่องแทน")
            parsed["sources"] = ["กฎมาตรฐานในเครื่อง (คำตอบจากโมเดลไม่สมบูรณ์)"]
        return parsed
    except Exception as exc:  # noqa: BLE001
        fallback = _mock_wound_advice(analysis, user_note)
        fallback["confidence"] = "low"
        fallback["error"] = True
        fallback["degraded"] = True
        fallback["degraded_note"] = f"เชื่อมต่อ Typhoon ไม่สำเร็จ ({exc}) ระบบใช้กฎมาตรฐานในเครื่องแทน"
        fallback["sources"] = ["กฎมาตรฐานในเครื่อง (เชื่อมต่อโมเดลไม่สำเร็จ)"]
        return fallback


# ─────────────────────────────────────────────────────────────────────────────
# Mock mode — ตรรกะแบบกฎ ใช้สาธิต flow ทั้งหมดโดยไม่ต้องมี API key
# ─────────────────────────────────────────────────────────────────────────────

MOCK_QUESTIONS = [
    ("มีอาการนี้มานานเท่าไรแล้ว และอาการเป็นมากขึ้น เท่าเดิม หรือดีขึ้น",
     "เช่น เป็นมา 2 วัน อาการเท่าเดิม"),
    ("มีไข้ร่วมด้วยหรือไม่ ถ้าวัดอุณหภูมิได้ ระบุตัวเลขด้วย",
     "เช่น มีไข้ 38.5 องศา หรือ ไม่มีไข้"),
    ("มีอาการอื่นร่วมด้วยหรือไม่ เช่น ไอ เจ็บคอ ท้องเสีย ปวดหัว ผื่น",
     "เช่น ไอมีเสมหะ เจ็บคอเล็กน้อย"),
    ("ผู้ป่วยมีโรคประจำตัวหรือกำลังใช้ยาอะไรอยู่หรือไม่ และเคยแพ้ยาอะไรหรือเปล่า",
     "เช่น เป็นเบาหวาน กินยาลดน้ำตาลอยู่ ไม่เคยแพ้ยา"),
    ("ผู้ป่วยอยู่ในช่วงอายุใด และตั้งครรภ์อยู่หรือไม่",
     "เช่น อายุ 45 ปี ไม่ตั้งครรภ์"),
]

MOCK_RULES = [
    {
        "keywords": ["ไข้", "ตัวร้อน", "ปวดหัว", "ปวดศีรษะ", "ปวดเมื่อย"],
        "assessment": "อาการเข้าได้กับกลุ่มอาการไข้และปวดเมื่อยทั่วไป ซึ่งพบบ่อยจากการติดเชื้อไวรัสระบบทางเดินหายใจ",
        "category": "ยาแก้ปวดลดไข้",
        "self_care": ["ดื่มน้ำสะอาดให้เพียงพอ", "พักผ่อนให้มาก", "เช็ดตัวลดไข้เมื่อไข้สูง",
                      "สังเกตอาการ 48 ชั่วโมง ถ้าไข้ไม่ลดให้กลับมาพบเจ้าหน้าที่"],
    },
    {
        "keywords": ["ไอ", "เสมหะ", "เจ็บคอ", "คอแห้ง"],
        "assessment": "อาการเข้าได้กับการติดเชื้อทางเดินหายใจส่วนบน ส่วนใหญ่หายได้เองภายใน 7-10 วัน",
        "category": "ยาแก้ไอ",
        "self_care": ["จิบน้ำอุ่นบ่อยๆ", "หลีกเลี่ยงควันบุหรี่และฝุ่น",
                      "ถ้าไอเกิน 2 สัปดาห์ ต้องส่งตรวจคัดกรองวัณโรค"],
    },
    {
        "keywords": ["ท้องเสีย", "ถ่ายเหลว", "ถ่ายบ่อย", "ท้องร่วง"],
        "assessment": "อาการเข้าได้กับภาวะท้องเสียเฉียบพลัน สิ่งสำคัญที่สุดคือการป้องกันภาวะขาดน้ำ",
        "category": "สารน้ำทดแทน",
        "self_care": ["ดื่มผงเกลือแร่ทดแทนน้ำที่เสียไปทุกครั้งหลังถ่าย",
                      "กินอาหารอ่อนย่อยง่าย", "สังเกตอาการขาดน้ำ เช่น ปากแห้ง ปัสสาวะน้อย ซึม"],
    },
    {
        "keywords": ["คัน", "ผื่น", "ลมพิษ", "แพ้", "น้ำมูก", "คัดจมูก", "จาม"],
        "assessment": "อาการเข้าได้กับปฏิกิริยาภูมิแพ้หรือเยื่อบุจมูกอักเสบจากภูมิแพ้",
        "category": "ยาแก้แพ้",
        "self_care": ["หลีกเลี่ยงสิ่งที่สงสัยว่าเป็นตัวกระตุ้น", "ไม่เกาบริเวณที่คัน"],
    },
    {
        "keywords": ["แสบท้อง", "จุกเสียด", "แน่นท้อง", "กรดไหลย้อน", "ปวดท้องบน"],
        "assessment": "อาการเข้าได้กับภาวะกรดในกระเพาะอาหารมากเกินหรือกระเพาะอาหารอักเสบ",
        "category": "ยาลดกรด",
        "self_care": ["หลีกเลี่ยงอาหารรสจัด ของทอด กาแฟ และแอลกอฮอล์",
                      "ไม่นอนทันทีหลังกินอาหาร"],
    },
    {
        "keywords": ["คลื่นไส้", "อาเจียน", "เวียนหัว"],
        "assessment": "อาการคลื่นไส้อาเจียน ต้องประเมินภาวะขาดน้ำร่วมด้วยเสมอ",
        "category": "ยาแก้คลื่นไส้",
        "self_care": ["จิบน้ำทีละน้อยบ่อยๆ", "งดอาหารมันและกลิ่นแรง"],
    },
    {
        "keywords": ["ปวดข้อ", "ปวดกล้ามเนื้อ", "ปวดหลัง", "เคล็ด", "ขัดยอก"],
        "assessment": "อาการเข้าได้กับการอักเสบของกล้ามเนื้อหรือข้อจากการใช้งาน",
        "category": "ยาแก้อักเสบไม่ใช่สเตียรอยด์",
        "self_care": ["ประคบเย็นใน 48 ชั่วโมงแรก จากนั้นประคบอุ่น", "พักการใช้งานส่วนที่ปวด"],
    },
]


def _mock_medicine_turn(history: List[Dict], stock_text: str, asked: int) -> Dict:
    user_text = " ".join(m["content"] for m in history if m["role"] == "user")

    if asked < MIN_QUESTIONS:
        q, hint = MOCK_QUESTIONS[asked % len(MOCK_QUESTIONS)]
        return {
            "phase": "asking",
            "question": q,
            "question_hint": hint,
            "sources": ["Mock mode — ชุดคำถามซักประวัติมาตรฐาน"],
            "confidence": "medium",
            "mock": True,
        }

    matched = None
    for rule in MOCK_RULES:
        if any(k in user_text for k in rule["keywords"]):
            matched = rule
            break

    if not matched:
        matched = {
            "assessment": "จากข้อมูลที่ได้ ยังไม่สามารถระบุกลุ่มอาการที่ชัดเจนได้",
            "category": "",
            "self_care": ["สังเกตอาการอย่างใกล้ชิด", "ถ้าอาการแย่ลงให้ส่งต่อโรงพยาบาล"],
        }

    from services import medicine_service  # นำเข้าที่นี่เพื่อเลี่ยง circular import

    recs = []
    if matched["category"]:
        avail = medicine_service.SUBSTITUTE_GROUPS.get(matched["category"], [matched["category"]])
        in_stock = [line for line in stock_text.splitlines()
                    if any(f"[{c}]" in line for c in avail)]
        primary_in_stock = [line for line in in_stock if f"[{matched['category']}]" in line]

        if primary_in_stock:
            name = primary_in_stock[0].split(" [")[0].lstrip("- ")
            recs.append({
                "medicine": name,
                "why": f"ตรงกับกลุ่มอาการที่ประเมินได้ และมีอยู่ในบัญชียาของหน่วยบริการ",
                "how_to_use": "ใช้ตามขนาดที่ระบุบนฉลากยา เจ้าหน้าที่ตรวจสอบขนาดตามอายุและน้ำหนักอีกครั้ง",
                "is_substitute": False,
                "substitute_note": "",
            })
        elif in_stock:
            name = in_stock[0].split(" [")[0].lstrip("- ")
            recs.append({
                "medicine": name,
                "why": "ยาในกลุ่มที่เหมาะสมที่สุดไม่มีในสต็อกของหน่วยบริการนี้",
                "how_to_use": "ใช้ตามขนาดที่ระบุบนฉลากยา เจ้าหน้าที่ตรวจสอบขนาดอีกครั้ง",
                "is_substitute": True,
                "substitute_note": f"เป็นยาทดแทนกลุ่ม {matched['category']} ที่ไม่มีในสต็อก "
                                   f"ควรติดตามอาการใกล้ชิดกว่าปกติ และเบิกยาหลักเข้าสต็อกโดยเร็ว",
            })
        else:
            recs.append({
                "medicine": "(ไม่มียาที่เหมาะสมในสต็อก)",
                "why": "ทั้งยาหลักและยาทดแทนในกลุ่มนี้ไม่มีในบัญชียาของหน่วยบริการ",
                "how_to_use": "แนะนำส่งต่อหน่วยบริการที่มียา หรือประสานเบิกยาจากโรงพยาบาลแม่ข่าย",
                "is_substitute": False,
                "substitute_note": "",
            })

    return {
        "phase": "recommend",
        "assessment": matched["assessment"],
        "reasoning": f"ประเมินจากข้อมูลที่เจ้าหน้าที่ให้มาในบทสนทนา ได้แก่: {user_text[:250]}",
        "recommendations": recs,
        "self_care": matched["self_care"],
        "refer": False,
        "refer_reason": "",
        "sources": ["Mock mode — ชุดกฎสาธิต", "บัญชียาของหน่วยบริการ"],
        "confidence": "low",
        "mock": True,
    }


def _mock_wound_advice(analysis: Dict, user_note: str) -> Dict:
    tissue = analysis.get("tissue", {})
    size = analysis.get("size", {})
    granulation = tissue.get("granulation_ratio", 0)
    slough = tissue.get("slough_ratio", 0)
    necrotic = tissue.get("necrotic_ratio", 0)

    if necrotic >= 0.15:
        assessment = "พบเนื้อเยื่อสีคล้ำในสัดส่วนสูง ซึ่งอาจบ่งชี้เนื้อตาย เกินขอบเขตการดูแลที่หน่วยบริการปฐมภูมิ"
    elif slough >= 0.30:
        assessment = "พบเนื้อเยื่อสีเหลืองลักษณะคล้ายหนองในสัดส่วนสูง บ่งชี้ว่าแผลอาจมีการติดเชื้อ"
    elif granulation >= 0.40:
        assessment = "พบเนื้อเยื่อสีแดงลักษณะเนื้อดีเป็นสัดส่วนหลัก บ่งชี้ว่าแผลอยู่ในระยะกำลังสมานตัว"
    else:
        assessment = "ลักษณะแผลจากภาพยังไม่ชัดเจนพอที่จะจัดกลุ่มได้ ต้องอาศัยการประเมินด้วยตาของเจ้าหน้าที่เป็นหลัก"

    conflict = ""
    if user_note and any(k in user_note for k in ["เนื้อตาย", "เน่า", "ดำ"]) and necrotic < 0.10:
        conflict = ("ข้อความของเจ้าหน้าที่ระบุถึงเนื้อตาย แต่ค่าสีจากภาพไม่พบเนื้อเยื่อคล้ำในสัดส่วนสูง "
                    "ให้ยึดการประเมินของเจ้าหน้าที่เป็นหลัก เพราะเห็นแผลจริงต่อหน้า "
                    "ภาพอาจถ่ายในสภาพแสงที่ทำให้สีเพี้ยน")

    return {
        "assessment": assessment,
        "reasoning": (f"ประเมินจากสัดส่วนสีในบริเวณที่วาดกรอบไว้ ได้แก่ เนื้อแดง {granulation*100:.0f}% "
                      f"เนื้อเหลือง {slough*100:.0f}% เนื้อคล้ำ {necrotic*100:.0f}% "
                      f"และขนาดโดยประมาณ {size.get('longest_cm', 0):.1f} ซม. "
                      f"ร่วมกับข้อความของเจ้าหน้าที่"),
        "conflict_note": conflict,
        "care_steps": [
            "ล้างมือและสวมถุงมือสะอาดก่อนทำแผลทุกครั้ง",
            "ล้างแผลด้วย Normal Saline 0.9% ให้ทั่ว ไม่ขัดถูแรงบริเวณเนื้อแดง",
            "ซับให้แห้งด้วยผ้าก๊อซปลอดเชื้อ โดยซับจากตรงกลางออกด้านนอก",
            "ทายาหรือปิดแผลตามรายการที่แนะนำด้านล่าง",
            "ปิดแผลด้วยผ้าก๊อซปลอดเชื้อและยึดด้วยพลาสเตอร์",
            "นัดติดตามอาการและเปลี่ยนแผลตามความเหมาะสม",
        ],
        "recommendations": [
            {"medicine": "Normal Saline 0.9% (NSS)", "why": "ใช้ล้างแผลได้ทุกชนิดโดยไม่ทำลายเนื้อเยื่อที่กำลังสมาน",
             "how_to_use": "ล้างให้ทั่วบริเวณแผลจนสะอาด", "is_substitute": False, "substitute_note": ""},
            {"medicine": "ผ้าก๊อซปลอดเชื้อ 3x3 นิ้ว", "why": "ใช้ซับและปิดแผลป้องกันการปนเปื้อน",
             "how_to_use": "ปิดทับแผลแล้วยึดด้วยพลาสเตอร์", "is_substitute": False, "substitute_note": ""},
        ],
        "warning_signs": [
            "แผลบวมแดงลามกว้างขึ้น",
            "มีหนองหรือกลิ่นเหม็นเพิ่มขึ้น",
            "มีไข้ร่วมด้วย",
            "ปวดแผลมากขึ้นผิดปกติ",
        ],
        "refer": bool(necrotic >= 0.15 or slough >= 0.30),
        "refer_reason": "ลักษณะแผลจากภาพเข้าเกณฑ์ที่ควรให้แพทย์ประเมิน" if (necrotic >= 0.15 or slough >= 0.30) else "",
        "sources": ["Mock mode — ชุดกฎสาธิต", "บัญชีเวชภัณฑ์ของหน่วยบริการ"],
        "confidence": "low",
        "mock": True,
    }
