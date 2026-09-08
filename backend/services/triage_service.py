"""
triage_service.py — ชั้นความปลอดภัยของระบบทั้งหมด

หลักการออกแบบ (สำคัญที่สุดในระบบนี้):
ตามคำตอบข้อ 10 ผู้ออกแบบเลือกให้ AI ประเมินความเสี่ยงแบบยืดหยุ่น
แต่งานวิจัยพบว่า AI symptom checker วินิจฉัยถูกต้องเป็นอันดับแรกเพียงราว 36-44%
จึงวางสถาปัตยกรรมเป็นสองชั้น:

  ชั้นที่ 1 (hard rules)  — กฎตายตัวจาก data/red_flags.json ทำงานก่อน AI เสมอ
                            ถ้าเข้าเงื่อนไข จะบังคับส่งต่อทันทีโดย AI ไม่มีสิทธิ์ยกเลิก
  ชั้นที่ 2 (flexible AI) — ถ้าไม่เข้ากฎตายตัว จึงให้ AI ประเมินความเสี่ยงเองแบบยืดหยุ่น

ชั้นที่ 1 คือตาข่ายนิรภัย ไม่ได้แทนที่ชั้นที่ 2 แต่กันกรณีที่โมเดลพลาด
"""
import json
from typing import Dict, List, Optional

from config import DATA_DIR

_rules_cache: Optional[dict] = None


def load_rules() -> dict:
    global _rules_cache
    if _rules_cache is None:
        path = DATA_DIR / "red_flags.json"
        _rules_cache = json.loads(path.read_text(encoding="utf-8"))
    return _rules_cache


def reload_rules() -> dict:
    """โหลดกฎใหม่ ใช้เมื่อบุคลากรทางการแพทย์แก้ไฟล์ red_flags.json"""
    global _rules_cache
    _rules_cache = None
    return load_rules()


def _normalize(text: str) -> str:
    return (text or "").replace(" ", "").lower()


def check_symptoms(conversation_text: str) -> Dict:
    """
    ตรวจข้อความสนทนาทั้งหมดกับกฎตายตัวด้านอาการ

    คืนค่า:
      triggered      — True ถ้าเข้าเงื่อนไขอย่างน้อยหนึ่งข้อ
      matches        — รายการกฎที่เข้าเงื่อนไข พร้อมเหตุผล
      severity       — ระดับความรุนแรงสูงสุดที่พบ
    """
    rules = load_rules()
    haystack = _normalize(conversation_text)
    matches: List[Dict] = []

    for rule in rules.get("symptom_red_flags", []):
        hit_kw = next((k for k in rule["keywords"] if _normalize(k) in haystack), None)
        if not hit_kw:
            continue

        if rule.get("require_co"):
            hit_co = next((c for c in rule.get("co_keywords", []) if _normalize(c) in haystack), None)
            if not hit_co:
                continue
            matched_terms = [hit_kw, hit_co]
        else:
            matched_terms = [hit_kw]

        matches.append({
            "id": rule["id"],
            "name": rule["name"],
            "severity": rule["severity"],
            "reason": rule["reason"],
            "action": rule["action"],
            "matched_terms": matched_terms,
        })

    return {
        "triggered": bool(matches),
        "matches": matches,
        "severity": "critical" if matches else "none",
        "layer": "hard_rule",
    }


def check_wound(analysis: Dict) -> Dict:
    """
    ตรวจผลวิเคราะห์ภาพแผลกับกฎตายตัวด้านบาดแผล
    ตามคำตอบข้อ 18 ถ้าเข้าเงื่อนไขอันตราย ให้ frontend แสดง popup
    ที่มีสองทางเลือกคือ ไปโรงพยาบาล หรือ ถามต่อ
    """
    rules = load_rules()
    by_id = {r["id"]: r for r in rules.get("wound_red_flags", [])}
    matches: List[Dict] = []

    def add(rule_id: str, evidence: str):
        rule = by_id.get(rule_id)
        if rule and not any(m["id"] == rule_id for m in matches):
            matches.append({**rule, "evidence": evidence})

    # WRF-002 เนื้อตาย ตรวจจากสัดส่วนพิกเซลสีดำ/คล้ำในบริเวณแผล
    if analysis.get("tissue", {}).get("necrotic_ratio", 0) >= 0.15:
        add("WRF-002", f"พบเนื้อเยื่อสีคล้ำ {analysis['tissue']['necrotic_ratio']*100:.0f}% ของพื้นที่แผล")

    # WRF-003 การติดเชื้อลุกลาม ตรวจจากสัดส่วนสีเหลือง/เขียว (หนอง) และแดงกระจาย
    tissue = analysis.get("tissue", {})
    if tissue.get("slough_ratio", 0) >= 0.30:
        add("WRF-003", f"พบเนื้อเยื่อสีเหลืองคล้ายหนอง {tissue['slough_ratio']*100:.0f}% ของพื้นที่แผล")
    if analysis.get("infection", {}).get("score", 0) >= 3:
        add("WRF-003", "พบสัญญาณบ่งชี้การติดเชื้อหลายอย่างร่วมกัน")

    # WRF-004 ขนาดแผล
    # สำคัญ: ใช้กฎนี้เฉพาะเมื่อผู้ใช้ระบุขนาดอ้างอิงจริงเท่านั้น
    # เพราะถ้าไม่มีการสอบเทียบ ค่าขนาดเป็นเพียงการเดาจากสมมติฐานระยะถ่ายภาพ
    # การบังคับส่งต่อจากค่าที่เดามาจะทำให้ระบบเตือนผิดพลาดบ่อยจนเจ้าหน้าที่เลิกเชื่อการแจ้งเตือน
    # ซึ่งอันตรายกว่าการไม่เตือนเลย เพราะจะทำให้การแจ้งเตือนที่ถูกต้องถูกมองข้ามไปด้วย
    size = analysis.get("size", {})
    size_cm = size.get("longest_cm")
    is_calibrated = size.get("calibrated", False)
    if size_cm and size_cm >= 5 and is_calibrated:
        add("WRF-004", f"ขนาดแผลด้านยาวที่สุด {size_cm:.1f} ซม. (วัดจากขนาดอ้างอิงที่เจ้าหน้าที่ระบุ)")

    # กฎที่ต้องอาศัยข้อความจากเจ้าหน้าที่ (ส่วน hybrid text)
    note = _normalize(analysis.get("user_note", ""))
    for kw in ["เห็นกระดูก", "ลึกมาก", "เห็นเอ็น", "เห็นไขมัน", "ลึกถึงกระดูก"]:
        if _normalize(kw) in note:
            add("WRF-001", f"เจ้าหน้าที่ระบุว่า {kw}")
            break
    for kw in ["เลือดออกไม่หยุด", "เลือดไหลไม่หยุด", "ห้ามเลือดไม่ได้"]:
        if _normalize(kw) in note:
            add("WRF-004", f"เจ้าหน้าที่ระบุว่า {kw}")
            break
    for kw in ["สุนัขกัด", "หมากัด", "แมวกัด", "งูกัด", "สัตว์กัด", "ตะปูตำ", "สนิม"]:
        if _normalize(kw) in note:
            add("WRF-005", f"เจ้าหน้าที่ระบุว่า {kw}")
            break
    for kw in ["เนื้อตาย", "เน่า", "ดำคล้ำ", "มีกลิ่นเหม็น"]:
        if _normalize(kw) in note:
            add("WRF-002", f"เจ้าหน้าที่ระบุว่า {kw}")
            break

    # ข้อสังเกตเชิงคำแนะนำ ไม่บังคับส่งต่อ แต่แจ้งให้เจ้าหน้าที่ตรวจสอบเอง
    advisories = []
    if size_cm and size_cm >= 5 and not is_calibrated:
        advisories.append({
            "id": "ADV-001",
            "name": "แผลอาจมีขนาดใหญ่",
            "note": (f"ระบบประมาณขนาดด้านยาวที่สุดได้ {size_cm:.1f} ซม. "
                     f"แต่ยังไม่ได้สอบเทียบกับขนาดจริง ค่านี้จึงเชื่อถือไม่ได้ "
                     f"กรุณาวัดขนาดแผลจริงด้วยไม้บรรทัด ถ้าเกิน 5 ซม. ให้พิจารณาส่งต่อ"),
        })

    return {
        "triggered": bool(matches),
        "matches": matches,
        "advisories": advisories,
        "severity": "critical" if matches else "none",
        "layer": "hard_rule",
    }


def build_referral_message(matches: List[Dict]) -> str:
    """สร้างข้อความแจ้งเตือนส่งต่อที่ถ้อยคำคงที่ ไม่ให้ AI แต่งเอง"""
    if not matches:
        return ""
    lines = ["ระบบตรวจพบสัญญาณที่เข้าเกณฑ์ส่งต่อโรงพยาบาลทันที"]
    for m in matches:
        lines.append(f"— {m['name']}: {m['reason']}")
    lines.append("")
    lines.append("คำแนะนำ: " + matches[0]["action"])
    lines.append("")
    lines.append("การตัดสินใจสุดท้ายเป็นของเจ้าหน้าที่ผู้ดูแลผู้ป่วยเสมอ "
                 "ระบบนี้ทำหน้าที่แจ้งเตือนและช่วยเสนอข้อมูลเท่านั้น")
    return "\n".join(lines)
