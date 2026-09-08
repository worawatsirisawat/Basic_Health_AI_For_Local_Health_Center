"""
medicine_service.py — จัดการบัญชียาของหน่วยบริการ

ตามคำตอบข้อ 11 อัปเดตจำนวนเป็นรอบโดยเจ้าหน้าที่ ไม่หักสต็อกอัตโนมัติเมื่อ AI แนะนำยา
ตามคำตอบข้อ 12 เจ้าหน้าที่กรอกชื่อยาและจำนวนเอง ไม่เชื่อมฐานข้อมูลกลาง
                 แต่มี autocomplete จากรายการยาที่พบบ่อยเพื่อลดการพิมพ์ผิด
ตามคำตอบข้อ 9  ถ้ายาที่เหมาะที่สุดไม่มีในสต็อก ให้หายาทดแทนที่มีจริงในหน่วยบริการ
"""
import json
from typing import Dict, List, Optional

from config import DATA_DIR
from database import get_conn

# กลุ่มยาที่ทดแทนกันได้ในเชิงข้อบ่งใช้เบื้องต้น
# หมายเหตุ: ตารางนี้ยังไม่ผ่านการตรวจสอบโดยเภสัชกร ใช้สาธิตระบบเท่านั้น
SUBSTITUTE_GROUPS = {
    "ยาแก้ปวดลดไข้": ["ยาแก้ปวดลดไข้", "ยาแก้อักเสบไม่ใช่สเตียรอยด์"],
    "ยาแก้อักเสบไม่ใช่สเตียรอยด์": ["ยาแก้อักเสบไม่ใช่สเตียรอยด์", "ยาแก้ปวดลดไข้"],
    "ยาปฏิชีวนะ": ["ยาปฏิชีวนะ"],
    "ยาทาปฏิชีวนะ": ["ยาทาปฏิชีวนะ", "เวชภัณฑ์ทำแผล"],
    "เวชภัณฑ์ทำแผล": ["เวชภัณฑ์ทำแผล"],
    "ยาแก้แพ้": ["ยาแก้แพ้"],
    "ยาลดกรด": ["ยาลดกรด"],
    "ยาแก้คลื่นไส้": ["ยาแก้คลื่นไส้"],
    "ยาแก้ไอ": ["ยาแก้ไอ"],
    "สารน้ำทดแทน": ["สารน้ำทดแทน"],
    "ยาทาแผลไฟไหม้": ["ยาทาแผลไฟไหม้", "เวชภัณฑ์ทำแผล"],
}


def list_medicines(unit_code: str, in_stock_only: bool = False) -> List[Dict]:
    sql = "SELECT * FROM medicines WHERE unit_code = ?"
    if in_stock_only:
        sql += " AND quantity > 0"
    sql += " ORDER BY category, name"
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(sql, (unit_code,)).fetchall()]


def get_medicine(med_id: int) -> Optional[Dict]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM medicines WHERE id = ?", (med_id,)).fetchone()
        return dict(row) if row else None


def update_quantity(med_id: int, quantity: int) -> Optional[Dict]:
    """ปรับจำนวนยา — เจ้าหน้าที่ทำเป็นรอบตามคำตอบข้อ 11"""
    quantity = max(0, int(quantity))
    with get_conn() as conn:
        conn.execute(
            "UPDATE medicines SET quantity = ?, updated_at = datetime('now','localtime') WHERE id = ?",
            (quantity, med_id),
        )
    return get_medicine(med_id)


def add_medicine(unit_code: str, name: str, **kwargs) -> Dict:
    """
    เพิ่มยาที่ระบบยังไม่มี ตามคำตอบข้อ 12
    เจ้าหน้าที่กรอกเอง ระบบไม่ตรวจสอบย้อนกับฐานข้อมูลกลาง
    แต่บันทึก is_custom = 1 ไว้ เพื่อให้ตรวจสอบย้อนหลังได้ว่ารายการใดกรอกเอง
    """
    name = (name or "").strip()
    if not name:
        raise ValueError("ต้องระบุชื่อยา")

    with get_conn() as conn:
        exists = conn.execute(
            "SELECT id FROM medicines WHERE unit_code = ? AND name = ?", (unit_code, name)
        ).fetchone()
        if exists:
            raise ValueError(f"มีรายการ '{name}' ในบัญชียาของหน่วยบริการนี้อยู่แล้ว")

        cur = conn.execute(
            """INSERT INTO medicines
               (unit_code, name, generic, form, category, indications, quantity, unit, is_custom)
               VALUES (?,?,?,?,?,?,?,?,1)""",
            (unit_code, name,
             kwargs.get("generic", ""), kwargs.get("form", ""),
             kwargs.get("category", "อื่นๆ"), kwargs.get("indications", ""),
             max(0, int(kwargs.get("quantity", 0))), kwargs.get("unit", "หน่วย")),
        )
        new_id = cur.lastrowid
    return get_medicine(new_id)


def delete_medicine(med_id: int) -> bool:
    with get_conn() as conn:
        cur = conn.execute("DELETE FROM medicines WHERE id = ?", (med_id,))
        return cur.rowcount > 0


def autocomplete_names(query: str, limit: int = 8) -> List[Dict]:
    """
    รายชื่อยาที่พบบ่อยสำหรับช่วยพิมพ์ ลดความเสี่ยงพิมพ์ชื่อยาผิด
    ทำงานแบบ local ล้วน ไม่เรียก API ภายนอก
    """
    seed_path = DATA_DIR / "seed_medicines.json"
    if not seed_path.exists():
        return []
    seed = json.loads(seed_path.read_text(encoding="utf-8")).get("medicines", [])
    q = (query or "").strip().lower()
    if not q:
        return seed[:limit]
    hits = [m for m in seed if q in m["name"].lower() or q in m.get("generic", "").lower()]
    return hits[:limit]


def find_available(unit_code: str, category: str = "", keywords: List[str] = None) -> List[Dict]:
    """หายาที่มีในสต็อกจริงตามหมวดหรือคำค้น ใช้ป้อนให้ LLM เป็นตัวเลือกที่แนะนำได้"""
    meds = list_medicines(unit_code, in_stock_only=True)
    if category:
        allowed = SUBSTITUTE_GROUPS.get(category, [category])
        meds = [m for m in meds if m["category"] in allowed]
    if keywords:
        kws = [k.lower() for k in keywords if k]
        meds = [
            m for m in meds
            if any(k in (m["name"] + m["generic"] + m["indications"] + m["category"]).lower() for k in kws)
        ]
    return meds


def stock_summary(unit_code: str) -> Dict:
    meds = list_medicines(unit_code)
    in_stock = [m for m in meds if m["quantity"] > 0]
    out_of_stock = [m for m in meds if m["quantity"] == 0]
    low = [m for m in in_stock if m["quantity"] <= 10]
    return {
        "total_items": len(meds),
        "in_stock": len(in_stock),
        "out_of_stock": len(out_of_stock),
        "low_stock": len(low),
        "out_of_stock_names": [m["name"] for m in out_of_stock],
        "low_stock_names": [m["name"] for m in low],
    }


def format_for_prompt(meds: List[Dict], max_items: int = 40) -> str:
    """แปลงรายการยาเป็นข้อความสั้นสำหรับใส่ใน prompt ของ LLM"""
    if not meds:
        return "(หน่วยบริการนี้ไม่มีรายการยาในสต็อกเลย)"
    lines = []
    for m in meds[:max_items]:
        lines.append(
            f"- {m['name']} [{m['category']}] คงเหลือ {m['quantity']} {m['unit']}"
            + (f" | ข้อบ่งใช้: {m['indications']}" if m["indications"] else "")
        )
    return "\n".join(lines)
