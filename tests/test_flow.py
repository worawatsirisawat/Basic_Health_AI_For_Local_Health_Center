"""
test_flow.py — ทดสอบ endpoint ทั้งหมดจากต้นจนจบ

วิธีรัน: เปิดเซิร์ฟเวอร์ด้วย python backend/main.py ก่อน แล้วรัน
  python tests/test_flow.py
"""
import base64
import io
import sys

import requests
from PIL import Image

BASE = "http://127.0.0.1:8000"
passed = failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [ผ่าน] {name}")
    else:
        failed += 1
        print(f"  [ไม่ผ่าน] {name}  {detail}")


def make_image(color=(180, 40, 40), size=(400, 300)):
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, "JPEG")
    return buf.getvalue()


def main():
    print("\n=== ทดสอบ Local Health AI Prototype ===\n")

    # 1. สถานะระบบ
    print("[1] สถานะระบบ")
    r = requests.get(f"{BASE}/api/system/status")
    check("เรียกสถานะระบบได้", r.status_code == 200, r.text[:120])
    st = r.json()
    check("มีกฎความปลอดภัยด้านอาการอย่างน้อย 5 ข้อ",
          st["safety_rules"]["symptom_rules"] >= 5)
    check("มีกฎความปลอดภัยด้านบาดแผลอย่างน้อย 3 ข้อ",
          st["safety_rules"]["wound_rules"] >= 3)
    print(f"        โหมด AI: {st['llm']['mode']}")

    # 2. รายชื่อหน่วยบริการ
    print("\n[2] หน่วยบริการและการเข้าสู่ระบบ")
    r = requests.get(f"{BASE}/api/auth/units")
    units = r.json()["units"]
    check("มีรายชื่อหน่วยบริการ", len(units) > 0)

    # เข้าสู่ระบบผิด
    r = requests.post(f"{BASE}/api/auth/login", json={
        "username": "local_health_care_user", "password": "wrong", "unit_code": units[0]["code"]})
    check("รหัสผ่านผิดถูกปฏิเสธ", r.status_code == 401)

    # เข้าสู่ระบบถูก
    r = requests.post(f"{BASE}/api/auth/login", json={
        "username": "local_health_care_user", "password": "1234", "unit_code": units[0]["code"]})
    check("เข้าสู่ระบบสำเร็จ", r.status_code == 200, r.text[:120])
    session = r.json()
    H = {"X-Token": session["token"]}
    print(f"        หน่วยบริการ: {session['unit_name']}")

    # เรียก endpoint โดยไม่มี token
    r = requests.get(f"{BASE}/api/medicine/stock")
    check("เรียก API โดยไม่มี token ถูกปฏิเสธ", r.status_code == 401)

    # 3. บัญชียา
    print("\n[3] บัญชียาและเวชภัณฑ์")
    r = requests.get(f"{BASE}/api/medicine/stock", headers=H)
    check("ดึงบัญชียาได้", r.status_code == 200)
    stock = r.json()
    check("มีรายการยาตั้งต้น", len(stock["medicines"]) > 10)
    print(f"        ทั้งหมด {stock['summary']['total_items']} รายการ "
          f"มีในสต็อก {stock['summary']['in_stock']} หมด {stock['summary']['out_of_stock']}")

    med_id = stock["medicines"][0]["id"]
    original_qty = stock["medicines"][0]["quantity"]
    r = requests.patch(f"{BASE}/api/medicine/stock/{med_id}", headers=H, json={"quantity": 77})
    check("ปรับจำนวนยาได้", r.status_code == 200 and r.json()["quantity"] == 77)
    requests.patch(f"{BASE}/api/medicine/stock/{med_id}", headers=H, json={"quantity": original_qty})

    # autocomplete
    r = requests.get(f"{BASE}/api/medicine/autocomplete?q=para", headers=H)
    check("autocomplete ทำงาน", r.status_code == 200 and len(r.json()["suggestions"]) > 0)

    # เพิ่มยาใหม่
    import uuid
    new_name = f"ยาทดสอบ {uuid.uuid4().hex[:6]}"
    r = requests.post(f"{BASE}/api/medicine/stock", headers=H, json={
        "name": new_name, "category": "อื่นๆ", "quantity": 25, "unit": "เม็ด"})
    check("เพิ่มยาใหม่ได้", r.status_code == 200, r.text[:120])
    new_id = r.json()["id"]
    check("ยาที่เพิ่มเองถูกทำเครื่องหมาย is_custom", r.json()["is_custom"] == 1)

    # เพิ่มซ้ำต้องถูกปฏิเสธ
    r = requests.post(f"{BASE}/api/medicine/stock", headers=H, json={
        "name": new_name, "category": "อื่นๆ", "quantity": 5})
    check("เพิ่มยาชื่อซ้ำถูกปฏิเสธ", r.status_code == 400)

    r = requests.delete(f"{BASE}/api/medicine/stock/{new_id}", headers=H)
    check("ลบยาที่เพิ่มเองได้", r.status_code == 200)

    # 4. Medicine Mode — กรณีปกติ
    print("\n[4] Medicine Mode — บทสนทนาปกติ")
    history = []
    r = requests.post(f"{BASE}/api/medicine/chat", headers=H, json={
        "history": history, "message": "มีไข้ ปวดหัว มา 2 วัน", "use_pubmed": False})
    check("รอบที่ 1 ตอบกลับได้", r.status_code == 200, r.text[:150])
    res = r.json()
    check("รอบที่ 1 อยู่ในช่วงซักถาม", res["phase"] == "asking", f"phase={res.get('phase')}")
    check("มีคำถามกลับมา", bool(res.get("question")))
    history = res["history"]

    for i, msg in enumerate(["เป็นมา 2 วัน อาการเท่าเดิม", "มีไข้ 38 องศา", "ไม่มีอาการอื่น"], start=2):
        r = requests.post(f"{BASE}/api/medicine/chat", headers=H, json={
            "history": history, "message": msg, "use_pubmed": False})
        res = r.json()
        history = res["history"]
        print(f"        รอบที่ {i}: phase = {res.get('phase')}")

    check("สรุปคำแนะนำได้ในที่สุด", res["phase"] == "recommend", f"phase={res.get('phase')}")
    check("มีเหตุผลประกอบคำแนะนำ", bool(res.get("reasoning")))
    check("มีรายการยาที่แนะนำ", len(res.get("recommendations", [])) > 0)
    check("ระบุแหล่งที่มาของคำแนะนำ", len(res.get("sources", [])) > 0)
    check("มีข้อความกำกับความรับผิดชอบ", bool(res.get("disclaimer")))
    if res.get("recommendations"):
        print(f"        แนะนำ: {res['recommendations'][0]['medicine']}")

    # 5. Medicine Mode — กฎตายตัวต้องทำงาน
    print("\n[5] ชั้นกฎตายตัวด้านความปลอดภัย")
    cases = [
        ("เจ็บหน้าอกมาก หายใจลำบาก เหงื่อแตก", "RF-001"),
        ("ผู้ป่วยหมดสติ เรียกไม่รู้สึกตัว", "RF-002"),
        ("ปากเบี้ยว แขนขาอ่อนแรงข้างซ้าย พูดไม่ชัด", "RF-005"),
        ("อาเจียนเป็นเลือด", "RF-006"),
    ]
    for text, expect_id in cases:
        r = requests.post(f"{BASE}/api/medicine/chat", headers=H, json={
            "history": [], "message": text, "use_pubmed": False})
        res = r.json()
        ids = [m["id"] for m in res.get("red_flag", {}).get("matches", [])]
        check(f"ตรวจจับ {expect_id} ได้ ({text[:22]}...)",
              res.get("phase") == "refer_immediately" and expect_id in ids,
              f"phase={res.get('phase')} ids={ids}")

    # กรณีที่ต้องไม่ trigger (มีคำหลักแต่ไม่มีคำร่วม)
    r = requests.post(f"{BASE}/api/medicine/chat", headers=H, json={
        "history": [], "message": "เจ็บหน้าอกนิดหน่อยตอนไอ", "use_pubmed": False})
    check("ไม่ trigger เมื่อไม่มีอาการร่วมที่กำหนด",
          r.json().get("phase") != "refer_immediately", r.json().get("phase"))

    # 6. Wounded Mode
    print("\n[6] Wounded Mode")
    img_bytes = make_image((170, 60, 55))
    files = {"file": ("wound.jpg", img_bytes, "image/jpeg")}
    data = {"bbox_x": 0.2, "bbox_y": 0.2, "bbox_w": 0.5, "bbox_h": 0.5,
            "user_note": "แผลถลอกจากล้ม ทำความสะอาดแล้ว", "use_pubmed": "false"}
    r = requests.post(f"{BASE}/api/wound/analyze", headers=H, files=files, data=data)
    check("วิเคราะห์ภาพแผลได้", r.status_code == 200, r.text[:200])
    w = r.json()
    check("มีผลวิเคราะห์เนื้อเยื่อ", "granulation_ratio" in w["analysis"]["tissue"])
    check("มีการประมาณขนาดแผล", "longest_cm" in w["analysis"]["size"])
    check("มีการประเมินความเสี่ยงติดเชื้อ", "level" in w["analysis"]["infection"])
    check("มีขั้นตอนการทำแผล", len(w["advice"].get("care_steps", [])) > 0)
    check("ระบุว่าตัด metadata แล้ว", "GPS" in w["analysis"]["privacy_note"])
    check("ระบุข้อจำกัดของการวิเคราะห์", bool(w["analysis"].get("limitation")))
    print(f"        เนื้อแดง {w['analysis']['tissue']['granulation_ratio']*100:.0f}% "
          f"ความเสี่ยงติดเชื้อ {w['analysis']['infection']['level']}")

    # กฎตายตัวด้านแผล — ผ่านข้อความ
    files = {"file": ("wound2.jpg", make_image((120, 90, 80)), "image/jpeg")}
    data = {"bbox_x": 0, "bbox_y": 0, "bbox_w": 1, "bbox_h": 1,
            "user_note": "โดนตะปูตำเมื่อวาน แผลลึกมาก", "use_pubmed": "false"}
    r = requests.post(f"{BASE}/api/wound/analyze", headers=H, files=files, data=data)
    w2 = r.json()
    ids = [m["id"] for m in w2["red_flag"]["matches"]]
    check("กฎตายตัวด้านแผลทำงานจากข้อความ", w2["red_flag"]["triggered"], f"ids={ids}")
    check("มี popup สองทางเลือกตามที่ออกแบบ",
          w2.get("popup") and len(w2["popup"]["options"]) == 2)
    print(f"        กฎที่ตรวจพบ: {ids}")

    # กฎเนื้อตายจากสีภาพ
    files = {"file": ("wound3.jpg", make_image((18, 15, 14)), "image/jpeg")}
    data = {"bbox_x": 0, "bbox_y": 0, "bbox_w": 1, "bbox_h": 1,
            "user_note": "", "use_pubmed": "false"}
    r = requests.post(f"{BASE}/api/wound/analyze", headers=H, files=files, data=data)
    w3 = r.json()
    check("ตรวจจับเนื้อคล้ำจากค่าสีในภาพได้",
          "WRF-002" in [m["id"] for m in w3["red_flag"]["matches"]],
          f"necrotic={w3['analysis']['tissue']['necrotic_ratio']}")

    # base64 endpoint
    b64 = base64.b64encode(make_image()).decode()
    r = requests.post(f"{BASE}/api/wound/analyze-base64", headers=H, json={
        "image_base64": b64, "bbox": {"x":0,"y":0,"w":1,"h":1},
        "user_note": "ทดสอบ", "use_pubmed": False})
    check("endpoint แบบ base64 ทำงาน", r.status_code == 200, r.text[:120])

    # 7. Consent
    print("\n[7] ความยินยอมและการบันทึกประวัติ")
    r = requests.get(f"{BASE}/api/consent/prompt")
    check("มีข้อความขอความยินยอมกำหนดตายตัว", r.status_code == 200 and "ยินยอม" in r.json()["text"])

    sid = requests.post(f"{BASE}/api/consent/session").json()["session_id"]

    # ไม่ยินยอม — ต้องไม่บันทึก
    r = requests.post(f"{BASE}/api/consent/save", headers=H, json={
        "session_id": sid, "mode": "medicine", "consent_given": False,
        "transcript": [{"role":"user","content":"ทดสอบ"}]})
    check("ไม่ยินยอมแล้วไม่บันทึกข้อมูล", r.json()["saved"] is False)

    # ยินยอม — ต้องบันทึกและกรอง PII
    r = requests.post(f"{BASE}/api/consent/save", headers=H, json={
        "session_id": sid, "mode": "medicine", "consent_given": True,
        "transcript": [{"role":"user","content":"ผู้ป่วยเลขบัตร 1234567890123 เบอร์ 0812345678 มีไข้"}],
        "outcome": "แนะนำยาลดไข้", "referred": False, "red_flag_ids": []})
    check("ยินยอมแล้วบันทึกได้", r.json()["saved"] is True)
    log_id = r.json()["log_id"]

    r = requests.get(f"{BASE}/api/consent/logs/{log_id}", headers=H)
    content = r.json()["transcript"][0]["content"]
    check("กรองเลขบัตรประชาชนออกแล้ว", "1234567890123" not in content, content)
    check("กรองเบอร์โทรออกแล้ว", "0812345678" not in content, content)
    print(f"        ข้อความที่บันทึกจริง: {content}")

    r = requests.get(f"{BASE}/api/consent/logs", headers=H)
    check("ดึงรายการบันทึกได้", r.status_code == 200 and len(r.json()["logs"]) > 0)

    # 8. หน้าเว็บ
    print("\n[8] หน้าเว็บ")
    for page in ["/", "/index.html", "/select.html", "/medicine.html",
                 "/wound.html", "/stock.html", "/css/style.css", "/js/common.js"]:
        r = requests.get(BASE + page)
        check(f"โหลด {page} ได้", r.status_code == 200, f"status={r.status_code}")

    # 9. logout
    print("\n[9] ออกจากระบบ")
    r = requests.post(f"{BASE}/api/auth/logout", headers=H)
    check("ออกจากระบบได้", r.json()["ok"] is True)
    r = requests.get(f"{BASE}/api/medicine/stock", headers=H)
    check("token ใช้ไม่ได้หลังออกจากระบบ", r.status_code == 401)

    print(f"\n{'='*46}")
    print(f"  ผ่าน {passed} รายการ   ไม่ผ่าน {failed} รายการ")
    print(f"{'='*46}\n")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
