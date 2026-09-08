"""
wound_service.py — วิเคราะห์ภาพแผลจากบริเวณที่ผู้ใช้วาดกรอบเอง

ตามคำตอบข้อ 13 ผู้ใช้ทำ self segmentation ด้วยการวาดกรอบสี่เหลี่ยมคร่าวๆ
ตามคำตอบข้อ 14 วิเคราะห์ขนาด สี เนื้อเยื่อ และสัญญาณติดเชื้อ
ตามคำตอบข้อ 24 ไม่เก็บข้อมูลระบุตัวตน จึงตัด EXIF ทั้งหมดรวมถึงพิกัด GPS ทิ้งก่อนบันทึก

หมายเหตุด้านเทคนิคที่ต้องเข้าใจตรงกัน:
เวอร์ชันนี้ใช้การวิเคราะห์ค่าสีในปริภูมิ HSV ซึ่งเป็นวิธี rule-based ที่โปร่งใสและอธิบายได้
ยังไม่ใช่โมเดล deep learning จริง ค่าที่ได้เป็นการประมาณเบื้องต้นเท่านั้น

เส้นทางอัปเกรดในเฟสถัดไป: เปลี่ยน segment_from_bbox() ไปเรียก Segment Anything Model (SAM)
ซึ่งรับ prompt แบบกรอบสี่เหลี่ยมได้ตรงกับ UX ที่ออกแบบไว้พอดี แล้วส่ง mask ที่ได้
เข้าฟังก์ชัน analyze_region() เดิมโดยไม่ต้องแก้ส่วนอื่นของระบบ
"""
import colorsys
import io
import math
import uuid
from typing import Dict, Optional, Tuple

from PIL import Image

from config import UPLOAD_DIR

MAX_DIMENSION = 1024  # ย่อภาพเพื่อความเร็วในการประมวลผล


def strip_metadata_and_save(image_bytes: bytes) -> Tuple[str, Image.Image]:
    """
    ตัด metadata ทั้งหมด (รวมพิกัด GPS ที่กล้องมือถือแนบมาอัตโนมัติ) แล้วบันทึกไฟล์
    ตามคำตอบข้อ 24 ที่ตั้งใจไม่เก็บข้อมูลระบุตัวตน
    """
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")

    if max(img.size) > MAX_DIMENSION:
        ratio = MAX_DIMENSION / max(img.size)
        img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)

    # สร้างภาพใหม่จากข้อมูลพิกเซลล้วน ทำให้ EXIF ทุกอย่างหายไปโดยสมบูรณ์
    clean = Image.new("RGB", img.size)
    clean.putdata(list(img.getdata()))

    filename = f"wound_{uuid.uuid4().hex[:12]}.jpg"
    clean.save(UPLOAD_DIR / filename, "JPEG", quality=88)
    return filename, clean


def quick_quality_check(img: Image.Image) -> Dict:
    """
    ตรวจคุณภาพภาพแบบเบาที่สุด — ความสว่างและความคมชัด
    ตามคำตอบข้อ 16 ยังไม่บังคับใช้ในเวอร์ชันแรก จึงทำงานแบบไม่บล็อก
    เพียงแจ้งเตือนให้เจ้าหน้าที่ทราบว่าผลอาจคลาดเคลื่อน
    """
    gray = img.convert("L")
    pixels = list(gray.getdata())
    n = len(pixels)
    mean = sum(pixels) / n
    variance = sum((p - mean) ** 2 for p in pixels) / n

    warnings = []
    if mean < 55:
        warnings.append("ภาพค่อนข้างมืด อาจทำให้การประเมินสีเนื้อเยื่อคลาดเคลื่อน")
    elif mean > 210:
        warnings.append("ภาพสว่างจ้าเกินไป อาจทำให้สีของแผลเพี้ยน")
    if variance < 250:
        warnings.append("ภาพอาจเบลอหรือมีรายละเอียดน้อย แนะนำถ่ายใหม่ให้ชัดขึ้น")

    return {
        "brightness": round(mean, 1),
        "sharpness_proxy": round(variance, 1),
        "warnings": warnings,
        "blocking": False,  # ไม่บล็อกการวิเคราะห์ ตามคำตอบข้อ 16
    }


def segment_from_bbox(img: Image.Image, bbox: Dict) -> Tuple[Image.Image, Dict]:
    """
    ตัดบริเวณแผลตามกรอบที่ผู้ใช้วาด
    bbox รับเป็นสัดส่วน 0-1 เพื่อให้ไม่ขึ้นกับขนาดภาพที่แสดงบนหน้าจอ

    จุดขยายในอนาคต: แทนที่บรรทัด crop ด้วยการเรียก SAM พร้อม box prompt
    แล้วคืน mask จริงแทนกรอบสี่เหลี่ยม
    """
    w, h = img.size
    x1 = max(0, int(bbox.get("x", 0) * w))
    y1 = max(0, int(bbox.get("y", 0) * h))
    x2 = min(w, int((bbox.get("x", 0) + bbox.get("w", 1)) * w))
    y2 = min(h, int((bbox.get("y", 0) + bbox.get("h", 1)) * h))

    if x2 - x1 < 10 or y2 - y1 < 10:
        x1, y1, x2, y2 = 0, 0, w, h  # กรอบเล็กเกินไป ใช้ทั้งภาพแทน

    region = img.crop((x1, y1, x2, y2))
    meta = {
        "bbox_pixels": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "region_size_px": {"w": x2 - x1, "h": y2 - y1},
        "method": "user_bbox_crop",
        "note": "เวอร์ชันนี้ใช้กรอบสี่เหลี่ยมที่ผู้ใช้วาด ยังไม่ใช่ mask ตามรูปร่างแผลจริง",
    }
    return region, meta


def _classify_pixel(r: int, g: int, b: int) -> str:
    """จำแนกพิกเซลตามค่าสีในปริภูมิ HSV เป็นชนิดเนื้อเยื่อโดยประมาณ"""
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    hue = h * 360

    if v < 0.22:
        return "necrotic"        # ดำคล้ำ อาจเป็นเนื้อตายหรือเงามืด
    if s < 0.18 and v > 0.75:
        return "background"      # ขาว อาจเป็นผ้าก๊อซหรือแสงสะท้อน
    if s < 0.15:
        return "background"

    if hue < 12 or hue >= 345:
        return "granulation" if s > 0.35 else "periwound"   # แดง เนื้อดี
    if 12 <= hue < 45:
        return "periwound" if v > 0.55 and s < 0.45 else "granulation"  # ส้ม/ผิวหนัง
    if 45 <= hue < 75:
        return "slough"          # เหลือง อาจเป็นหนองหรือเนื้อตายชนิดอ่อน
    if 75 <= hue < 170:
        return "slough"          # เขียวอมเหลือง อาจบ่งชี้การติดเชื้อบางชนิด
    return "other"


def analyze_region(region: Image.Image, reference_cm: Optional[float] = None,
                   full_image_size: Tuple[int, int] = None,
                   region_meta: Dict = None) -> Dict:
    """วิเคราะห์สัดส่วนเนื้อเยื่อ สี ขนาด และสัญญาณติดเชื้อจากบริเวณที่ตัดมา"""
    # ย่อลงเพื่อความเร็ว การวิเคราะห์สัดส่วนสีไม่ต้องการความละเอียดสูง
    work = region.copy()
    if max(work.size) > 300:
        ratio = 300 / max(work.size)
        work = work.resize((max(1, int(work.width * ratio)), max(1, int(work.height * ratio))))

    counts = {"granulation": 0, "slough": 0, "necrotic": 0, "periwound": 0,
              "background": 0, "other": 0}
    r_sum = g_sum = b_sum = 0
    pixels = list(work.getdata())

    for (r, g, b) in pixels:
        counts[_classify_pixel(r, g, b)] += 1
        r_sum += r; g_sum += g; b_sum += b

    total = len(pixels)
    wound_pixels = counts["granulation"] + counts["slough"] + counts["necrotic"]
    denom = wound_pixels if wound_pixels > 0 else total

    tissue = {
        "granulation_ratio": round(counts["granulation"] / denom, 3),
        "slough_ratio": round(counts["slough"] / denom, 3),
        "necrotic_ratio": round(counts["necrotic"] / denom, 3),
        "periwound_ratio": round(counts["periwound"] / total, 3),
        "wound_area_ratio": round(wound_pixels / total, 3),
        "_labels": {
            "granulation": "เนื้อแดง (granulation) มักบ่งชี้แผลกำลังสมานตัว",
            "slough": "เนื้อเหลือง (slough) อาจเป็นหนองหรือเนื้อตายชนิดอ่อน",
            "necrotic": "เนื้อคล้ำ (necrotic) อาจเป็นเนื้อตาย",
        },
    }

    avg_color = {"r": r_sum // total, "g": g_sum // total, "b": b_sum // total}

    # ประมาณขนาด — ถ้าไม่มีวัตถุอ้างอิงจะใช้ค่าสมมติฐานตามระยะถ่ายภาพทั่วไป
    meta = region_meta or {}
    px_w = meta.get("region_size_px", {}).get("w", region.width)
    px_h = meta.get("region_size_px", {}).get("h", region.height)

    if reference_cm and full_image_size:
        cm_per_px = reference_cm / max(full_image_size)
        calib = f"สอบเทียบจากขนาดอ้างอิงที่เจ้าหน้าที่ระบุ ({reference_cm} ซม. ต่อความกว้างภาพ)"
        calibrated = True
    else:
        # สมมติฐานสำหรับ Prototype: ภาพถ่ายแผลระยะใกล้ทั่วไปครอบคลุมประมาณ 15 ซม.
        cm_per_px = 15.0 / max(full_image_size or (1024, 1024))
        calib = ("ประมาณจากสมมติฐานระยะถ่ายภาพ ยังไม่ได้สอบเทียบ ค่านี้เชื่อถือไม่ได้ "
                 "กรุณาวัดขนาดแผลจริงด้วยไม้บรรทัด หรือระบุขนาดอ้างอิงของภาพเพื่อให้ค่าแม่นขึ้น")
        calibrated = False

    width_cm = px_w * cm_per_px
    height_cm = px_h * cm_per_px

    size = {
        "width_cm": round(width_cm, 1),
        "height_cm": round(height_cm, 1),
        "longest_cm": round(max(width_cm, height_cm), 1),
        "area_cm2": round(width_cm * height_cm * tissue["wound_area_ratio"], 1),
        "calibration": calib,
        "calibrated": calibrated,
    }

    # สัญญาณติดเชื้อ — ให้คะแนนจากหลายปัจจัยประกอบกัน
    infection_signals = []
    score = 0
    if tissue["slough_ratio"] >= 0.20:
        infection_signals.append("พบเนื้อเยื่อสีเหลืองในสัดส่วนสูง อาจเป็นหนอง"); score += 2
    if tissue["periwound_ratio"] >= 0.35 and avg_color["r"] > avg_color["g"] + 40:
        infection_signals.append("ผิวหนังรอบแผลมีสีแดงเข้มกว่าปกติ อาจบ่งชี้การอักเสบลุกลาม"); score += 1
    if tissue["necrotic_ratio"] >= 0.10:
        infection_signals.append("พบเนื้อเยื่อสีคล้ำ อาจเป็นเนื้อตาย"); score += 2
    if counts["slough"] > 0 and counts["granulation"] > 0 and \
            tissue["slough_ratio"] > tissue["granulation_ratio"]:
        infection_signals.append("สัดส่วนเนื้อเหลืองมากกว่าเนื้อแดง บ่งชี้ว่าแผลยังไม่เข้าสู่ระยะสมานตัว"); score += 1

    infection = {
        "score": score,
        "level": "สูง" if score >= 3 else ("ปานกลาง" if score >= 1 else "ต่ำ"),
        "signals": infection_signals or ["ไม่พบสัญญาณติดเชื้อที่ชัดเจนจากค่าสีในภาพ"],
    }

    return {
        "tissue": tissue,
        "avg_color": avg_color,
        "size": size,
        "infection": infection,
        "region_meta": meta,
        "method": "HSV rule-based color analysis",
        "limitation": ("การวิเคราะห์นี้ประมาณจากค่าสีเท่านั้น ไม่ใช่การวินิจฉัยทางการแพทย์ "
                       "ผลอาจคลาดเคลื่อนจากแสง เงา สีผิว และมุมกล้อง "
                       "ให้ใช้การประเมินด้วยตาของเจ้าหน้าที่เป็นหลักเสมอ"),
    }


def analyze(image_bytes: bytes, bbox: Dict, user_note: str = "",
            reference_cm: Optional[float] = None) -> Dict:
    """ขั้นตอนทั้งหมดของ Wounded Mode ตั้งแต่รับภาพจนได้ผลวิเคราะห์"""
    filename, img = strip_metadata_and_save(image_bytes)
    quality = quick_quality_check(img)
    region, meta = segment_from_bbox(img, bbox)
    result = analyze_region(region, reference_cm, img.size, meta)

    result["image_file"] = filename
    result["quality"] = quality
    result["user_note"] = user_note
    result["privacy_note"] = "ตัด metadata ของภาพทั้งหมดรวมถึงพิกัด GPS ออกแล้วก่อนบันทึก"
    return result
