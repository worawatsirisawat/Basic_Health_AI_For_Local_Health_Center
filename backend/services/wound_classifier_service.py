"""
wound_classifier_service.py
จำแนกประเภทบาดแผลด้วยการเปรียบเทียบภาพกับคลังภาพอ้างอิงที่จำแนกไว้แล้ว

หลักการทำงาน
------------
ระบบไม่ได้ "เรียนรู้" น้ำหนักใดๆ จากข้อมูล แต่ใช้กฎที่เขียนไว้ตายตัวในการถอด
คุณลักษณะของภาพ (hand-crafted features) แล้วเทียบความคล้ายกับภาพอ้างอิงที่สุ่ม
มาจากแต่ละโฟลเดอร์ประเภทแผล โฟลเดอร์ใดได้คะแนนความคล้ายสูงสุดจะถูกระบุว่าเป็น
ประเภทของแผลนั้น

คุณลักษณะที่ใช้เปรียบเทียบมีสี่กลุ่ม ทุกกลุ่มคำนวณด้วยสูตรตายตัว ตรวจสอบย้อนกลับได้
  1. HSV colour histogram        เทียบด้วย histogram intersection
  2. Tissue composition vector   ใช้กฎชุดเดียวกับ wound_service._classify_pixel
  3. Colour moments              ค่าเฉลี่ยและส่วนเบี่ยงเบนของแต่ละ channel
  4. Texture (gradient) profile  ค่าเฉลี่ยและส่วนเบี่ยงเบนของ gradient magnitude

หมายเหตุทางวิชาการที่ต้องระบุให้ตรงเมื่อนำเสนอ
---------------------------------------------
การเทียบภาพเข้ากับตัวอย่างที่มี label แล้วเลือกคลาสที่คล้ายที่สุด จัดอยู่ในกลุ่ม
instance-based learning (lazy learning) ซึ่งเป็น machine learning แบบคลาสสิก
ไม่ใช่ rule-based ล้วน และไม่ใช่ deep learning จุดแข็งคืออธิบายได้ทั้งกระบวนการ
เพราะแสดงได้ว่าเทียบกับภาพอ้างอิงไฟล์ใดบ้างและได้คะแนนเท่าใด ส่วนกฎตายตัวล้วน
คือชั้น triage_service ที่ทำงานแยกและอยู่เหนือผลลัพธ์ของโมดูลนี้เสมอ

ข้อจำกัดที่ต้องรู้
-----------------
โมดูลนี้ระบุ "ประเภทของแผล" เพื่อใช้เลือกแนวทางทำแผลเท่านั้น ไม่ใช่การวินิจฉัยโรค
และไม่มีสิทธิ์ยกเลิกผลจาก triage_service ผลลัพธ์ยังไม่ผ่านการตรวจสอบโดยบุคลากร
ทางการแพทย์ ห้ามใช้ตัดสินใจกับผู้ป่วยจริงก่อนผ่านการตรวจสอบ
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

# ---------------------------------------------------------------- ค่าตั้งต้น

# ขนาดที่ย่อภาพก่อนคำนวณ ใหญ่กว่านี้ไม่ได้เพิ่มความแม่นแต่ช้าขึ้นชัดเจน
WORK_SIZE = 128

# จำนวนภาพอ้างอิงที่สุ่มต่อหนึ่งโฟลเดอร์
DEFAULT_SAMPLE_K = 5

# ใช้ค่าเฉลี่ยของคะแนนสูงสุดกี่อันดับจากภาพที่สุ่มมา
# ตั้งเป็น 3 จาก 5 เพื่อกันภาพอ้างอิงที่สุ่มได้ไม่ดีมาฉุดคะแนนทั้งโฟลเดอร์
DEFAULT_TOP_M = 3

# อุณหภูมิของ softmax ที่ใช้แปลงคะแนนดิบเป็นเปอร์เซ็นต์
# ค่าน้อยทำให้ผลต่างถูกขยาย ค่ามากทำให้เปอร์เซ็นต์เกลี่ยเท่ากันมากขึ้น
DEFAULT_TEMPERATURE = 0.045

# ถ้าเปอร์เซ็นต์อันดับหนึ่งห่างอันดับสองน้อยกว่านี้ ถือว่ายังไม่มั่นใจ
LOW_CONFIDENCE_GAP = 8.0

# คะแนนดิบของอันดับหนึ่งที่ต่ำกว่านี้ แปลว่าภาพไม่คล้ายภาพอ้างอิงใดเลยในคลัง
# เช่นภาพเบลอ ภาพมืด หรือภาพที่ไม่ใช่บาดแผล ต้องเตือนแทนการเลือกคลาสที่แย่น้อยที่สุด
# ค่านี้ได้จากการวัดคะแนนภาพแผลจริงเทียบกันเองซึ่งอยู่ราว 0.32 ถึง 0.74
MIN_RAW_SCORE = 0.30

# น้ำหนักของคุณลักษณะแต่ละกลุ่ม รวมกันได้ 1.0
WEIGHTS = {
    "histogram": 0.45,
    "tissue": 0.25,
    "moments": 0.20,
    "texture": 0.10,
}

# หมุนค่า hue ไป 0.5 รอบวง เพื่อให้โทนแดงซึ่งอยู่คร่อมรอยต่อ 0/1 มาอยู่กลางแกน
# ถ้าไม่หมุน พิกเซลสีแดงจะถูกแยกไปอยู่คนละปลายของ histogram ทำให้เทียบผิด
HUE_ROTATION = 0.5

CATEGORY_TH = {
    "Abrasions": "แผลถลอก",
    "Bruises": "แผลฟกช้ำ",
    "Burns": "แผลไหม้",
    "Cut": "แผลบาด",
    "Diabetic Wounds": "แผลเบาหวาน",
    "Laseration": "แผลฉีกขาด",
    "Laceration": "แผลฉีกขาด",
    "Normal": "ผิวหนังปกติ",
    "Pressure Wounds": "แผลกดทับ",
    "Surgical Wounds": "แผลผ่าตัด",
    "Venous Wounds": "แผลจากหลอดเลือดดำ",
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ภาพที่ชื่อขึ้นต้นด้วยคำเหล่านี้เป็นภาพกลับด้านของภาพต้นฉบับในโฟลเดอร์เดียวกัน
# คุณลักษณะที่โมดูลนี้ใช้เกือบทั้งหมดไม่เปลี่ยนเมื่อกลับด้านภาพ จึงถือเป็นภาพซ้ำ
# และถูกตัดออกตอนสุ่มภาพอ้างอิงเป็นค่าเริ่มต้น เพื่อให้ตัวอย่างที่สุ่มได้หลากหลายจริง
MIRROR_PREFIXES = ("mirrored_", "mirror_", "flipped_")


# ---------------------------------------------------------------- การถอดคุณลักษณะ


def _tissue_vector(hue_deg: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    แปลงภาพเป็นสัดส่วนชนิดเนื้อเยื่อ 6 ค่า
    ใช้ลำดับเงื่อนไขเดียวกับ wound_service._classify_pixel ทุกประการ
    เพื่อให้ผลจากสองโมดูลอธิบายด้วยกฎชุดเดียวกันได้
    ลำดับผลลัพธ์ granulation, slough, necrotic, periwound, background, other
    """
    labels = np.full(hue_deg.shape, -1, dtype=np.int8)

    def assign(mask: np.ndarray, code: int) -> None:
        target = mask & (labels < 0)
        labels[target] = code

    assign(v < 0.22, 2)                                   # necrotic
    assign((s < 0.18) & (v > 0.75), 4)                    # background
    assign(s < 0.15, 4)                                   # background

    red = (hue_deg < 12) | (hue_deg >= 345)
    assign(red & (s > 0.35), 0)                           # granulation
    assign(red, 3)                                        # periwound

    orange = (hue_deg >= 12) & (hue_deg < 45)
    assign(orange & (v > 0.55) & (s < 0.45), 3)           # periwound
    assign(orange, 0)                                     # granulation

    assign((hue_deg >= 45) & (hue_deg < 170), 1)          # slough
    assign(labels < 0, 5)                                 # other

    counts = np.bincount(labels.ravel(), minlength=6).astype(np.float64)
    total = counts.sum()
    return counts / total if total else counts


def _histograms(hue_rot: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """สร้าง histogram สามตัวต่อกัน แต่ละตัวถูก normalize ให้ผลรวมเป็น 1 แยกกัน"""
    parts = []
    for channel, bins in ((hue_rot, 24), (s, 12), (v, 12)):
        hist, _ = np.histogram(channel, bins=bins, range=(0.0, 1.0))
        total = hist.sum()
        parts.append(hist / total if total else hist.astype(np.float64))
    return np.concatenate(parts)


def extract_features(path: str | Path) -> Dict[str, List[float]]:
    """ถอดคุณลักษณะทั้งสี่กลุ่มจากไฟล์ภาพหนึ่งไฟล์"""
    with Image.open(path) as raw:
        img = raw.convert("RGB").resize((WORK_SIZE, WORK_SIZE), Image.BILINEAR)
        hsv = np.asarray(img.convert("HSV"), dtype=np.float64) / 255.0

    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    hue_rot = (h + HUE_ROTATION) % 1.0
    hue_deg = h * 360.0

    gx = np.diff(v, axis=1)[:-1, :]
    gy = np.diff(v, axis=0)[:, :-1]
    grad = np.sqrt(gx * gx + gy * gy)

    return {
        "histogram": _histograms(hue_rot, s, v).tolist(),
        "tissue": _tissue_vector(hue_deg, s, v).tolist(),
        "moments": [
            float(hue_rot.mean()), float(hue_rot.std()),
            float(s.mean()), float(s.std()),
            float(v.mean()), float(v.std()),
        ],
        "texture": [float(grad.mean()), float(grad.std())],
    }


# ---------------------------------------------------------------- การวัดความคล้าย


def _histogram_intersection(a: np.ndarray, b: np.ndarray) -> float:
    """
    histogram ถูกต่อกันสามช่วง (24, 12, 12) แต่ละช่วงมีผลรวมเป็น 1
    จึงหา intersection แยกช่วงแล้วเฉลี่ย ไม่เช่นนั้นช่วง hue จะถูกให้น้ำหนักเกินจริง
    """
    spans = ((0, 24), (24, 36), (36, 48))
    return float(np.mean([np.minimum(a[i:j], b[i:j]).sum() for i, j in spans]))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), 0.0, 1.0))


def _distance_score(a: np.ndarray, b: np.ndarray, scale: float) -> float:
    """แปลงระยะห่างเป็นคะแนนความคล้ายในช่วง 0 ถึง 1 ด้วยฟังก์ชันเอกซ์โพเนนเชียล"""
    return float(math.exp(-float(np.linalg.norm(a - b)) / scale))


def similarity(f1: Dict[str, Sequence[float]], f2: Dict[str, Sequence[float]]) -> Dict[str, float]:
    """คืนคะแนนความคล้ายรายกลุ่มและคะแนนรวมถ่วงน้ำหนัก ทุกค่าอยู่ในช่วง 0 ถึง 1"""
    parts = {
        "histogram": _histogram_intersection(
            np.asarray(f1["histogram"]), np.asarray(f2["histogram"])),
        "tissue": _cosine(np.asarray(f1["tissue"]), np.asarray(f2["tissue"])),
        "moments": _distance_score(
            np.asarray(f1["moments"]), np.asarray(f2["moments"]), scale=0.45),
        "texture": _distance_score(
            np.asarray(f1["texture"]), np.asarray(f2["texture"]), scale=0.12),
    }
    parts["total"] = sum(parts[k] * WEIGHTS[k] for k in WEIGHTS)
    return parts


# ---------------------------------------------------------------- คลังภาพอ้างอิง


class ReferenceLibrary:
    """
    อ่านโฟลเดอร์ภาพอ้างอิงที่จัดกลุ่มไว้แล้ว หนึ่งโฟลเดอร์ย่อยคือหนึ่งประเภทแผล
    คุณลักษณะของแต่ละภาพถูก cache ลงไฟล์ JSON โดยผูกกับ mtime และขนาดไฟล์
    ถ้าไฟล์ต้นทางถูกแก้หรือถูกแทนที่ ระบบจะคำนวณใหม่ให้เองโดยไม่ต้องล้าง cache
    """

    def __init__(self, root: str | Path, cache_path: Optional[str | Path] = None,
                 include_mirrored: bool = False):
        self.root = Path(root)
        self.include_mirrored = include_mirrored
        self.cache_path = Path(cache_path) if cache_path else self.root / "_feature_cache.json"
        self._cache: Dict[str, dict] = {}
        self._dirty = False
        self._load_cache()

    # ---------------- cache

    def _load_cache(self) -> None:
        try:
            if self.cache_path.exists():
                self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception:
            self._cache = {}

    def save_cache(self) -> None:
        if not self._dirty:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps(self._cache, ensure_ascii=False), encoding="utf-8")
            self._dirty = False
        except Exception:
            pass

    def features_for(self, path: Path) -> Optional[Dict[str, List[float]]]:
        key = str(path.relative_to(self.root)) if path.is_relative_to(self.root) else str(path)
        try:
            stat = path.stat()
        except OSError:
            return None
        stamp = [int(stat.st_mtime), stat.st_size]

        hit = self._cache.get(key)
        if hit and hit.get("stamp") == stamp:
            return hit["features"]

        try:
            feats = extract_features(path)
        except Exception:
            return None
        self._cache[key] = {"stamp": stamp, "features": feats}
        self._dirty = True
        return feats

    # ---------------- โครงสร้างโฟลเดอร์

    def categories(self) -> List[str]:
        if not self.root.is_dir():
            return []
        return sorted(d.name for d in self.root.iterdir()
                      if d.is_dir() and not d.name.startswith(("_", ".")))

    def images_in(self, category: str) -> List[Path]:
        folder = self.root / category
        if not folder.is_dir():
            return []
        files = [p for p in sorted(folder.iterdir())
                 if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
        if not self.include_mirrored:
            originals = [p for p in files if not p.name.lower().startswith(MIRROR_PREFIXES)]
            # ถ้าโฟลเดอร์มีแต่ภาพกลับด้าน ให้ใช้ทั้งหมดแทนการคืนค่าว่าง
            if originals:
                return originals
        return files

    def inventory(self) -> Dict[str, int]:
        return {c: len(self.images_in(c)) for c in self.categories()}


# ---------------------------------------------------------------- การจำแนก


def _softmax_percent(scores: Dict[str, float], temperature: float) -> Dict[str, float]:
    if not scores:
        return {}
    top = max(scores.values())
    exp = {k: math.exp((v - top) / temperature) for k, v in scores.items()}
    total = sum(exp.values()) or 1.0
    return {k: 100.0 * v / total for k, v in exp.items()}


def classify(
    image_path: str | Path,
    library: ReferenceLibrary,
    sample_k: int = DEFAULT_SAMPLE_K,
    top_m: int = DEFAULT_TOP_M,
    temperature: float = DEFAULT_TEMPERATURE,
    seed: Optional[int] = None,
    exclude: Optional[set] = None,
) -> dict:
    """
    จำแนกประเภทของภาพแผลหนึ่งภาพ

    sample_k   จำนวนภาพอ้างอิงที่สุ่มต่อโฟลเดอร์ ตั้ง 0 เพื่อใช้ทุกภาพในโฟลเดอร์
    top_m      ใช้ค่าเฉลี่ยของคะแนนสูงสุดกี่อันดับจากที่สุ่มมา
    seed       กำหนดเพื่อให้การสุ่มซ้ำได้ผลเดิม เหมาะกับการสาธิตและการทดสอบ
    exclude    เซ็ตของ Path ที่ห้ามใช้เป็นภาพอ้างอิง ใช้ตอนประเมินเพื่อกันภาพทดสอบรั่ว
    """
    query = extract_features(image_path)
    rng = random.Random(seed)
    exclude = exclude or set()

    raw: Dict[str, float] = {}
    evidence: Dict[str, List[dict]] = {}

    for category in library.categories():
        pool = [p for p in library.images_in(category) if p not in exclude]
        if not pool:
            continue
        picked = pool if sample_k <= 0 or sample_k >= len(pool) else rng.sample(pool, sample_k)

        scored: List[Tuple[float, Path, Dict[str, float]]] = []
        for ref in picked:
            feats = library.features_for(ref)
            if feats is None:
                continue
            parts = similarity(query, feats)
            scored.append((parts["total"], ref, parts))

        if not scored:
            continue

        scored.sort(key=lambda t: t[0], reverse=True)
        keep = scored[:top_m] if top_m > 0 else scored
        raw[category] = sum(t[0] for t in keep) / len(keep)
        evidence[category] = [
            {
                "file": t[1].name,
                "score": round(t[0], 4),
                "breakdown": {k: round(v, 4) for k, v in t[2].items() if k != "total"},
            }
            for t in scored
        ]

    library.save_cache()

    if not raw:
        return {
            "ok": False,
            "reason": "ไม่พบภาพอ้างอิงที่ใช้เปรียบเทียบได้ ตรวจสอบโฟลเดอร์คลังภาพอ้างอิง",
            "ranking": [],
        }

    percent = _softmax_percent(raw, temperature)
    ranking = sorted(
        (
            {
                "category": c,
                "category_th": CATEGORY_TH.get(c, c),
                "percent": round(percent[c], 2),
                "raw_score": round(raw[c], 4),
                "compared_with": evidence.get(c, []),
            }
            for c in raw
        ),
        key=lambda d: d["percent"],
        reverse=True,
    )

    gap = ranking[0]["percent"] - (ranking[1]["percent"] if len(ranking) > 1 else 0.0)
    out_of_range = ranking[0]["raw_score"] < MIN_RAW_SCORE
    confident = gap >= LOW_CONFIDENCE_GAP and not out_of_range

    if out_of_range:
        note = ("ภาพนี้ไม่คล้ายภาพอ้างอิงใดในคลังมากพอ อาจเป็นภาพเบลอ ภาพมืด "
                "หรือถ่ายไม่ตรงบริเวณแผล ให้ถ่ายใหม่หรือให้เจ้าหน้าที่ระบุประเภทเอง")
    elif not confident:
        note = ("คะแนนอันดับหนึ่งกับอันดับสองใกล้กันมาก ระบบยังแยกประเภทไม่ชัด "
                "ให้เจ้าหน้าที่เป็นผู้ระบุประเภทแผลเอง")
    else:
        note = ("ผลนี้เป็นการเทียบภาพกับคลังภาพอ้างอิง ไม่ใช่การวินิจฉัย "
                "เจ้าหน้าที่เป็นผู้ตัดสินใจสุดท้ายเสมอ")

    return {
        "ok": True,
        "top": ranking[0]["category"],
        "top_th": ranking[0]["category_th"],
        "percent": ranking[0]["percent"],
        "confidence_gap": round(gap, 2),
        "confident": confident,
        "out_of_range": out_of_range,
        "note": note,
        "method": "instance-based comparison with hand-crafted HSV features",
        "sample_k": sample_k,
        "ranking": ranking,
    }


# ---------------------------------------------------------------- ตัวช่วยระดับแอป

_LIBRARY: Optional[ReferenceLibrary] = None


def get_library(root: Optional[str | Path] = None) -> ReferenceLibrary:
    """คืน ReferenceLibrary ที่ใช้ร่วมกันทั้งแอป สร้างครั้งเดียวแล้วใช้ซ้ำ"""
    global _LIBRARY
    if _LIBRARY is None or (root is not None and Path(root) != _LIBRARY.root):
        if root is None:
            here = Path(__file__).resolve().parents[2]
            root = os.environ.get("WOUND_REFERENCE_DIR") or here / "data" / "Image_Classifier"
        _LIBRARY = ReferenceLibrary(root)
    return _LIBRARY


def classify_image(image_path: str | Path, **kwargs) -> dict:
    """ทางเข้าที่ให้ส่วนอื่นของระบบเรียกใช้ ใช้คลังภาพอ้างอิงที่ตั้งไว้เป็นค่าเริ่มต้น"""
    lib = get_library()
    if not lib.root.is_dir():
        return {
            "ok": False,
            "reason": f"ไม่พบโฟลเดอร์คลังภาพอ้างอิงที่ {lib.root}",
            "ranking": [],
        }
    return classify(image_path, lib, **kwargs)


def warm_cache(root: Optional[str | Path] = None, verbose: bool = True) -> dict:
    """คำนวณคุณลักษณะของทุกภาพในคลังไว้ล่วงหน้า ทำครั้งเดียวหลังเพิ่มหรือแก้ภาพ"""
    lib = get_library(root)
    started = time.time()
    done = failed = 0
    for category in lib.categories():
        for path in lib.images_in(category):
            if lib.features_for(path) is None:
                failed += 1
            else:
                done += 1
        if verbose:
            print(f"  {category:<20} เสร็จแล้ว {done} ภาพ")
    lib.save_cache()
    return {
        "images": done,
        "failed": failed,
        "seconds": round(time.time() - started, 1),
        "cache": str(lib.cache_path),
    }
