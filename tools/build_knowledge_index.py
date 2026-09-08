"""
build_knowledge_index.py
สร้างดัชนีค้นคืนออฟไลน์จากไฟล์ PDF ในโฟลเดอร์คู่มือ

วิธีรัน (จากโฟลเดอร์หลักของโปรเจกต์)
    python tools/build_knowledge_index.py
    python tools/build_knowledge_index.py --ocr          เปิด OCR สำหรับ PDF ที่เป็นภาพสแกน
    python tools/build_knowledge_index.py --rebuild      ทิ้งข้อความที่เคยแตกไว้แล้วทำใหม่ทั้งหมด

สิ่งที่สคริปต์นี้ทำ
    1. อ่าน PDF ทุกไฟล์ในโฟลเดอร์คู่มือ รวมโฟลเดอร์ย่อย
    2. แตกข้อความทีละหน้า เก็บเป็นไฟล์ .txt ไว้ใน _extracted เพื่อไม่ต้องแตกซ้ำ
    3. ติดป้ายหัวข้อให้เอกสารโดยนับคำสำคัญ เช่น wound, medicine, firstaid
    4. ตัดข้อความเป็นท่อนพร้อมเลขหน้า แล้วสร้างสถิติ BM25
    5. เขียนไฟล์ดัชนีเดียวที่ระบบใช้ตอนทำงานจริงโดยไม่ต้องมีไลบรารีอ่าน PDF

หลังจากนี้ระบบตอนใช้งานจริงไม่ต้องใช้อินเทอร์เน็ตและไม่ต้องมี PDF ต้นฉบับก็ค้นได้
แต่ควรเก็บ PDF ไว้เพื่อให้เจ้าหน้าที่เปิดยืนยันหน้าที่ระบบอ้างถึงได้
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.knowledge_service import (  # noqa: E402
    available_tokenizer, normalize_text, tokenize,
)

CHUNK_CHARS = 450
CHUNK_OVERLAP = 90
MIN_CHUNK_CHARS = 80

# ป้ายหัวข้อกำหนดว่าเอกสารจะถูกใช้ตอบคำถามฝั่งไหน คำใดพบมากกว่าเกณฑ์จึงติดป้ายนั้น
TOPIC_KEYWORDS = {
    "wound": ["แผล", "บาดแผล", "ทำแผล", "ทําแผล", "กดทับ", "สมานแผล",
              "เนื้อตาย", "dressing", "wound", "ulcer"],
    "medicine": ["ขนาดยา", "ข้อบ่งใช้", "บัญชียา", "ผลข้างเคียง", "ยาปฏิชีวนะ",
                 "รับประทานยา", "มิลลิกรัม", "dose", "dosage", "drug", "antibiotic"],
    "firstaid": ["ปฐมพยาบาล", "ห้ามเลือด", "กู้ชีพ", "ฟื้นคืนชีพ", "cpr",
                 "first aid", "ช็อก"],
}
TOPIC_MIN_HITS = 8


# ---------------------------------------------------------------- การแตกข้อความ


def _pdftotext_pages(pdf: Path) -> Optional[List[str]]:
    """ใช้ pdftotext จากชุด poppler ให้ผลดีที่สุดกับ PDF ภาษาไทยที่มีชั้นข้อความ"""
    if not shutil.which("pdftotext"):
        return None
    try:
        out = subprocess.run(
            ["pdftotext", "-layout", str(pdf), "-"],
            capture_output=True, timeout=180,
        )
        if out.returncode != 0:
            return None
        return out.stdout.decode("utf-8", errors="replace").split("\f")
    except Exception:
        return None


def _pypdf_pages(pdf: Path) -> Optional[List[str]]:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        return None
    try:
        reader = PdfReader(str(pdf))
        return [(p.extract_text() or "") for p in reader.pages]
    except Exception:
        return None


def _ocr_pages(pdf: Path, dpi: int = 300) -> Optional[List[str]]:
    """
    OCR สำหรับ PDF ที่เป็นภาพสแกน ต้องมี pdftoppm และ tesseract พร้อมชุดภาษาไทย
    ข้อความที่ได้จะถูกกำกับว่าคุณภาพต่ำกว่า เพราะภาษาอังกฤษและตารางมักเพี้ยน
    """
    if not (shutil.which("pdftoppm") and shutil.which("tesseract")):
        return None
    import tempfile
    try:
        with tempfile.TemporaryDirectory() as tmp:
            stem = Path(tmp) / "p"
            subprocess.run(["pdftoppm", "-r", str(dpi), "-png", str(pdf), str(stem)],
                           capture_output=True, timeout=900, check=True)
            pages = []
            for img in sorted(Path(tmp).glob("p-*.png")):
                res = subprocess.run(
                    ["tesseract", str(img), "stdout", "-l", "tha+eng", "--psm", "3"],
                    capture_output=True, timeout=300,
                )
                pages.append(res.stdout.decode("utf-8", errors="replace"))
            return pages or None
    except Exception:
        return None


def extract_document(pdf: Path, cache_dir: Path, use_ocr: bool,
                     rebuild: bool) -> Tuple[List[str], str]:
    """
    คืนข้อความรายหน้าและระดับคุณภาพของแหล่ง
    quality เป็น text เมื่อได้จากชั้นข้อความใน PDF และเป็น ocr เมื่อได้จากการอ่านภาพ
    """
    cache = cache_dir / (pdf.stem + ".json")
    if cache.exists() and not rebuild:
        try:
            saved = json.loads(cache.read_text(encoding="utf-8"))
            if saved.get("mtime") == int(pdf.stat().st_mtime):
                return saved["pages"], saved.get("quality", "text")
        except Exception:
            pass

    pages = _pdftotext_pages(pdf) or _pypdf_pages(pdf) or []
    quality = "text"
    thai_chars = sum(len(re.findall(r"[฀-๿]", p)) for p in pages)

    # ถ้าแทบไม่มีตัวอักษรเลย แปลว่าเป็นภาพสแกน ต้องใช้ OCR
    if thai_chars < 200 * max(1, len(pages)) // 100:
        if use_ocr:
            print(f"    ไม่พบชั้นข้อความ กำลังทำ OCR (ใช้เวลาสักครู่)")
            ocr = _ocr_pages(pdf)
            if ocr:
                pages, quality = ocr, "ocr"
            else:
                print(f"    ทำ OCR ไม่ได้ ข้ามไฟล์นี้ ต้องติดตั้ง tesseract พร้อมภาษาไทย")
        else:
            print(f"    ไม่พบชั้นข้อความ ข้ามไฟล์นี้ ใช้ตัวเลือก --ocr เพื่อลองอ่านจากภาพ")

    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        cache.write_text(json.dumps(
            {"mtime": int(pdf.stat().st_mtime), "quality": quality, "pages": pages},
            ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    return pages, quality


# ---------------------------------------------------------------- การตัดท่อน


def chunk_page(text: str) -> List[str]:
    """ตัดข้อความหนึ่งหน้าเป็นท่อนที่ยาวพอให้มีบริบท แต่สั้นพอให้ตรงประเด็น"""
    text = normalize_text(text)
    if len(text) < MIN_CHUNK_CHARS:
        return []

    chunks, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_CHARS, len(text))
        if end < len(text):
            # ถอยไปหาจุดตัดที่เป็นช่องว่างหรือจุด เพื่อไม่ให้ตัดกลางคำ
            window = text[start:end]
            cut = max(window.rfind(" "), window.rfind("."), window.rfind("ๆ"))
            if cut > CHUNK_CHARS // 2:
                end = start + cut
        piece = text[start:end].strip()
        if len(piece) >= MIN_CHUNK_CHARS:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


def detect_topics(full_text: str) -> List[str]:
    low = full_text.lower()
    found = []
    for topic, words in TOPIC_KEYWORDS.items():
        hits = sum(low.count(w.lower()) for w in words)
        if hits >= TOPIC_MIN_HITS:
            found.append((topic, hits))
    if not found:
        return ["general"]
    found.sort(key=lambda t: t[1], reverse=True)
    return [t for t, _ in found]


def guess_title(pages: List[str], fallback: str) -> str:
    """ใช้บรรทัดแรกที่ยาวพอของหน้าแรกเป็นชื่อเอกสาร ถ้าหาไม่ได้ใช้ชื่อไฟล์"""
    for page in pages[:2]:
        for line in normalize_text(page).split("\n"):
            line = line.strip()
            if 12 <= len(line) <= 120 and re.search(r"[฀-๿A-Za-z]", line):
                return line
    return fallback


# ---------------------------------------------------------------- ตัวหลัก


def main() -> int:
    root_dir = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(root_dir / "data" / "Guide_Book"))
    ap.add_argument("--out", default=str(root_dir / "data" / "knowledge_index.json"))
    ap.add_argument("--ocr", action="store_true", help="อ่าน PDF ที่เป็นภาพสแกนด้วย OCR")
    ap.add_argument("--rebuild", action="store_true", help="แตกข้อความใหม่ทั้งหมด")
    ap.add_argument("--min-df", type=int, default=1,
                    help="ตัดคำที่พบในท่อนน้อยกว่านี้ออกจากดัชนี เพื่อลดขนาดไฟล์")
    args = ap.parse_args()

    source = Path(args.source)
    if not source.is_dir():
        print(f"ไม่พบโฟลเดอร์คู่มือที่ {source}")
        return 1

    pdfs = sorted(p for p in source.rglob("*.pdf") if not p.name.startswith("."))
    if not pdfs:
        print(f"ไม่พบไฟล์ PDF ใน {source}")
        return 1

    tokenizer = available_tokenizer()
    print(f"โฟลเดอร์คู่มือ  {source}")
    print(f"ตัวตัดคำ        {tokenizer}")
    if tokenizer.startswith("ngram"):
        print("                (ถ้าติดตั้ง pythainlp จะได้ผลค้นที่แม่นขึ้น)")
    print(f"พบเอกสาร {len(pdfs)} ไฟล์\n")

    cache_dir = source / "_extracted"
    docs: List[dict] = []
    chunks: List[dict] = []
    started = time.time()

    for pdf in pdfs:
        rel = pdf.relative_to(source)
        print(f"  {rel}")
        pages, quality = extract_document(pdf, cache_dir, args.ocr, args.rebuild)
        if not pages:
            print("    ข้าม ไม่ได้ข้อความ")
            continue

        full = "\n".join(pages)
        topics = detect_topics(full)
        # โฟลเดอร์ย่อยมีสิทธิ์กำหนดหัวข้อเหนือการเดาจากเนื้อหา
        parent = rel.parent.name.lower()
        if parent and parent not in (".", "_extracted") and parent in TOPIC_KEYWORDS:
            topics = [parent] + [t for t in topics if t != parent]

        doc_id = len(docs)
        docs.append({
            "id": doc_id,
            "file": str(rel).replace("\\", "/"),
            "title": guess_title(pages, pdf.stem),
            "topics": topics,
            "quality": quality,
            "pages": len(pages),
        })

        added = 0
        for page_no, page_text in enumerate(pages, start=1):
            for piece in chunk_page(page_text):
                chunks.append({
                    "id": len(chunks),
                    "doc": doc_id,
                    "page": page_no,
                    "text": piece,
                    "len": 0,
                })
                added += 1
        print(f"    {len(pages)} หน้า  {added} ท่อน  หัวข้อ {','.join(topics)}  คุณภาพ {quality}")

    if not chunks:
        print("\nไม่ได้ข้อความจากเอกสารใดเลย")
        return 1

    print(f"\nกำลังสร้างสถิติ BM25 จาก {len(chunks)} ท่อน")
    postings: Dict[str, List[List[int]]] = {}
    doc_freq: Dict[str, int] = {}
    total_len = 0

    for chunk in chunks:
        tokens = tokenize(chunk["text"], mode=tokenizer)
        chunk["len"] = len(tokens)
        total_len += len(tokens)
        tf: Dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        for term, freq in tf.items():
            postings.setdefault(term, []).append([chunk["id"], freq])
            doc_freq[term] = doc_freq.get(term, 0) + 1

    if args.min_df > 1:
        drop = [t for t, df in doc_freq.items() if df < args.min_df]
        for t in drop:
            postings.pop(t, None)
            doc_freq.pop(t, None)
        print(f"  ตัดคำที่พบน้อยกว่า {args.min_df} ท่อนออก {len(drop)} คำ")

    index = {
        "version": 1,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tokenizer": tokenizer,
        "source": str(source),
        "avgdl": total_len / len(chunks),
        "docs": docs,
        "chunks": chunks,
        "doc_freq": doc_freq,
        "postings": postings,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")

    size_mb = out.stat().st_size / 1024 / 1024
    print(f"\nเขียนดัชนีแล้ว  {out}")
    print(f"  เอกสาร {len(docs)}  ท่อน {len(chunks)}  คำในดัชนี {len(doc_freq)}")
    print(f"  ขนาดไฟล์ {size_mb:.1f} MB   ใช้เวลา {time.time()-started:.1f} วินาที")
    print("\nระบบพร้อมค้นแบบออฟไลน์แล้ว ทดสอบด้วย")
    print("  python tools/ask_offline.py \"แผลกดทับระดับ 2 ดูแลอย่างไร\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
