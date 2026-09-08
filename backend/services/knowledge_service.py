"""
knowledge_service.py
ค้นคืนข้อความจากคู่มือและเอกสารวิชาการที่เก็บไว้ในเครื่อง แบบทำงานออฟไลน์เต็มรูปแบบ

บทบาทในระบบ
-----------
เป็นชั้นค้นคืนหลักฐานฝั่งออฟไลน์ คู่กับ pubmed_service ที่เป็นฝั่งออนไลน์
เมื่อไม่มีอินเทอร์เน็ต ซึ่งเป็นสภาพปกติของ รพ.สต. หลายแห่ง ระบบยังต้องอ้างอิง
คู่มือได้ตามปกติ โมดูลนี้จึงถูกออกแบบให้ไม่เรียกเครือข่ายเลยแม้แต่ครั้งเดียว
และใช้เฉพาะไลบรารีมาตรฐานของ Python ไม่ต้องติดตั้งอะไรเพิ่ม

วิธีค้นคืน
---------
ใช้ BM25 ซึ่งเป็นสูตรจัดอันดับเชิงสถิติที่เขียนไว้ตายตัว ไม่มีการเรียนรู้น้ำหนัก
ไม่ต้องใช้โมเดลภาษา ไม่ต้องดาวน์โหลด embedding และอธิบายได้ว่าเอกสารถูกเลือก
เพราะคำใดตรงกันบ้าง เหมาะกับบริบทที่ต้องตรวจสอบย้อนกลับได้ทุกคำแนะนำ

การตัดคำภาษาไทย
--------------
ถ้าเครื่องมี pythainlp จะใช้ตัวตัดคำ newmm ซึ่งให้คำที่ถูกต้องกว่า
ถ้าไม่มี จะถอยไปใช้ n-gram ระดับตัวอักษร ซึ่งเป็นวิธีมาตรฐานสำหรับการค้นคืน
ภาษาไทยที่ไม่มีตัวตัดคำ และทนต่อความผิดพลาดของการแตกข้อความจาก PDF ได้ดีกว่า
ดัชนีจะบันทึกไว้ว่าสร้างด้วยตัวตัดคำใด ถ้าตอนค้นใช้คนละตัวระบบจะเตือนให้สร้างใหม่

ข้อจำกัด
-------
ระบบคืนข้อความจากคู่มือตามคำที่ตรงกัน ไม่ได้ตีความหรือสรุปแทนเจ้าหน้าที่
เอกสารที่ได้จาก OCR จะถูกกำกับว่าคุณภาพต่ำกว่า เพราะอาจมีตัวอักษรเพี้ยน
เจ้าหน้าที่ต้องเปิดเอกสารต้นทางหน้าที่ระบุเพื่อยืนยันก่อนใช้ทุกครั้ง
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

# ---------------------------------------------------------------- ค่าตั้งต้น

BM25_K1 = 1.2
BM25_B = 0.75

# ความยาว n-gram ที่ใช้เมื่อไม่มีตัวตัดคำภาษาไทย
NGRAM_N = 3

THAI_RANGE = "฀-๿"
_THAI_RUN = re.compile(f"[{THAI_RANGE}]+")
_LATIN_RUN = re.compile(r"[A-Za-z][A-Za-z0-9\-]*|\d+(?:\.\d+)?")
_SPACE_IN_THAI = re.compile(f"(?<=[{THAI_RANGE}])[ \\t]+(?=[{THAI_RANGE}])")
_WS = re.compile(r"[ \t ]+")

DEFAULT_INDEX = Path(__file__).resolve().parents[2] / "data" / "knowledge_index.json"

# คำที่พบบ่อยจนไม่ช่วยแยกเอกสาร ตัดออกเพื่อลดสัญญาณรบกวน
STOP_TOKENS = {
    "และ", "หรือ", "ของ", "ที่", "ใน", "การ", "ความ", "เป็น", "ได้", "ให้",
    "กับ", "จาก", "โดย", "แต่", "ไม่", "มี", "จะ", "ซึ่ง", "ต้อง", "นี้",
    "the", "and", "or", "of", "in", "to", "is", "for", "with", "a", "an",
}


# ---------------------------------------------------------------- การเตรียมข้อความ


def normalize_text(text: str) -> str:
    """ทำความสะอาดข้อความสำหรับแสดงผล แก้ปัญหาที่เกิดจากการแตกข้อความจาก PDF"""
    text = unicodedata.normalize("NFC", text)
    for junk in ("​", "﻿", "\xad"):
        text = text.replace(junk, "")
    text = _WS.sub(" ", text)
    return text.strip()


def _tokenize_stream(text: str) -> str:
    """
    เตรียมข้อความสำหรับตัดคำ
    ลบช่องว่างที่คั่นกลางอักษรไทย เพราะการแตกข้อความจาก PDF มักแทรกช่องว่างผิดที่
    ทำให้คำเดียวกันในคำค้นกับในเอกสารกลายเป็นคนละรูป
    """
    return _SPACE_IN_THAI.sub("", normalize_text(text))


_PYTHAINLP = None


def _pythainlp_tokenizer():
    """โหลด pythainlp แบบครั้งเดียว คืน None ถ้าเครื่องไม่มี"""
    global _PYTHAINLP
    if _PYTHAINLP is None:
        try:
            from pythainlp.tokenize import word_tokenize  # type: ignore
            _PYTHAINLP = word_tokenize
        except Exception:
            _PYTHAINLP = False
    return _PYTHAINLP or None


def available_tokenizer() -> str:
    """ชื่อตัวตัดคำที่เครื่องนี้ใช้ได้จริง"""
    return "pythainlp-newmm" if _pythainlp_tokenizer() else f"ngram{NGRAM_N}"


def tokenize(text: str, mode: Optional[str] = None) -> List[str]:
    """
    ตัดข้อความเป็น token สำหรับสร้างดัชนีและค้นคืน
    mode  ระบุชื่อตัวตัดคำเพื่อบังคับให้ตรงกับที่ใช้ตอนสร้างดัชนี
    """
    mode = mode or available_tokenizer()
    stream = _tokenize_stream(text)
    tokens: List[str] = []

    # ส่วนที่เป็นอังกฤษและตัวเลข ตัดด้วยขอบเขตคำตามปกติ
    tokens.extend(m.group(0).lower() for m in _LATIN_RUN.finditer(stream))

    thai_runs = _THAI_RUN.findall(stream)
    if mode.startswith("pythainlp"):
        wt = _pythainlp_tokenizer()
        if wt is not None:
            for run in thai_runs:
                tokens.extend(w for w in wt(run, engine="newmm") if w.strip())
        else:
            mode = f"ngram{NGRAM_N}"

    if not mode.startswith("pythainlp"):
        n = NGRAM_N
        for run in thai_runs:
            if len(run) <= n:
                tokens.append(run)
            else:
                tokens.extend(run[i:i + n] for i in range(len(run) - n + 1))

    return [t for t in tokens if t and t not in STOP_TOKENS]


# ---------------------------------------------------------------- ตัวค้นคืน


class KnowledgeIndex:
    """
    ตัวค้นคืนที่อ่านดัชนีซึ่งสร้างไว้ล่วงหน้าด้วย tools/build_knowledge_index.py
    ใช้เฉพาะไลบรารีมาตรฐาน ไม่เรียกเครือข่าย และไม่ต้องมีไลบรารีอ่าน PDF ตอนใช้งาน
    """

    def __init__(self, path: str | Path = DEFAULT_INDEX):
        self.path = Path(path)
        self.loaded = False
        self.error: Optional[str] = None
        self.tokenizer = f"ngram{NGRAM_N}"
        self.docs: List[dict] = []
        self.chunks: List[dict] = []
        self.postings: Dict[str, List[List[int]]] = {}
        self.doc_freq: Dict[str, int] = {}
        self.avgdl = 1.0
        self.built_at = ""
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self.error = (f"ยังไม่มีไฟล์ดัชนีที่ {self.path} "
                          f"ให้รัน python tools/build_knowledge_index.py ก่อน")
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.error = f"อ่านไฟล์ดัชนีไม่สำเร็จ {exc}"
            return

        self.tokenizer = data.get("tokenizer", f"ngram{NGRAM_N}")
        self.docs = data.get("docs", [])
        self.chunks = data.get("chunks", [])
        self.postings = data.get("postings", {})
        self.doc_freq = data.get("doc_freq", {})
        self.avgdl = data.get("avgdl", 1.0) or 1.0
        self.built_at = data.get("built_at", "")
        self.loaded = bool(self.chunks)
        if not self.loaded:
            self.error = "ไฟล์ดัชนีไม่มีข้อความใดเลย ให้สร้างใหม่"

    # ---------------- สถานะ

    def status(self) -> dict:
        mismatch = self.loaded and self.tokenizer != available_tokenizer()
        return {
            "ready": self.loaded,
            "error": self.error,
            "index_path": str(self.path),
            "built_at": self.built_at,
            "documents": len(self.docs),
            "passages": len(self.chunks),
            "tokenizer_in_index": self.tokenizer,
            "tokenizer_available": available_tokenizer(),
            "tokenizer_mismatch": mismatch,
            "warning": (
                "ดัชนีถูกสร้างด้วยตัวตัดคำคนละตัวกับที่เครื่องนี้มี ผลค้นอาจแย่ลง "
                "ให้สร้างดัชนีใหม่บนเครื่องนี้" if mismatch else None
            ),
            "offline": True,
        }

    # ---------------- การค้น

    @staticmethod
    def _highlights(query: str, text: str, limit: int = 6) -> List[str]:
        """
        หาวลีที่ปรากฏร่วมกันระหว่างคำค้นกับข้อความที่ค้นเจอ เพื่อให้เจ้าหน้าที่เห็นว่า
        ระบบเลือกข้อความนี้เพราะอะไร การแสดง n-gram ดิบอ่านไม่รู้เรื่องสำหรับผู้ใช้จริง
        """
        q = _SPACE_IN_THAI.sub("", normalize_text(query)).lower()
        body = _SPACE_IN_THAI.sub("", normalize_text(text)).lower()
        found: List[str] = []
        i = 0
        while i < len(q):
            best = ""
            for j in range(len(q), i + 2, -1):
                piece = q[i:j].strip()
                if len(piece) >= 3 and piece in body:
                    best = piece
                    break
            if best:
                if best not in found:
                    found.append(best)
                i += len(best)
            else:
                i += 1
            if len(found) >= limit:
                break
        return found

    def _idf(self, term: str) -> float:
        n = len(self.chunks)
        df = self.doc_freq.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def search(
        self,
        query: str,
        k: int = 5,
        topics: Optional[Sequence[str]] = None,
        min_score: float = 0.0,
        per_document: int = 2,
    ) -> List[dict]:
        """
        ค้นข้อความที่เกี่ยวข้องที่สุด

        topics        จำกัดเฉพาะเอกสารที่มีป้ายหัวข้อนี้ เช่น ["wound"] หรือ ["medicine"]
        per_document  จำกัดจำนวนผลจากเอกสารเดียวกัน เพื่อไม่ให้เอกสารเดียวกินผลทั้งหมด
        """
        if not self.loaded:
            return []

        terms = tokenize(query, mode=self.tokenizer)
        if not terms:
            return []

        allowed: Optional[set] = None
        if topics:
            wanted = {t.lower() for t in topics}
            allowed = {i for i, d in enumerate(self.docs)
                       if wanted & {x.lower() for x in d.get("topics", [])}}
            if not allowed:
                allowed = None  # ไม่มีเอกสารตรงหัวข้อ ให้ค้นทั้งหมดแทนการคืนค่าว่าง

        scores: Dict[int, float] = {}
        matched: Dict[int, set] = {}
        seen_terms = set()

        for term in terms:
            if term in seen_terms:
                continue
            seen_terms.add(term)
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self._idf(term)
            if idf <= 0:
                continue
            for chunk_id, tf in postings:
                if allowed is not None and self.chunks[chunk_id]["doc"] not in allowed:
                    continue
                dl = self.chunks[chunk_id]["len"] or 1
                denom = tf + BM25_K1 * (1 - BM25_B + BM25_B * dl / self.avgdl)
                scores[chunk_id] = scores.get(chunk_id, 0.0) + idf * tf * (BM25_K1 + 1) / denom
                matched.setdefault(chunk_id, set()).add(term)

        if not scores:
            return []

        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_raw = ordered[0][1] or 1.0

        results: List[dict] = []
        used_per_doc: Dict[int, int] = {}
        for chunk_id, score in ordered:
            if score < min_score:
                break
            chunk = self.chunks[chunk_id]
            doc_id = chunk["doc"]
            if used_per_doc.get(doc_id, 0) >= per_document:
                continue
            used_per_doc[doc_id] = used_per_doc.get(doc_id, 0) + 1
            doc = self.docs[doc_id]
            results.append({
                "text": chunk["text"],
                "score": round(score, 3),
                "relevance": round(100.0 * score / top_raw, 1),
                "source": doc.get("title") or doc.get("file"),
                "file": doc.get("file"),
                "page": chunk.get("page"),
                "topics": doc.get("topics", []),
                "quality": doc.get("quality", "text"),
                "quality_note": (
                    "เอกสารนี้ได้ข้อความจาก OCR อาจมีตัวอักษรคลาดเคลื่อน "
                    "ต้องเปิดต้นฉบับหน้าที่ระบุเพื่อยืนยัน"
                    if doc.get("quality") == "ocr" else None
                ),
                "matched_phrases": self._highlights(query, chunk["text"]),
                "matched_terms": sorted(matched.get(chunk_id, set()))[:12],
            })
            if len(results) >= k:
                break
        return results


# ---------------------------------------------------------------- ตัวช่วยระดับแอป

_INDEX: Optional[KnowledgeIndex] = None


def get_index(path: Optional[str | Path] = None) -> KnowledgeIndex:
    global _INDEX
    if _INDEX is None or (path is not None and Path(path) != _INDEX.path):
        _INDEX = KnowledgeIndex(path or DEFAULT_INDEX)
    return _INDEX


def reload_index(path: Optional[str | Path] = None) -> dict:
    """โหลดดัชนีใหม่หลังสร้างไฟล์ใหม่ โดยไม่ต้องรีสตาร์ตเซิร์ฟเวอร์"""
    global _INDEX
    _INDEX = None
    return get_index(path).status()


def search(query: str, **kwargs) -> List[dict]:
    return get_index().search(query, **kwargs)


def status() -> dict:
    return get_index().status()


# ---------------------------------------------------------------- ใช้ร่วมกับสองโหมด

# คำที่ใช้ขยายคำค้นให้ตรงกับภาษาที่ใช้ในคู่มือมากขึ้น
# คู่มือมักใช้ศัพท์วิชาการ แต่เจ้าหน้าที่พิมพ์ภาษาพูด จึงต้องเชื่อมสองอย่างเข้าหากัน
QUERY_EXPANSION = {
    "แผลกดทับ": ["pressure ulcer", "แผลกดทับ", "การป้องกันแผลกดทับ"],
    "แผลเบาหวาน": ["diabetic foot", "แผลที่เท้า", "เบาหวาน"],
    "แผลไหม้": ["burn", "แผลไฟไหม้", "น้ำร้อนลวก"],
    "แผลถลอก": ["abrasion", "แผลถลอก", "ทำความสะอาดแผล"],
    "แผลบาด": ["laceration", "แผลฉีกขาด", "การเย็บแผล"],
    "แผลผ่าตัด": ["surgical wound", "แผลผ่าตัด", "การดูแลแผลหลังผ่าตัด"],
    "ติดเชื้อ": ["infection", "การติดเชื้อ", "หนอง"],
    "ห้ามเลือด": ["hemostasis", "การห้ามเลือด", "กดห้ามเลือด"],
}

# แผนที่จากชื่อคลาสที่ wound_classifier_service จำแนกได้ ไปยังคำค้นในคู่มือ
WOUND_CLASS_QUERY = {
    "Abrasions": "แผลถลอก ทำความสะอาดแผล",
    "Bruises": "แผลฟกช้ำ ห้อเลือด ประคบ",
    "Burns": "แผลไหม้ น้ำร้อนลวก การปฐมพยาบาล",
    "Cut": "แผลบาด ห้ามเลือด ทำแผล",
    "Diabetic Wounds": "แผลเบาหวานที่เท้า การดูแลแผลเรื้อรัง",
    "Laseration": "แผลฉีกขาด การเย็บแผล ทำแผล",
    "Laceration": "แผลฉีกขาด การเย็บแผล ทำแผล",
    "Normal": "การดูแลผิวหนัง ป้องกันการเกิดแผล",
    "Pressure Wounds": "แผลกดทับ การป้องกันแผลกดทับ พลิกตะแคงตัว",
    "Surgical Wounds": "แผลผ่าตัด การดูแลแผลหลังผ่าตัด",
    "Venous Wounds": "แผลหลอดเลือดดำ แผลเรื้อรังที่ขา การพันผ้ายืด",
}


def expand_query(query: str) -> str:
    """เติมคำพ้องความหมายเพื่อให้คำค้นภาษาพูดไปตรงกับศัพท์ในคู่มือ"""
    extra: List[str] = []
    for key, values in QUERY_EXPANSION.items():
        if key in query:
            extra.extend(values)
    return query + (" " + " ".join(extra) if extra else "")


def for_wound_class(wound_class: str, extra_query: str = "", k: int = 4) -> List[dict]:
    """ดึงคู่มือการทำแผลที่ตรงกับประเภทแผลที่ระบบจำแนกได้"""
    base = WOUND_CLASS_QUERY.get(wound_class, wound_class)
    return search(expand_query(f"{base} {extra_query}".strip()), k=k, topics=["wound", "firstaid"])


def for_medicine(symptom_text: str, drug_names: Iterable[str] = (), k: int = 4) -> List[dict]:
    """
    ดึงข้อความอ้างอิงฝั่งการจ่ายยาจากคู่มือที่มีในเครื่อง

    ต้องมีเอกสารอ้างอิงด้านยาอยู่ในโฟลเดอร์คู่มือก่อน เช่น บัญชียาหลักแห่งชาติ
    หรือคู่มือการใช้ยาสำหรับหน่วยบริการปฐมภูมิ ถ้ายังไม่มี ระบบจะคืนค่าว่าง
    แล้วให้ส่วนที่เรียกใช้แจ้งเจ้าหน้าที่ว่ายังไม่มีหลักฐานออฟไลน์รองรับ
    """
    query = expand_query(f"{symptom_text} {' '.join(drug_names)}".strip())
    return search(query, k=k, topics=["medicine"])
