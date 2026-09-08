"""
pubmed_service.py — ดึงหลักฐานงานวิจัยจาก NCBI E-utilities

ตามคำตอบข้อ 21 เรียกแบบเรียลไทม์ทุกครั้งที่มีคำถาม
ข้อควรทราบเรื่องความเร็ว: NCBI จำกัด 3 คำขอ/วินาที ถ้าไม่มี API key
และ 10 คำขอ/วินาที ถ้ามี key ส่วนแต่ละรอบต้องเรียกสองขั้นคือ esearch แล้วตามด้วย efetch
จึงใส่ cache ใน SQLite ไว้ 24 ชั่วโมง เพื่อไม่ให้คำถามซ้ำๆ ต้องรอใหม่ทุกครั้ง
และมี timeout สั้น เพื่อไม่ให้บทสนทนาค้างเมื่อ NCBI ตอบช้า

อ้างอิงรูปแบบ URL จากเอกสาร Entrez Programming Utilities:
  esearch.fcgi?db=pubmed&term=<query>
  efetch.fcgi?db=pubmed&id=<pmids>&rettype=abstract&retmode=xml
"""
import json
import time
import xml.etree.ElementTree as ET
from typing import Dict, List
from urllib.parse import urlencode

import requests

from config import (PUBMED_API_KEY, PUBMED_BASE_URL, PUBMED_CACHE_TTL,
                    PUBMED_EMAIL, PUBMED_ENABLED, PUBMED_MAX_RESULTS,
                    PUBMED_TIMEOUT, PUBMED_TOOL_NAME)
from database import get_conn

# คำที่ช่วยจำกัดผลลัพธ์ให้เป็นงานที่ใช้อ้างอิงเชิงปฏิบัติได้
QUERY_SUFFIX = ' AND (review[pt] OR guideline[pt] OR practice guideline[pt])'

# แปลคำอาการภาษาไทยเป็นศัพท์ค้นภาษาอังกฤษ
# ตารางนี้เป็นชุดตั้งต้นสำหรับ Prototype ควรขยายและให้บุคลากรทางการแพทย์ตรวจสอบ
TH_EN_TERMS = {
    "ไข้": "fever", "ปวดหัว": "headache", "ปวดศีรษะ": "headache",
    "ไอ": "cough", "เจ็บคอ": "sore throat pharyngitis",
    "ท้องเสีย": "acute diarrhea", "ถ่ายเหลว": "acute diarrhea",
    "อาเจียน": "vomiting", "คลื่นไส้": "nausea",
    "ผื่น": "skin rash", "คัน": "pruritus", "ลมพิษ": "urticaria",
    "แผล": "wound care", "แผลติดเชื้อ": "wound infection",
    "แผลเบาหวาน": "diabetic foot ulcer", "แผลกดทับ": "pressure ulcer",
    "แผลไฟไหม้": "burn wound", "เนื้อตาย": "wound necrosis debridement",
    "ไข้เลือดออก": "dengue fever", "ความดัน": "hypertension",
    "เบาหวาน": "diabetes mellitus primary care",
    "ปวดข้อ": "joint pain", "ปวดกล้ามเนื้อ": "myalgia",
    "แสบท้อง": "dyspepsia", "กรดไหลย้อน": "gastroesophageal reflux",
    "คัดจมูก": "allergic rhinitis", "น้ำมูก": "allergic rhinitis",
}


def _params(extra: Dict) -> str:
    base = {"tool": PUBMED_TOOL_NAME}
    if PUBMED_EMAIL:
        base["email"] = PUBMED_EMAIL
    if PUBMED_API_KEY:
        base["api_key"] = PUBMED_API_KEY
    base.update(extra)
    return urlencode(base)


def build_query(thai_text: str, extra_terms: str = "") -> str:
    """แปลงข้อความอาการภาษาไทยเป็น query ภาษาอังกฤษสำหรับ PubMed"""
    terms = []
    for th, en in TH_EN_TERMS.items():
        if th in thai_text and en not in terms:
            terms.append(en)
    if extra_terms:
        terms.append(extra_terms)
    if not terms:
        return ""
    return " AND ".join(f"({t})" for t in terms[:3])


def _cache_get(query: str):
    with get_conn() as conn:
        row = conn.execute("SELECT payload, fetched_at FROM pubmed_cache WHERE query = ?",
                           (query,)).fetchone()
    if row and (time.time() - row["fetched_at"]) < PUBMED_CACHE_TTL:
        return json.loads(row["payload"])
    return None


def _cache_set(query: str, payload):
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO pubmed_cache (query, payload, fetched_at) VALUES (?,?,?)",
            (query, json.dumps(payload, ensure_ascii=False), time.time()),
        )


def search(query: str, max_results: int = None) -> List[Dict]:
    """
    ค้น PubMed แล้วดึงบทคัดย่อ คืนรายการ [{pmid, title, abstract, journal, year, url}]
    ถ้าล้มเหลวจะคืนลิสต์ว่าง ไม่ทำให้บทสนทนาหลักพัง
    """
    if not PUBMED_ENABLED or not query:
        return []

    max_results = max_results or PUBMED_MAX_RESULTS
    cache_key = f"{query}|{max_results}"
    cached = _cache_get(cache_key)
    if cached is not None:
        for item in cached:
            item["from_cache"] = True
        return cached

    try:
        # ขั้นที่ 1 — esearch หา PMID
        url = f"{PUBMED_BASE_URL}/esearch.fcgi?" + _params({
            "db": "pubmed", "term": query + QUERY_SUFFIX,
            "retmax": max_results, "retmode": "json", "sort": "relevance",
        })
        r = requests.get(url, timeout=PUBMED_TIMEOUT)
        r.raise_for_status()
        pmids = r.json().get("esearchresult", {}).get("idlist", [])

        if not pmids:
            # ผ่อนเงื่อนไข ลองใหม่โดยไม่จำกัดประเภทบทความ
            url = f"{PUBMED_BASE_URL}/esearch.fcgi?" + _params({
                "db": "pubmed", "term": query, "retmax": max_results,
                "retmode": "json", "sort": "relevance",
            })
            r = requests.get(url, timeout=PUBMED_TIMEOUT)
            r.raise_for_status()
            pmids = r.json().get("esearchresult", {}).get("idlist", [])

        if not pmids:
            _cache_set(cache_key, [])
            return []

        # ขั้นที่ 2 — efetch ดึงบทคัดย่อ
        url = f"{PUBMED_BASE_URL}/efetch.fcgi?" + _params({
            "db": "pubmed", "id": ",".join(pmids),
            "rettype": "abstract", "retmode": "xml",
        })
        r = requests.get(url, timeout=PUBMED_TIMEOUT)
        r.raise_for_status()
        results = _parse_efetch(r.text)
        _cache_set(cache_key, results)
        return results

    except Exception:  # noqa: BLE001
        # ตามหลักการออกแบบ: PubMed เป็นชั้นเสริม ถ้าล้มเหลวต้องไม่ทำให้ระบบหลักหยุด
        return []


def _parse_efetch(xml_text: str) -> List[Dict]:
    out = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return out

    for art in root.findall(".//PubmedArticle"):
        pmid_el = art.find(".//PMID")
        title_el = art.find(".//ArticleTitle")
        journal_el = art.find(".//Journal/Title")
        year_el = art.find(".//JournalIssue/PubDate/Year")

        abstract_parts = []
        for ab in art.findall(".//Abstract/AbstractText"):
            label = ab.get("Label")
            text = "".join(ab.itertext()).strip()
            abstract_parts.append(f"{label}: {text}" if label else text)

        pmid = pmid_el.text if pmid_el is not None else ""
        out.append({
            "pmid": pmid,
            "title": "".join(title_el.itertext()).strip() if title_el is not None else "(ไม่มีชื่อเรื่อง)",
            "abstract": " ".join(abstract_parts)[:1500],
            "journal": journal_el.text if journal_el is not None else "",
            "year": year_el.text if year_el is not None else "",
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
            "from_cache": False,
        })
    return out


def search_for_symptoms(thai_text: str) -> List[Dict]:
    q = build_query(thai_text)
    return search(q) if q else []


def search_for_wound(thai_note: str, tissue_hint: str = "") -> List[Dict]:
    q = build_query(thai_note, extra_terms=tissue_hint or "wound care primary health care")
    return search(q) if q else search("wound care primary health care")


def status() -> Dict:
    return {
        "enabled": PUBMED_ENABLED,
        "has_api_key": bool(PUBMED_API_KEY),
        "rate_limit": "10 คำขอ/วินาที" if PUBMED_API_KEY else "3 คำขอ/วินาที (ไม่มี API key)",
        "cache_ttl_hours": round(PUBMED_CACHE_TTL / 3600, 1),
        "timeout_seconds": PUBMED_TIMEOUT,
    }
