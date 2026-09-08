"""
ask_offline.py
ทดสอบการค้นคืนจากคู่มือแบบออฟไลน์ด้วยคำถามภาษาไทย

    python tools/ask_offline.py "แผลกดทับระดับ 2 ดูแลอย่างไร"
    python tools/ask_offline.py "แผลไฟไหม้ปฐมพยาบาล" --topics firstaid --k 3
    python tools/ask_offline.py --status
    python tools/ask_offline.py --wound-class "Pressure Wounds"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services import knowledge_service as ks  # noqa: E402


def show(results, query_label: str) -> None:
    print(f"\nคำค้น  {query_label}")
    if not results:
        print("  ไม่พบข้อความที่เกี่ยวข้องในคู่มือที่มีอยู่")
        return
    for i, r in enumerate(results, 1):
        print(f"\n  [{i}] {r['source']}")
        print(f"      หน้า {r['page']}  ความเกี่ยวข้อง {r['relevance']}%  "
              f"คุณภาพแหล่ง {r['quality']}  หัวข้อ {','.join(r['topics'])}")
        if r.get("quality_note"):
            print(f"      {r['quality_note']}")
        text = r["text"].replace("\n", " ")
        print(f"      {text[:260]}{'…' if len(text) > 260 else ''}")
        print(f"      วลีที่ตรงกัน {', '.join(r['matched_phrases']) or '-'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="*", help="คำถามภาษาไทย")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--topics", nargs="*", default=None)
    ap.add_argument("--wound-class", default=None,
                    help="ชื่อคลาสจาก wound_classifier_service เช่น Pressure Wounds")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    st = ks.status()
    if args.status or not st["ready"]:
        print("สถานะดัชนีความรู้ออฟไลน์")
        for key, value in st.items():
            if value not in (None, False) or key in ("ready",):
                print(f"  {key:<22} {value}")
        if not st["ready"]:
            return 1
        if args.status:
            return 0

    if args.wound_class:
        show(ks.for_wound_class(args.wound_class, k=args.k),
             f"ประเภทแผล {args.wound_class}")
        return 0

    if not args.query:
        print("ใส่คำถามด้วย เช่น  python tools/ask_offline.py \"ทำแผลสะอาดอย่างไร\"")
        return 1

    query = " ".join(args.query)
    show(ks.search(ks.expand_query(query), k=args.k, topics=args.topics), query)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
