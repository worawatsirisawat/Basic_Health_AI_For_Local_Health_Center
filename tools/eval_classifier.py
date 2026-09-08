"""
eval_classifier.py
ประเมินความแม่นยำของการจำแนกประเภทแผลด้วยการเทียบภาพ

วิธีรัน (จากโฟลเดอร์หลักของโปรเจกต์)
    python tools/eval_classifier.py
    python tools/eval_classifier.py --per-class 15 --repeats 5

สิ่งที่สคริปต์นี้ตอบ
    1. วิธีเทียบภาพแบบสุ่มอ้างอิง 5 ภาพต่อโฟลเดอร์ แม่นยำกี่เปอร์เซ็นต์
    2. ความแม่นยำแกว่งแค่ไหนเมื่อสุ่มใหม่ ซึ่งเป็นจุดอ่อนสำคัญของการสุ่มน้อยภาพ
    3. ถ้าเทียบกับทุกภาพในโฟลเดอร์แทนการสุ่ม จะแม่นขึ้นเท่าไร
    4. ประเภทแผลใดถูกสับสนกับประเภทใดบ่อยที่สุด

การกันข้อมูลรั่ว
    ภาพที่ถูกใช้เป็นภาพทดสอบ และภาพกลับด้านของภาพนั้น จะถูกกันออกจากคลังอ้างอิง
    เสมอ เพื่อไม่ให้ระบบเทียบภาพกับตัวมันเอง
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.wound_classifier_service import (  # noqa: E402
    CATEGORY_TH, DEFAULT_SAMPLE_K, DEFAULT_TOP_M, DEFAULT_TEMPERATURE,
    ReferenceLibrary, classify,
)

MIRROR_PREFIXES = ("mirrored_", "mirror_", "flipped_")


def siblings_of(path: Path) -> set:
    """คืนไฟล์ที่ถือว่าเป็นภาพเดียวกัน คือตัวมันเองและคู่กลับด้านของมัน"""
    out = {path}
    name = path.name
    lowered = name.lower()
    for prefix in MIRROR_PREFIXES:
        if lowered.startswith(prefix):
            out.add(path.with_name(name[len(prefix):]))
        else:
            out.add(path.with_name(prefix + name))
    return {p for p in out if p.exists()}


def build_testset(lib: ReferenceLibrary, per_class: int, rng: random.Random):
    tests = []
    for category in lib.categories():
        pool = lib.images_in(category)
        if len(pool) <= per_class + 1:
            continue
        for path in rng.sample(pool, min(per_class, len(pool) - 1)):
            tests.append((category, path))
    return tests


def run_pass(lib, tests, sample_k, top_m, temperature, seed):
    correct = 0
    confusion = {}
    for truth, path in tests:
        result = classify(
            path, lib,
            sample_k=sample_k, top_m=top_m, temperature=temperature,
            seed=seed, exclude=siblings_of(path),
        )
        if not result.get("ok"):
            continue
        pred = result["top"]
        confusion.setdefault(truth, {}).setdefault(pred, 0)
        confusion[truth][pred] += 1
        if pred == truth:
            correct += 1
    return correct / len(tests) if tests else 0.0, confusion


def print_confusion(confusion, categories):
    short = {c: (c[:9]) for c in categories}
    header = "จริง \\ ทาย".ljust(20) + "".join(short[c].rjust(11) for c in categories)
    print(header)
    print("-" * len(header))
    for truth in categories:
        row = confusion.get(truth, {})
        total = sum(row.values()) or 1
        line = f"{truth[:19]:<20}"
        for pred in categories:
            n = row.get(pred, 0)
            cell = f"{n}" if n else "."
            if pred == truth and n:
                cell = f"[{n}]"
            line += cell.rjust(11)
        print(line + f"   ถูก {row.get(truth,0)}/{total}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="โฟลเดอร์คลังภาพอ้างอิง")
    ap.add_argument("--per-class", type=int, default=10, help="จำนวนภาพทดสอบต่อประเภท")
    ap.add_argument("--sample-k", type=int, default=DEFAULT_SAMPLE_K)
    ap.add_argument("--top-m", type=int, default=DEFAULT_TOP_M)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--repeats", type=int, default=3, help="จำนวนรอบที่สุ่มภาพอ้างอิงใหม่")
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--skip-full", action="store_true", help="ข้ามการเทียบกับทุกภาพในโฟลเดอร์")
    args = ap.parse_args()

    root = args.root or (Path(__file__).resolve().parents[1] / "data" / "Image_Classifier")
    lib = ReferenceLibrary(root)

    categories = lib.categories()
    if not categories:
        print(f"ไม่พบโฟลเดอร์ประเภทแผลใน {root}")
        return 1

    print(f"คลังภาพอ้างอิง  {root}")
    inv = lib.inventory()
    for c in categories:
        print(f"  {c:<20} {inv[c]:>5} ภาพ   {CATEGORY_TH.get(c, '')}")
    print(f"  รวม {sum(inv.values())} ภาพ (ไม่นับภาพกลับด้าน)\n")

    rng = random.Random(args.seed)
    tests = build_testset(lib, args.per_class, rng)
    print(f"ชุดทดสอบ {len(tests)} ภาพ  ({args.per_class} ภาพต่อประเภท)\n")

    print(f"=== โหมดสุ่มภาพอ้างอิง {args.sample_k} ภาพต่อโฟลเดอร์ "
          f"(ใช้ค่าเฉลี่ยของ {args.top_m} อันดับแรก) ===")
    accs, last_conf = [], {}
    for i in range(args.repeats):
        acc, conf = run_pass(lib, tests, args.sample_k, args.top_m,
                             args.temperature, seed=args.seed + i)
        accs.append(acc)
        last_conf = conf
        print(f"  รอบที่ {i+1}  ความแม่นยำ {acc*100:5.1f}%")

    if len(accs) > 1:
        print(f"\n  เฉลี่ย {statistics.mean(accs)*100:.1f}%  "
              f"ต่ำสุด {min(accs)*100:.1f}%  สูงสุด {max(accs)*100:.1f}%  "
              f"ส่วนเบี่ยงเบน {statistics.pstdev(accs)*100:.1f} จุด")
        print("  ยิ่งส่วนเบี่ยงเบนสูง แปลว่าผลขึ้นกับว่าสุ่มภาพอ้างอิงได้ภาพไหน "
              "ซึ่งไม่เหมาะกับการใช้งานจริง")

    print(f"\nตารางความสับสน (รอบสุดท้าย)")
    print_confusion(last_conf, categories)

    if not args.skip_full:
        print(f"\n=== โหมดเทียบกับทุกภาพในโฟลเดอร์ (ไม่สุ่ม) ===")
        print("  กำลังคำนวณ อาจใช้เวลาสักครู่ในรอบแรกเพราะต้องสร้าง cache")
        acc_full, conf_full = run_pass(lib, tests, 0, args.top_m, args.temperature, seed=0)
        print(f"  ความแม่นยำ {acc_full*100:5.1f}%   (คงที่ทุกครั้งเพราะไม่มีการสุ่ม)")
        print(f"\nตารางความสับสน")
        print_confusion(conf_full, categories)

    print("\nข้อควรระวัง ตัวเลขนี้วัดกับภาพจากชุดข้อมูลเดียวกันซึ่งถ่ายในบริบทคลินิก")
    print("ภาพที่ อสม. ถ่ายเองด้วยมือถือในแสงจริงจะให้ผลต่ำกว่านี้ ต้องทดสอบซ้ำก่อนใช้จริง")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
