#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo ""
echo "  Local Health AI — Prototype"
echo "  ==========================="
echo ""
echo "  [1/2] ตรวจสอบและติดตั้ง dependencies..."
python3 -m pip install -q -r requirements.txt

echo "  [2/2] เริ่มต้นเซิร์ฟเวอร์..."
echo ""
echo "  เปิดเบราว์เซอร์ไปที่  http://127.0.0.1:8000"
echo "  กด Ctrl+C เพื่อหยุดการทำงาน"
echo ""

python3 backend/main.py
