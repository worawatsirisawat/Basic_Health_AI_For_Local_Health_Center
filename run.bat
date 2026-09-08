@echo off
chcp 65001 >nul
title Local Health AI - Prototype

echo.
echo   Local Health AI - Prototype
echo   ============================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo   [ผิดพลาด] ไม่พบ Python บนเครื่องนี้
    echo   ติดตั้งจาก https://www.python.org/downloads/ แล้วลองใหม่
    echo   ตอนติดตั้ง อย่าลืมติ๊ก "Add Python to PATH"
    pause
    exit /b 1
)

echo   [1/2] ตรวจสอบและติดตั้ง dependencies...
python -m pip install -q -r requirements.txt
if errorlevel 1 (
    echo   ติดตั้ง dependencies ไม่สำเร็จ
    pause
    exit /b 1
)

echo   [2/2] เริ่มต้นเซิร์ฟเวอร์...
echo.
echo   เปิดเบราว์เซอร์ไปที่  http://127.0.0.1:8000
echo   กด Ctrl+C เพื่อหยุดการทำงาน
echo.

python backend\main.py
pause
