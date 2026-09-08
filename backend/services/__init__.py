"""services — โมดูลย่อยแยกตามหน้าที่การทำงาน

auth_service      ลงทะเบียนหน่วยบริการและยืนยันตัวตน
medicine_service  บัญชียาของหน่วยบริการและการหายาทดแทน
triage_service    ชั้นกฎตายตัวด้านความปลอดภัย ทำงานก่อน AI เสมอ
llm_service       เชื่อมต่อ Typhoon LLM และ Mock mode
pubmed_service    ดึงหลักฐานงานวิจัยจาก NCBI E-utilities
wound_service     ประมวลผลภาพแผลจากกรอบที่ผู้ใช้วาด
consent_service   ความยินยอมตาม PDPA และบันทึกประวัติ
"""
