"""ตรวจไฟล์โมเดลและแสดง shape/class โดยไม่เปิดเว็บ"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
from config import MODEL1, MODEL2
from model_service import KerasModelService

failed=False
for config in (MODEL1,MODEL2):
    service=KerasModelService(config); ok=service.load(); print(config.slot, service.metadata())
    failed = failed or not ok
raise SystemExit(1 if failed else 0)
