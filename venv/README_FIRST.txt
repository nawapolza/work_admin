โฟลเดอร์นี้เป็นตำแหน่งสำหรับ Python Virtual Environment ของโปรเจกต์

เหตุผลที่ไม่บรรจุไฟล์ Python และ TensorFlow ที่ติดตั้งแล้ว:
- venv จาก Windows ใช้บน macOS/Linux ไม่ได้
- venv จาก macOS/Linux ใช้บน Windows ไม่ได้
- path ภายใน venv ผูกกับตำแหน่งและ Python ของเครื่องที่สร้าง

สร้าง venv ให้ถูกต้องอัตโนมัติ:
- Windows: ดับเบิลคลิก SETUP_WINDOWS.bat ที่โฟลเดอร์หลัก
- macOS/Linux: รัน bash setup_linux.sh

สคริปต์จะสร้าง venv ไว้ในโฟลเดอร์นี้และติดตั้ง Flask, TensorFlow/Keras,
Pillow, NumPy และ dependency ทั้งหมดจาก backend/requirements.txt

