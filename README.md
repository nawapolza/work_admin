# Mango AI Assistant — University Full Stack ML Project

โปรเจกต์วิเคราะห์ความสุกมะม่วงแบบ 2 โมเดล แยกหน้าที่และ API ชัดเจน

เวอร์ชันนี้เพิ่ม Test-Time Augmentation (TTA) สำหรับ Model 1 โดยทำนายจากภาพต้นฉบับ ภาพกลับด้าน และการปรับแสง/คอนทราสต์เล็กน้อยก่อนเฉลี่ยผล ช่วยให้ผลนิ่งขึ้นในสภาพแสงต่างกัน แต่ความแม่นยำจริงยังขึ้นกับคุณภาพ dataset และค่าที่ใช้ตอนเทรน

โครงสร้างแยกชัดเจนเป็น `frontend/`, `backend/`, `venv/`, `docs/` พร้อมสคริปต์ติดตั้งและรันอัตโนมัติ ดูรายละเอียดทั้งหมดที่ `docs/PROJECT_STRUCTURE.md`

## สถาปัตยกรรม

1. **Model 1 — Mango Detector** รับภาพและจำแนก `mango / non_mango`
2. ถ้า class เป็น `mango` และ confidence ถึงเกณฑ์ จึงเข้าสู่ Model 2
3. **Model 2 — Ripeness Classifier** จำแนก `stage1 / stage2 / stage3 / stage4 / stage5`
4. ถ้าไม่ใช่มะม่วงหรือ confidence ต่ำ ระบบหยุดทันทีและไม่เรียก Model 2

API แยกกัน:

- `POST /api/v1/predict/identify` — Model 1 เท่านั้น
- `POST /api/v1/predict/ripeness` — Model 2 เท่านั้น
- `POST /api/v1/predict/pipeline` — ทำงานตามลำดับ Model 1 → เงื่อนไข → Model 2
- `GET /api/v1/health` — สถานะ, class, input size, preprocessing และ hash ของโมเดล
- `POST /api/v1/admin/login` — เข้าสู่ระบบผู้ดูแล
- `GET /api/v1/admin/models` — ดูสถานะโมเดล (Admin เท่านั้น)
- `POST /api/v1/admin/models/replace` — เปลี่ยนโมเดลและตรวจ shape (Admin เท่านั้น)

## 1. ใส่โมเดล

นำไฟล์มาวางดังนี้:

```text
backend/models/mango_detector.h5
backend/models/ripeness_classifier.h5
```

หรือเปลี่ยน path ใน `.env` ได้ ถ้าชื่อ class, input size และ preprocessing ตอนเทรนต่างจากค่าเริ่มต้น ต้องแก้ให้ตรงทุกค่า ไม่เช่นนั้นผลอาจไม่แม่นยำ

## วิธีง่ายที่สุดสำหรับ Windows

1. แตก ZIP
2. ดับเบิลคลิก `SETUP_WINDOWS.bat` เพื่อสร้าง `venv` และติดตั้งทุก dependency
3. วางโมเดล 2 ไฟล์ใน `backend/models/`
4. ดับเบิลคลิก `RUN_WINDOWS.bat`
5. เปิด `http://localhost:5173`

`venv` ต้องถูกสร้างบนเครื่องที่จะใช้งาน เพราะไฟล์ภายในผูกกับระบบปฏิบัติการและตำแหน่ง Python ของเครื่องนั้น จึงมีโฟลเดอร์ `venv/` พร้อมตัวสร้างอัตโนมัติให้แล้ว

## 2. ตั้งค่า ENV

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

macOS/Linux:

```bash
cp .env.example .env
```

ค่าที่สำคัญ:

| ค่า | ความหมาย |
|---|---|
| `MODEL1_CLASSES` | ลำดับ class ต้องตรงกับตอนเทรน Model 1 |
| `MODEL2_CLASSES` | ลำดับ class ต้องตรงกับตอนเทรน Model 2 |
| `MODEL*_INPUT_SIZE` | ขนาดภาพ เช่น `224,224` |
| `MODEL*_PREPROCESS` | `rescale`, `minus_one_one`, `imagenet`, `none` |
| `MODEL*_OUTPUT_MODE` | `auto`, `softmax`, `logits` |
| `MODEL1_BINARY_POSITIVE_INDEX` | สำหรับ sigmoid 1 ค่า: index ของ positive class; ปกติ Keras binary ใช้ `1` |
| `MODEL1_THRESHOLD` | เกณฑ์ผ่าน Model 1 |
| `MODEL1_TTA` | เปิด/ปิดการเฉลี่ยผลหลายมุมของ Model 1 |
| `MODEL1_TTA_VARIANTS` | จำนวนภาพแปรผันที่ใช้เฉลี่ย 1–6 |
| `MAX_UPLOAD_MB` | `0` หมายถึงไม่มีเพดานขนาดไฟล์แบบ MB |
| `MAX_IMAGE_PIXELS` | เพดานพิกเซลก่อนระบบย่อภาพเพื่อควบคุมหน่วยความจำ |
| `VITE_API_BASE_URL` | URL Flask ที่หน้าเว็บเรียก |
| `ADMIN_USERNAME` | ชื่อผู้ใช้สำหรับหน้า `/admin` |
| `ADMIN_PASSWORD` | รหัสผ่านผู้ดูแล ควรเปลี่ยนก่อนใช้งานจริง |
| `SECRET_KEY` | คีย์สำหรับลงลายเซ็น Session ต้องใช้ข้อความยาวและคาดเดายาก |
| `ADMIN_SESSION_HOURS` | อายุการเข้าสู่ระบบผู้ดูแลเป็นชั่วโมง |
| `COOKIE_SECURE` | ตั้ง `true` เมื่อใช้งานผ่าน HTTPS |

## 3. ติดตั้งและรัน

ต้องมี Node.js 20+ และ Python 3.10–3.12

```bash
npm --prefix frontend install
python -m venv venv
```

Windows PowerShell:

```powershell
venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
npm run dev
```

macOS/Linux:

```bash
source venv/bin/activate
pip install -r backend/requirements.txt
npm run dev
```

เปิด `http://localhost:5173` หากมือถืออยู่ Wi-Fi วงเดียวกัน ให้เปิด `http://IP-คอมพิวเตอร์:5173`

ในเวอร์ชันล่าสุดไม่ต้องใส่ IP ของ Backend แล้ว: Frontend เรียก `/api` ผ่าน Vite proxy อัตโนมัติ ดังนั้นมือถือให้เปิดเพียง `http://IP-คอมพิวเตอร์:5173` เท่านั้น หากเห็นข้อความเชื่อมต่อ Backend ไม่ได้ ให้ตรวจหน้าต่าง `Mango AI Backend` ว่ายังเปิดอยู่และไม่มี error ระหว่างติดตั้ง TensorFlow

หากพอร์ต `5173` มีโปรแกรมใช้งานอยู่ Vite จะเลือก `5174`, `5175` หรือพอร์ตถัดไปให้อัตโนมัติ ให้เปิด URL บรรทัด `Local` ที่แสดงในหน้าต่าง **Mango AI Frontend** และใช้เลขพอร์ตเดียวกันเมื่อเปิดจากมือถือ ไม่ควรกด `RUN_WINDOWS.bat` ซ้ำหากหน้าต่าง Backend/Frontend เดิมยังเปิดอยู่

สำหรับ Docker หน้า Nginx จะ proxy `/api` ไปยัง Flask และตั้ง `client_max_body_size 0` เพื่อรองรับไฟล์โมเดลขนาดใหญ่

## การเปลี่ยนโมเดล

ผู้ใช้งานทั่วไปเปิด `http://localhost:5173` เพื่ออัปโหลดรูปและดูผลทำนาย โดยจะไม่เห็นเมนูหรือข้อมูลไฟล์โมเดล

ผู้ดูแลเปิด `http://localhost:5173/admin` และเข้าสู่ระบบด้วยค่าจาก `.env` จากนั้นจึงสามารถลากไฟล์ `.h5` หรือ `.keras` ลงในช่อง Model 1/2 ได้ ระบบจะแยกไฟล์ ตรวจ input/output สำรองไฟล์เดิม และ rollback อัตโนมัติหากโหลดไม่ผ่าน

ค่าเริ่มต้นสำหรับทดลองในเครื่องคือ `admin` / `MangoAdmin@2026` แต่ต้องเปลี่ยน `ADMIN_PASSWORD` และ `SECRET_KEY` ใน `.env` ก่อนเปิดให้เครื่องอื่นหรือเครือข่ายสาธารณะใช้งาน

เมื่อไฟล์ผ่าน ระบบจะบันทึกช่องโมเดลที่กำลังใช้งานพร้อมชื่อเรียก ชื่อไฟล์ต้นฉบับ วันที่ติดตั้ง และ Model ID ลงใน `backend/models/model1.metadata.json` หรือ `model2.metadata.json` โดยอัตโนมัติ จึงยังเห็นโมเดลเดิมหลังปิดเว็บหรือรีสตาร์ต Flask และไม่ต้องเลือกไฟล์ใหม่ทุกครั้ง

> ใช้เฉพาะไฟล์ `.h5` หรือ `.keras` ที่สร้างเองหรือเชื่อถือได้ และไม่ควรเปิดหน้าจัดการโมเดลสู่เครือข่ายสาธารณะ

## เรื่องความแม่นยำ

โค้ดทำ inference ตามลำดับอย่างถูกต้อง แต่ไม่สามารถรับประกันเปอร์เซ็นต์ความแม่นยำแทนตัวโมเดลได้ ความแม่นยำขึ้นกับ dataset, class order, input shape, color mode และ preprocessing ตอนเทรน ตรวจค่าเหล่านี้ให้ตรงกับ notebook ที่ใช้สร้างโมเดล

เวอร์ชันนี้แสดงผลดิบจาก Keras และหลักฐานทุกส่วนแยกกัน: Model 1 ใช้ TTA, Model 2 ใช้ TTA และหลักฐานเสริมจากสัดส่วนสีเขียว/เหลือง/ส้ม/จุดมืด กดการ์ดผลลัพธ์เพื่อดูคะแนนทุก class, ชื่อไฟล์โมเดล และเหตุผลการตัดสินใจ

- `MODEL2_VISUAL_WEIGHT=0.18` ปรับได้ช่วง `0.00-0.45`; ตั้ง `0` เพื่อใช้ Keras ล้วน
- Model 1 แบบ sigmoid 1 output ต้องตั้ง `MODEL1_CLASSES` และ `MODEL1_BINARY_POSITIVE_INDEX` ให้ตรง `class_indices` ตอน train มิฉะนั้นผล mango/non_mango อาจกลับด้าน
- ตัวอย่าง: ถ้า `class_indices` คือ `{'mango': 0, 'non_mango': 1}` ให้ใช้ `MODEL1_CLASSES=mango,non_mango` และ `MODEL1_BINARY_POSITIVE_INDEX=1` เพราะ sigmoid=1 หมายถึง `non_mango`

Model 2 ใหม่ควรเป็น 5 output `stage1-stage5` เท่านั้น ส่วนไฟล์โมเดลเดิมของคุณที่มี 6 output รองรับผ่าน Compatibility Adapter (`MODEL2_LEGACY_6_ENABLE=true`) ระบบจะตัดช่อง `non_mango` index 0 ภายในและ normalize เหลือ 5 Stage หน้าเว็บจึงยังแสดง Model 2 เป็น `stage1-stage5` เท่านั้น ส่วน Model 1 รับผิดชอบ `mango/non_mango` และหยุด Pipeline เมื่อไม่ใช่มะม่วง
"# work_admin" 
