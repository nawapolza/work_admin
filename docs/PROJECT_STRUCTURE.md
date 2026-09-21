# Project structure

```text
mango-ai-assistant/
├── frontend/                  React + Vite + Tailwind CSS
│   ├── src/
│   │   ├── App.jsx           หน้าหลัก อัปโหลด ผลลัพธ์ และจัดการโมเดล
│   │   ├── api.js            ตัวเรียก Flask API
│   │   └── styles.css        Responsive desktop/mobile
│   ├── public/
│   ├── package.json
│   └── vite.config.js
├── backend/                   Flask + TensorFlow/Keras
│   ├── app.py                API endpoints และ if/elif pipeline
│   ├── model_service.py      โหลด/ตรวจ/ทำนาย/เปลี่ยน .h5 และ .keras
│   ├── config.py             อ่านค่าจาก .env
│   ├── models/               ไฟล์ Model 1 และ Model 2
│   ├── uploads/              พื้นที่ไฟล์ชั่วคราว
│   ├── logs/                 บันทึกการทำงาน
│   └── requirements.txt
├── venv/                     ตำแหน่ง Python virtual environment
├── docs/                     เอกสารโครงสร้างและ API
├── .env.example              ตัวอย่าง environment variables
├── SETUP_WINDOWS.bat         สร้าง venv + ติดตั้งทุกอย่าง
├── RUN_WINDOWS.bat           รัน frontend และ backend
├── setup_linux.sh
├── run_linux.sh
└── docker-compose.yml
```
