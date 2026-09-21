# API specification

ทุก prediction endpoint รับ `multipart/form-data` โดยชื่อไฟล์คือ `image`

| Method | Endpoint | Model ที่ทำงาน |
|---|---|---|
| GET | `/api/v1/health` | ตรวจสถานะทั้งสองโมเดล |
| POST | `/api/v1/predict/identify` | Model 1 เท่านั้น |
| POST | `/api/v1/predict/ripeness` | Model 2 เท่านั้น |
| POST | `/api/v1/predict/pipeline` | Model 1 และ Model 2 ตามเงื่อนไข |
| POST | `/api/v1/admin/models/replace` | เปลี่ยนโมเดลตาม `slot` |

Pipeline:

```text
รับภาพ
  └─ Model 1
      ├─ class != mango         → หยุด ไม่เรียก Model 2
      ├─ confidence < threshold → หยุด ไม่เรียก Model 2
      └─ mango + ผ่าน threshold → Model 2 → stage1-stage5
```

