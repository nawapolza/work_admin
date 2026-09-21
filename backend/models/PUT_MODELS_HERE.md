# วางโมเดลไว้ที่นี่

- `mango_detector.h5` หรือ `.keras` — Model 1: `mango`, `non_mango`
- `ripeness_classifier.h5` หรือ `.keras` — Model 2: `stage1`, `stage2`, `stage3`, `stage4`, `stage5`

Model 2 ใหม่ควรมี 5 output เท่านั้น หากไฟล์เดิมมี 6 output ตามลำดับ `non_mango,stage1,stage2,stage3,stage4,stage5` ระบบ Compatibility Adapter จะตัด output index 0 ภายใน Backend และส่งออกเฉพาะ Stage 1-5 โดย Model 1 ยังเป็นผู้ตัดสินชนิดผลไม้เพียงตัวเดียว

ถ้าชื่อไฟล์หรือขนาด input ต่างออกไป ให้แก้ `MODEL1_*` / `MODEL2_*` ใน `.env`
