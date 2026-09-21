from __future__ import annotations
import logging
import os
import tempfile
import time
import hmac
from functools import wraps
from datetime import timedelta
from pathlib import Path
from PIL import Image, ImageOps, UnidentifiedImageError
from flask import Flask, jsonify, request, session
from flask_cors import CORS
from config import MANGO_CLASS, MANGO_THRESHOLD, MAX_UPLOAD_MB, MAX_IMAGE_PIXELS, MODEL1, MODEL1_TTA, MODEL1_TTA_VARIANTS, MODEL2, MODEL2_TTA, MODEL2_TTA_VARIANTS, MODEL2_VISUAL_WEIGHT
from image_features import analyze_mango_surface, fuse_stage_prediction
from model_service import KerasModelService, ModelError

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
log=logging.getLogger('mango-ai.api')
Image.MAX_IMAGE_PIXELS=None
app=Flask(__name__)
app.secret_key=os.getenv('SECRET_KEY','change-this-secret-before-public-deployment')
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Strict',
    SESSION_COOKIE_SECURE=os.getenv('COOKIE_SECURE','false').lower()=='true',
    PERMANENT_SESSION_LIFETIME=timedelta(hours=int(os.getenv('ADMIN_SESSION_HOURS','8'))),
)
if MAX_UPLOAD_MB > 0: app.config['MAX_CONTENT_LENGTH']=MAX_UPLOAD_MB*1024*1024
cors_value=os.getenv('FRONTEND_ORIGINS','*').strip()
cors_origins='*' if cors_value=='*' else [x.strip() for x in cors_value.split(',') if x.strip()]
CORS(app, resources={r'/api/*':{'origins':cors_origins}}, expose_headers=['Content-Length'], supports_credentials=True)

ADMIN_USERNAME=os.getenv('ADMIN_USERNAME','admin')
ADMIN_PASSWORD=os.getenv('ADMIN_PASSWORD','MangoAdmin@2026')
login_attempts={}

def require_admin(view):
    @wraps(view)
    def wrapped(*args,**kwargs):
        if session.get('role')!='admin':
            return jsonify(error='กรุณาเข้าสู่ระบบผู้ดูแล'),401
        return view(*args,**kwargs)
    return wrapped

def public_prediction(result):
    """Hide model registry data from normal users while keeping prediction evidence."""
    clean=dict(result)
    clean.pop('model',None)
    return clean

model1=KerasModelService(MODEL1); model2=KerasModelService(MODEL2)
model1.load(); model2.load()

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    log.info('HEIC/HEIF image support enabled')
except ImportError:
    log.warning('pillow-heif not installed; HEIC/HEIF disabled')

def image_from_request():
    file=request.files.get('image')
    if not file or not file.filename: raise ValueError('กรุณาแนบรูปในฟิลด์ image')
    try:
        image=Image.open(file.stream); image.seek(0)
        if image.width*image.height > MAX_IMAGE_PIXELS:
            ratio=(MAX_IMAGE_PIXELS/(image.width*image.height))**0.5
            target=(max(1,int(image.width*ratio)),max(1,int(image.height*ratio)))
            image.draft('RGB',target); image.thumbnail(target,Image.Resampling.LANCZOS)
        image=ImageOps.exif_transpose(image).convert('RGB')
        return image.copy()
    except (UnidentifiedImageError,OSError): raise ValueError('ไฟล์นี้ไม่ใช่รูปภาพที่ถูกต้อง')

@app.get('/api/v1/health')
def health():
    return jsonify(status='ok' if model1.model is not None and model2.model is not None else 'degraded')

@app.post('/api/v1/admin/login')
def admin_login():
    ip=request.headers.get('X-Forwarded-For',request.remote_addr or 'unknown').split(',')[0].strip()
    now=time.time(); recent=[t for t in login_attempts.get(ip,[]) if now-t<300]
    if len(recent)>=5:
        return jsonify(error='เข้าสู่ระบบไม่สำเร็จหลายครั้ง กรุณารอ 5 นาที'),429
    data=request.get_json(silent=True) or {}
    user=str(data.get('username','')); password=str(data.get('password',''))
    if not (hmac.compare_digest(user,ADMIN_USERNAME) and hmac.compare_digest(password,ADMIN_PASSWORD)):
        recent.append(now); login_attempts[ip]=recent
        return jsonify(error='ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง'),401
    login_attempts.pop(ip,None); session.clear(); session.permanent=True
    session['role']='admin'; session['username']=ADMIN_USERNAME
    return jsonify(authenticated=True,username=ADMIN_USERNAME)

@app.post('/api/v1/admin/logout')
def admin_logout():
    session.clear(); return jsonify(authenticated=False)

@app.get('/api/v1/admin/session')
def admin_session():
    return jsonify(authenticated=session.get('role')=='admin',username=session.get('username'))

@app.get('/api/v1/admin/models')
@require_admin
def admin_models():
    return jsonify(status='ok' if model1.model is not None and model2.model is not None else 'degraded',models={'model1':model1.metadata(),'model2':model2.metadata()},threshold=MANGO_THRESHOLD,visual_weight=MODEL2_VISUAL_WEIGHT)

@app.post('/api/v1/predict/identify')
def identify():
    image=image_from_request(); result=model1.predict_tta(image,MODEL1_TTA_VARIANTS) if MODEL1_TTA else model1.predict(image); result['is_mango']=result['class_name']==MANGO_CLASS and result['confidence']>=MANGO_THRESHOLD
    log.info('Model1 result=%s confidence=%.4f is_mango=%s',result['class_name'],result['confidence'],result['is_mango'])
    return jsonify(public_prediction(result))

@app.post('/api/v1/predict/ripeness')
def ripeness():
    image=image_from_request(); visual=analyze_mango_surface(image)
    result=model2.predict_tta(image,MODEL2_TTA_VARIANTS) if MODEL2_TTA else model2.predict(image)
    result=fuse_stage_prediction(result,visual,MODEL2_VISUAL_WEIGHT)
    log.info('Model2 result=%s confidence=%.4f',result['class_name'],result['confidence'])
    return jsonify(public_prediction(result))

@app.post('/api/v1/predict/pipeline')
def pipeline():
    started=time.perf_counter(); image=image_from_request(); trace=[]
    t=time.perf_counter(); first=model1.predict_tta(image,MODEL1_TTA_VARIANTS) if MODEL1_TTA else model1.predict(image); first['is_mango']=first['class_name']==MANGO_CLASS and first['confidence']>=MANGO_THRESHOLD
    trace.append({'step':'Model 1 · ตรวจจับวัตถุ','detail':f"ผล {first['class_name']} ความมั่นใจ {first['confidence']*100:.1f}%",'duration_ms':round((time.perf_counter()-t)*1000,2)})
    second=None
    # Explicit routing: Model 2 runs only when Model 1 passes both class and threshold.
    if first['class_name'] == MANGO_CLASS and first['confidence'] >= MANGO_THRESHOLD:
        t=time.perf_counter(); visual=analyze_mango_surface(image)
        second=model2.predict_tta(image,MODEL2_TTA_VARIANTS) if MODEL2_TTA else model2.predict(image)
        second=fuse_stage_prediction(second,visual,MODEL2_VISUAL_WEIGHT)
        trace.append({'step':'Model 2 · จำแนกระดับความสุก','detail':f"ผล {second['class_name']} ความมั่นใจ {second['confidence']*100:.1f}%",'duration_ms':round((time.perf_counter()-t)*1000,2)})
    elif first['class_name'] != MANGO_CLASS:
        trace.append({'step':'หยุด Pipeline','detail':'Model 1 จำแนกว่าไม่ใช่มะม่วง จึงไม่เรียก Model 2','duration_ms':0})
    else:
        trace.append({'step':'หยุด Pipeline','detail':f'เป็นมะม่วงแต่ความมั่นใจต่ำกว่าเกณฑ์ {MANGO_THRESHOLD*100:.0f}%','duration_ms':0})
    trace.append({'step':'สรุปผล','detail':'จัดรูปแบบผลลัพธ์และคำแนะนำสำหรับหน้าเว็บ','duration_ms':round((time.perf_counter()-started)*1000,2)})
    return jsonify(identification=public_prediction(first),ripeness=public_prediction(second) if second else None,trace=trace,total_duration_ms=round((time.perf_counter()-started)*1000,2))

@app.post('/api/v1/admin/models/replace')
@require_admin
def replace_model():
    slot=request.form.get('slot'); service={'model1':model1,'model2':model2}.get(slot)
    if not service: return jsonify(error='slot ต้องเป็น model1 หรือ model2'),400
    upload=request.files.get('model')
    suffix=Path(upload.filename).suffix.lower() if upload else ''
    if not upload or suffix not in ('.h5','.keras'): return jsonify(error='กรุณาเลือกไฟล์ .h5 หรือ .keras'),400
    with tempfile.NamedTemporaryFile(suffix=suffix,delete=False,dir=service.config.path.parent) as temp:
        upload.save(temp); temp_path=Path(temp.name)
    display_name=(request.form.get('display_name') or '').strip()[:100]
    try: service.replace(temp_path,original_filename=upload.filename,display_name=display_name); return jsonify(message=f'บันทึกและเปิดใช้ {slot} สำเร็จ',model=service.metadata())
    except Exception as exc: temp_path.unlink(missing_ok=True); log.exception('replace failed'); return jsonify(error=f'โมเดลไม่ผ่านการตรวจสอบ: {exc}'),400

@app.errorhandler(ValueError)
def bad_input(error): return jsonify(error=str(error)),400
@app.errorhandler(ModelError)
def model_error(error): return jsonify(error=str(error)),503
@app.errorhandler(Exception)
def unexpected(error): log.exception('unexpected'); return jsonify(error='เกิดข้อผิดพลาดภายในเซิร์ฟเวอร์'),500

if __name__=='__main__': app.run(host=os.getenv('FLASK_HOST','0.0.0.0'),port=int(os.getenv('FLASK_PORT','5000')),debug=os.getenv('FLASK_DEBUG','false').lower()=='true')
