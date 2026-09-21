from __future__ import annotations
import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

log = logging.getLogger('mango-ai.models')

class ModelError(RuntimeError): pass

class KerasModelService:
    """Thread-safe Keras H5 loader and predictor for exactly one model slot."""
    def __init__(self, config):
        self.config = config
        self.pointer = config.path.parent / f'{config.slot}.active'
        self.meta_path = config.path.parent / f'{config.slot}.metadata.json'
        self.active_path = self._resolve_active_path()
        self.model = None
        self.lock = threading.RLock()
        self.loaded_at = None
        self.error = None

    def _resolve_active_path(self):
        if self.pointer.exists():
            candidate = self.config.path.parent / self.pointer.read_text(encoding='utf-8').strip()
            if candidate.exists() and candidate.suffix.lower() in ('.h5', '.keras'):
                return candidate
        return self.config.path

    def load(self):
        with self.lock:
            self.active_path = self._resolve_active_path()
            if not self.active_path.exists():
                self.model = None; self.error = f'ไม่พบโมเดล: {self.config.path.name}'
                log.warning('[%s] %s', self.config.slot, self.error); return False
            try:
                os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
                import tensorflow as tf
                log.info('[%s] loading %s', self.config.slot, self.active_path)
                self.model = tf.keras.models.load_model(self.active_path, compile=False, safe_mode=True)
                self._validate_shape()
                self.loaded_at = time.time(); self.error = None
                log.info('[%s] ready input=%s output=%s classes=%s', self.config.slot, self.model.input_shape, self.model.output_shape, self.config.classes)
                return True
            except Exception as exc:
                self.model = None; self.error = f'{type(exc).__name__}: {exc}'
                log.exception('[%s] load failed', self.config.slot); return False

    def _validate_shape(self):
        output_shape = self.model.output_shape
        if isinstance(output_shape, list) and len(output_shape) == 1: output_shape = output_shape[0]
        output_units = int(output_shape[-1]) if isinstance(output_shape, tuple) else None
        expected = len(self.config.classes)
        accepted={expected,1} if expected==2 else {expected}
        if self._legacy_stage_adapter_enabled(): accepted.add(6)
        if output_units not in accepted:
            raise ModelError(f'output {output_units} ไม่ตรงกับจำนวน class ที่รองรับ {sorted(accepted)}')
        if output_units == 1 and expected != 2:
            raise ModelError('เอาต์พุต 1 ค่าใช้ได้เฉพาะ binary model ที่มี 2 classes')

    def metadata(self):
        accepted=[len(self.config.classes)]
        if self._legacy_stage_adapter_enabled(): accepted.append(6)
        saved=self._read_saved_metadata()
        return {'slot':self.config.slot,'loaded':self.model is not None,'display_name':saved.get('display_name') or saved.get('original_filename') or self.active_path.name,'original_filename':saved.get('original_filename') or self.active_path.name,'installed_at':saved.get('installed_at'),'filename':self.active_path.name,'path':str(self.active_path),'classes':self.config.classes,'accepted_output_units':accepted,'legacy_6_adapter':self._legacy_stage_adapter_enabled(),'input_size':self.config.input_size,'resample':os.getenv(f'{self.config.slot.upper()}_RESAMPLE','nearest'),'preprocess':self.config.preprocess,'output_mode':self.config.output_mode,'error':self.error,'sha256':self._sha256()}

    def _read_saved_metadata(self):
        try:
            return json.loads(self.meta_path.read_text(encoding='utf-8')) if self.meta_path.exists() else {}
        except (OSError, ValueError, TypeError):
            return {}

    def _write_saved_metadata(self, display_name, original_filename):
        data={'slot':self.config.slot,'display_name':display_name or original_filename,'original_filename':original_filename,'active_filename':self.active_path.name,'installed_at':datetime.now(timezone.utc).isoformat(),'sha256':self._sha256()}
        temp=self.meta_path.with_suffix('.tmp')
        temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
        temp.replace(self.meta_path)

    def _legacy_stage_adapter_enabled(self):
        return self.config.slot=='model2' and os.getenv('MODEL2_LEGACY_6_ENABLE','true').lower()=='true'

    def _sha256(self):
        if not self.active_path.exists(): return None
        h=hashlib.sha256()
        with self.active_path.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
        return h.hexdigest()[:16]

    def preprocess(self, image: Image.Image):
        mode = 'RGB' if self.config.color_mode.lower() == 'rgb' else 'L'
        resample_name=os.getenv(f'{self.config.slot.upper()}_RESAMPLE','nearest').lower()
        resamples={'nearest':Image.Resampling.NEAREST,'bilinear':Image.Resampling.BILINEAR,'bicubic':Image.Resampling.BICUBIC,'lanczos':Image.Resampling.LANCZOS}
        if resample_name not in resamples: raise ModelError(f'ไม่รู้จัก resize interpolation: {resample_name}')
        image = ImageOps.exif_transpose(image).convert(mode).resize(self.config.input_size, resamples[resample_name])
        arr = np.asarray(image, dtype=np.float32)
        if mode == 'L': arr = np.expand_dims(arr, -1)
        p = self.config.preprocess.lower()
        if p == 'rescale': arr /= 255.0
        elif p == 'minus_one_one': arr = (arr / 127.5) - 1.0
        elif p == 'imagenet':
            from tensorflow.keras.applications.imagenet_utils import preprocess_input
            arr = preprocess_input(arr, mode='caffe')
        elif p != 'none': raise ModelError(f'ไม่รู้จัก preprocessing: {p}')
        return np.expand_dims(arr, 0)

    def predict(self, image: Image.Image):
        if self.model is None: raise ModelError(f'{self.config.slot} ยังไม่พร้อม: {self.error}')
        started=time.perf_counter(); x=self.preprocess(image)
        with self.lock: raw=self._prediction_matrix(self.model.predict(x, verbose=0))[0]
        probs=self._class_probabilities(raw)
        index=int(np.argmax(probs)); scores={name:float(probs[i]) for i,name in enumerate(self.config.classes)}
        return {'class_name':self.config.classes[index],'confidence':float(probs[index]),'scores':scores,'duration_ms':round((time.perf_counter()-started)*1000,2),'model':self.metadata()}

    def predict_tta(self, image: Image.Image, variants=4):
        """Average safe test-time augmentations for a more stable classification."""
        if self.model is None: raise ModelError(f'{self.config.slot} ยังไม่พร้อม: {self.error}')
        started=time.perf_counter()
        candidates=[image, ImageOps.mirror(image), ImageEnhance.Brightness(image).enhance(0.92), ImageEnhance.Contrast(image).enhance(1.08), ImageEnhance.Brightness(image).enhance(1.08), ImageEnhance.Contrast(image).enhance(0.94)][:variants]
        batch=np.concatenate([self.preprocess(x) for x in candidates],axis=0)
        with self.lock: raw_batch=self._prediction_matrix(self.model.predict(batch,verbose=0))
        matrix=np.vstack([self._class_probabilities(raw) for raw in raw_batch])
        probs=matrix.mean(axis=0); index=int(np.argmax(probs))
        scores={name:float(probs[i]) for i,name in enumerate(self.config.classes)}
        agreement=float(np.mean(np.argmax(matrix,axis=1)==index))
        return {'class_name':self.config.classes[index],'confidence':float(probs[index]),'scores':scores,'tta_variants':len(candidates),'tta_agreement':agreement,'duration_ms':round((time.perf_counter()-started)*1000,2),'model':self.metadata()}

    def _prediction_matrix(self, raw):
        """รองรับ Keras output แบบ tensor, dict และ single-head list อย่างปลอดภัย."""
        if isinstance(raw, dict):
            preferred=('predictions','prediction','output','outputs','logits','probabilities','probs')
            raw=next((raw[k] for k in preferred if k in raw), next(iter(raw.values()), None))
        if isinstance(raw,(list,tuple)):
            if len(raw)!=1: raise ModelError(f'{self.config.slot} ต้องเป็น single-output classifier')
            raw=raw[0]
        arr=np.asarray(raw,dtype=np.float64)
        if arr.size==0 or not np.all(np.isfinite(arr)): raise ModelError('model output ว่างหรือมี NaN/Inf')
        if arr.ndim==0: arr=arr.reshape(1,1)
        elif arr.ndim==1: arr=arr.reshape(1,-1)
        elif arr.ndim>2: arr=arr.reshape(arr.shape[0],-1)
        return arr

    def _class_probabilities(self, raw):
        probs=self._probabilities(raw)
        expected=len(self.config.classes)
        if probs.size==expected:
            return probs
        if probs.size==6 and self._legacy_stage_adapter_enabled():
            try: ignored=int(os.getenv('MODEL2_LEGACY_6_NON_MANGO_INDEX','0'))
            except ValueError: raise ModelError('MODEL2_LEGACY_6_NON_MANGO_INDEX ต้องเป็นเลข 0-5')
            if ignored not in range(6): raise ModelError('MODEL2_LEGACY_6_NON_MANGO_INDEX ต้องอยู่ระหว่าง 0-5')
            kept=np.delete(probs,ignored)
            total=float(kept.sum())
            return kept/total if total>0 else np.ones(5,dtype=np.float64)/5
        raise ModelError(f'{self.config.slot} ต้อง output {expected} class เท่านั้น แต่ได้ {probs.size}')

    def _probabilities(self, raw):
        raw=np.atleast_1d(raw).astype(np.float64); mode=self.config.output_mode.lower()
        if raw.size == 1:
            value=float(raw[0]); positive=1/(1+np.exp(-value)) if mode=='logits' or value<0 or value>1 else value
            if self.config.binary_positive_index not in (0,1): raise ModelError('binary_positive_index ต้องเป็น 0 หรือ 1')
            probs=np.zeros(2,dtype=np.float64); probs[self.config.binary_positive_index]=positive; probs[1-self.config.binary_positive_index]=1-positive
            return probs
        if mode=='logits' or (mode=='auto' and (np.any(raw<0) or not np.isclose(raw.sum(),1,atol=.02))):
            exp=np.exp(raw-np.max(raw)); return exp/exp.sum()
        total=raw.sum(); return raw/total if total>0 else np.ones_like(raw)/raw.size

    def replace(self, temporary_path: Path, original_filename=None, display_name=None):
        """Validate first, then atomically replace this slot only."""
        suffix=temporary_path.suffix.lower()
        if suffix not in ('.h5','.keras'): raise ModelError('รองรับเฉพาะ .h5 และ .keras')
        original=self.config.path.parent / f'{self.config.slot}{suffix}'; backup=original.with_suffix(suffix+'.bak'); original.parent.mkdir(parents=True,exist_ok=True)
        previous_meta=self.meta_path.read_text(encoding='utf-8') if self.meta_path.exists() else None
        previous_pointer=self.pointer.read_text(encoding='utf-8') if self.pointer.exists() else None
        if original.exists(): original.replace(backup)
        try:
            temporary_path.replace(original); self.pointer.write_text(original.name,encoding='utf-8'); self.active_path=original
            if not self.load(): raise ModelError(self.error or 'โหลดโมเดลใหม่ไม่สำเร็จ')
            self._write_saved_metadata((display_name or '').strip(),original_filename or original.name)
            if backup.exists(): backup.unlink()
        except Exception:
            if original.exists(): original.unlink()
            if backup.exists(): backup.replace(original)
            if previous_meta is None: self.meta_path.unlink(missing_ok=True)
            else: self.meta_path.write_text(previous_meta,encoding='utf-8')
            if previous_pointer is None: self.pointer.unlink(missing_ok=True)
            else: self.pointer.write_text(previous_pointer,encoding='utf-8')
            self.active_path=self._resolve_active_path()
            self.load(); raise
