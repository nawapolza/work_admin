from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter, ImageOps


def analyze_mango_surface(image: Image.Image) -> dict:
    """Explainable colour/spot evidence. This assists Model 2; it is not a detector."""
    rgb = ImageOps.exif_transpose(image).convert('RGB')
    rgb.thumbnail((512, 512), Image.Resampling.LANCZOS)
    # A centre ellipse reduces background influence without pretending to segment perfectly.
    w, h = rgb.size
    yy, xx = np.ogrid[:h, :w]
    ellipse = ((xx - w / 2) / max(w * .43, 1)) ** 2 + ((yy - h / 2) / max(h * .43, 1)) ** 2 <= 1
    hsv = np.asarray(rgb.convert('HSV'), dtype=np.float32)
    value, saturation, hue = hsv[..., 2] / 255., hsv[..., 1] / 255., hsv[..., 0] / 255. * 360
    useful = ellipse & (saturation > .18) & (value > .10)
    count = int(useful.sum())
    if count < 250:
        return {'available': False, 'reason': 'พื้นที่สีของผลไม้ไม่เพียงพอ', 'suggested_stage': None, 'confidence': 0.0}

    def ratio(mask): return float((useful & mask).sum() / count)
    green = ratio((hue >= 55) & (hue < 165))
    yellow = ratio((hue >= 35) & (hue < 65))
    orange = ratio((hue >= 12) & (hue < 40))
    dark = float((ellipse & (value < .24) & (saturation < .75)).sum() / max(int(ellipse.sum()), 1))

    # Transparent rule-based evidence aligned with stage1..stage5 definitions.
    ripe_colour = yellow + orange
    if green >= .72 and ripe_colour < .20:
        stage, confidence = 'stage1', min(.92, .55 + green * .42)
    elif green >= .48 and ripe_colour < .42:
        stage, confidence = 'stage2', min(.86, .52 + green * .32)
    elif green >= .20 and ripe_colour >= .22:
        stage, confidence = 'stage3', min(.84, .54 + min(green, ripe_colour) * .7)
    elif orange >= .34 or ripe_colour >= .62:
        stage, confidence = ('stage5', min(.85, .55 + dark * 2.5)) if dark >= .055 else ('stage4', min(.89, .56 + ripe_colour * .42))
    else:
        stage, confidence = 'stage3', .42
    return {
        'available': True, 'suggested_stage': stage, 'confidence': round(float(confidence), 4),
        'green_ratio': round(green, 4), 'yellow_ratio': round(yellow, 4),
        'orange_ratio': round(orange, 4), 'black_spot_ratio': round(dark, 4),
        'note': 'คำนวณจากสีบริเวณกึ่งกลางภาพ จุดดำเป็นหลักฐานเสริม ไม่ใช่ข้อสรุปโรคหรือความสุกเพียงอย่างเดียว'
    }


def fuse_stage_prediction(prediction: dict, visual: dict, weight: float) -> dict:
    prediction['keras_class_name'] = prediction['class_name']
    prediction['keras_confidence'] = prediction['confidence']
    prediction['visual_analysis'] = visual
    prediction['fusion_weight'] = 0.0
    if not visual.get('available') or not visual.get('suggested_stage') or weight <= 0:
        prediction['decision_source'] = 'keras_model'
        return prediction
    classes = prediction['model']['classes']
    stage = visual['suggested_stage']
    if stage not in classes:
        prediction['decision_source'] = 'keras_model'
        return prediction
    applied = min(max(weight, 0.0), .45) * visual['confidence']
    model_scores = np.array([prediction['scores'][name] for name in classes], dtype=np.float64)
    visual_scores = np.full(len(classes), (1 - visual['confidence']) / max(len(classes) - 1, 1))
    visual_scores[classes.index(stage)] = visual['confidence']
    fused = (1 - applied) * model_scores + applied * visual_scores
    fused /= fused.sum()
    index = int(np.argmax(fused))
    prediction['class_name'] = classes[index]
    prediction['confidence'] = float(fused[index])
    prediction['scores'] = {name: float(fused[i]) for i, name in enumerate(classes)}
    prediction['fusion_weight'] = round(float(applied), 4)
    prediction['decision_source'] = 'keras_plus_visual_evidence'
    return prediction
