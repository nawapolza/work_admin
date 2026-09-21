import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')

def csv(name, default): return [x.strip() for x in os.getenv(name, default).split(',') if x.strip()]
def size(name, default='224,224'):
    value = csv(name, default)
    if len(value) != 2: raise ValueError(f'{name} must be WIDTH,HEIGHT')
    return tuple(map(int, value))
def model_path(name, default):
    path = Path(os.getenv(name, default))
    return path if path.is_absolute() else ROOT / path

@dataclass(frozen=True)
class ModelConfig:
    slot: str
    path: Path
    classes: list[str]
    input_size: tuple[int, int]
    color_mode: str
    preprocess: str
    output_mode: str
    binary_positive_index: int

MODEL1 = ModelConfig('model1', model_path('MODEL1_PATH','backend/models/mango_detector.h5'), csv('MODEL1_CLASSES','mango,non_mango'), size('MODEL1_INPUT_SIZE'), os.getenv('MODEL1_COLOR_MODE','rgb'), os.getenv('MODEL1_PREPROCESS','rescale'), os.getenv('MODEL1_OUTPUT_MODE','auto'), int(os.getenv('MODEL1_BINARY_POSITIVE_INDEX','1')))
MODEL2 = ModelConfig('model2', model_path('MODEL2_PATH','backend/models/ripeness_classifier.h5'), csv('MODEL2_CLASSES','stage1,stage2,stage3,stage4,stage5'), size('MODEL2_INPUT_SIZE'), os.getenv('MODEL2_COLOR_MODE','rgb'), os.getenv('MODEL2_PREPROCESS','rescale'), os.getenv('MODEL2_OUTPUT_MODE','auto'), int(os.getenv('MODEL2_BINARY_POSITIVE_INDEX','1')))

MANGO_CLASS = os.getenv('MODEL1_MANGO_CLASS','mango')
MANGO_THRESHOLD = float(os.getenv('MODEL1_THRESHOLD','0.70'))
MAX_UPLOAD_MB = int(os.getenv('MAX_UPLOAD_MB','0'))
MAX_IMAGE_PIXELS = int(os.getenv('MAX_IMAGE_PIXELS','80000000'))
MODEL1_TTA = os.getenv('MODEL1_TTA','true').lower() == 'true'
MODEL1_TTA_VARIANTS = max(1, min(6, int(os.getenv('MODEL1_TTA_VARIANTS','4'))))
MODEL2_TTA = os.getenv('MODEL2_TTA','false').lower() == 'true'
MODEL2_TTA_VARIANTS = max(1, min(6, int(os.getenv('MODEL2_TTA_VARIANTS','4'))))
MODEL2_VISUAL_WEIGHT = max(0.0, min(0.45, float(os.getenv('MODEL2_VISUAL_WEIGHT','0.0'))))
