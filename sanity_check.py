"""sanity_check.py — Quick verification that all imports and pipeline work correctly."""
import sys, os
sys.path.insert(0, os.path.abspath('.'))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np

print(f'TF version: {tf.__version__}')
gpus = tf.config.list_physical_devices('GPU')
gpu_msg = str(gpus) if gpus else 'None - CPU only'
print(f'GPUs: {gpu_msg}')

from src.config import CLASS_NAMES, NUM_CLASSES, TRAIN_DIR
from src.data_utils import get_split_summary, print_summary, compute_class_weights
from src.preprocessing import build_generators
from src.models import build_custom_cnn, build_mobilenetv2, build_efficientnetb0

summary = get_split_summary()
print_summary(summary)

cw = compute_class_weights(TRAIN_DIR, CLASS_NAMES)
print('Class weights:', cw)

train_gen, valid_gen, test_gen = build_generators('custom')
print('Class indices:', train_gen.class_indices)
print('Train samples:', train_gen.samples)
print('Valid samples:', valid_gen.samples)
print('Test  samples:', test_gen.samples)

model = build_custom_cnn()
dummy = np.zeros((2, 224, 224, 3), dtype='float32')
out = model(dummy, training=False)
print(f'Custom CNN output shape: {out.shape} (expected (2,4))')
assert out.shape == (2, 4), 'Shape mismatch!'

model2 = build_mobilenetv2()
out2 = model2(dummy, training=False)
print(f'MobileNetV2 output shape: {out2.shape}')
assert out2.shape == (2, 4)

model3 = build_efficientnetb0()
out3 = model3(dummy, training=False)
print(f'EfficientNetB0 output shape: {out3.shape}')
assert out3.shape == (2, 4)

print('\nALL SANITY CHECKS PASSED - Ready for training!')
