# DFRot: Achieving Outlier-Free and Massive Activation-Free for Rotated LLMs with Refined Rotation


## Run Scripts

You can get all scripts by:
```python
# Random, Hadamard, Procrustes Scripts
CUDA_VISIBLE_DEVICES=None python3 scripts/generate_scripts.py
# For QuaRot.FP16()
CUDA_VISIBLE_DEVICES=None python3 scripts/separate_scripts.py
# Calibration Dataset
CUDA_VISIBLE_DEVICES=None python3 scripts/generate_calibration.py
# Without Rotation
CUDA_VISIBLE_DEVICES=None python3 scripts/vanilla_scripts.py
```