#!/usr/bin/env bash
set -e
python src/inference.py --ckpt ./checkpoints/mtl_whisper_small --audio path/to/file.wav --intensity_method rms
