#!/usr/bin/env bash
set -e
python src/train_multitask_whisper.py \
  --model_id openai/whisper-small \
  --dataset librispeech --librispeech_config clean \
  --train_split train.100 --eval_split validation --test_split test \
  --language en --intensity_method rms \
  --epochs 3 --batch_size 8 --grad_accum 2 --lr 1e-5 --fp16 \
  --output_dir ./checkpoints/mtl_whisper_small
