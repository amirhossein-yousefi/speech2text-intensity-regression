PY=python

# Training (LibriSpeech clean-100)
train:
	$(PY) src/train_multitask_whisper.py --model_id openai/whisper-small \
		--dataset librispeech --librispeech_config clean \
		--train_split train.100 --eval_split validation --test_split test \
		--language en --intensity_method rms \
		--epochs 3 --batch_size 8 --grad_accum 2 --lr 1e-5 --fp16 \
		--output_dir ./checkpoints/mtl_whisper_small

# Evaluate a saved checkpoint on test split
evaluate:
	$(PY) src/evaluate.py --ckpt ./checkpoints/mtl_whisper_small \
		--dataset librispeech --language en --intensity_method rms

# Baseline intensity
baseline:
	$(PY) src/baseline/baseline_intensity_regressor.py --dataset librispeech --language en --intensity rms

# Run the app (point to your checkpoint)
app:
	CHECKPOINT=./checkpoints/mtl_whisper_small $(PY) app/app.py
