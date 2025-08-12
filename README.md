# Speech-to-Text + Voice Intensity (Multi-Task Whisper)

An end-to-end project that **transcribes speech** and **predicts voice intensity** (loudness) **simultaneously** using a multi‑task head on top of Whisper’s encoder. Includes:

- ✅ Multi-task training (ASR + intensity regression head)
- ✅ Evaluation on validation & test (WER + intensity RMSE)
- ✅ A baseline intensity regressor for comparison
- ✅ An interactive Gradio app for demo
- ✅ Dockerfile, Makefile, and clean repo structure

> Intensity target can be **RMS in dBFS** (simple, robust) or **LUFS** (perceived loudness, ITU-R BS.1770) via `pyloudnorm`.

---

## Repo Structure

```
speech-intensity-multitask/
├─ app/
│  └─ app.py                        # Gradio app for inference/demo
├─ scripts/
├─ src/
│  ├─ baseline/
│  │  └─ baseline_intensity_regressor.py
│  ├─ models/
│  │  └─ multitask_whisper.py       # Whisper + regression head
│  ├─ evaluate.py                   # Evaluate a checkpoint on test split
│  ├─ inference.py                  # CLI inference on a single audio file
│  └─ train_multitask_whisper.py    # Train + validate + test
├─ .gitignore
├─ Dockerfile
├─ LICENSE
├─ Makefile
└─ requirements.txt
```

---

## Quickstart

> Python 3.10+ recommended. If you have a GPU, ensure a matching PyTorch + CUDA wheel is installed.

```bash
pip install -r requirements.txt
```

### Train (LibriSpeech clean-100)
```bash
python src/train_multitask_whisper.py \
  --model_id openai/whisper-small \
  --dataset librispeech --librispeech_config clean \
  --train_split train.100 --eval_split validation --test_split test \
  --language en --intensity_method rms \
  --epochs 3 --batch_size 8 --grad_accum 2 --lr 1e-5 --fp16 \
  --output_dir ./checkpoints/mtl_whisper_small
```

### Evaluate on Test
```bash
python src/evaluate.py \
  --ckpt ./checkpoints/mtl_whisper_small \
  --dataset librispeech --language en --intensity_method rms
```

### Baseline Intensity Regressor
```bash
python src/baseline/baseline_intensity_regressor.py \
  --dataset librispeech --language en --intensity rms
```

### Run the App
```bash
CHECKPOINT=./checkpoints/mtl_whisper_small python app/app.py
```
Then open the printed Gradio URL. Upload a `.wav`/`.flac` and see the transcript + intensity.

---

## Datasets

Two easy options via 🤗 Datasets:
- **LibriSpeech** (`openslr/librispeech_asr`): use `"clean"` config, `train.100`, `validation`, `test` splits (English).
- **Common Voice** (`mozilla-foundation/common_voice_11_0`): multilingual; set `--language` (e.g., `en`, `hi`).

We compute intensity targets directly from audio (RMS dBFS or LUFS).

---

## Model

`WhisperForASRAndIntensity` extends `WhisperForConditionalGeneration` by attaching a small **regression head** on the **encoder’s mean‑pooled last hidden state**. Training minimizes:

```
total_loss = asr_ce_loss + λ * mse(intensity)
```

- Units: dBFS for RMS, or LUFS for perceived loudness.
- Set λ via `--lambda_intensity` (default `1.0`).

---

## Metrics

- **ASR**: Word Error Rate (WER) via `jiwer`.
- **Intensity**: RMSE (in dBFS or LUFS).

The training script validates each epoch and computes **test** metrics at the end.
 you can download the finetuned weights fo one epoch from https://drive.google.com/file/d/1PHc2CU3QAux2mDJ4483AJEo6SbmKk-eZ/view?usp=sharing
you can check the logs of training in [training-test-logs](training-test-logs)
---

## Notes & Tips

- Whisper expects **16 kHz** audio features; resampling is handled automatically.
- LUFS requires `pyloudnorm` (optional, recommended for perceptual alignment).
- If you want **human‑annotated intensity/arousal**, consider datasets like MSP‑Podcast or CREMA‑D (adapt labels & licensing accordingly).

---

## License

MIT
