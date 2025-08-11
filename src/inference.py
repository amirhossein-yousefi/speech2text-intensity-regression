import argparse, json, torch, soundfile as sf, numpy as np
from transformers import WhisperProcessor
from src.models.multitask_whisper import WhisperForASRAndIntensity

def rms_dbfs(x, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=np.float32)
    r = np.sqrt(np.maximum(np.mean(x ** 2), eps))
    db = 20.0 * np.log10(r + eps)
    return float(np.clip(db, -60.0, 0.0))

def lufs_level(x, sr: int):
    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(sr)
        lufs = meter.integrated_loudness(np.asarray(x, dtype=np.float32))
        return float(np.clip(lufs, -70.0, 0.0))
    except Exception:
        return rms_dbfs(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--audio", type=str, required=True, help="path to .wav/.flac")
    ap.add_argument("--intensity_method", choices=["rms","lufs"], default="rms")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhisperForASRAndIntensity.from_pretrained(args.ckpt).to(device).eval()
    processor = WhisperProcessor.from_pretrained(args.ckpt)
    sr = processor.feature_extractor.sampling_rate

    # Load audio
    x, file_sr = sf.read(args.audio)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if file_sr != sr:
        import librosa
        x = librosa.resample(x.astype(np.float32), orig_sr=file_sr, target_sr=sr)

    # Features
    inputs = processor(x, sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        pred_ids = model.generate(**inputs, max_new_tokens=225)
        out = model(**inputs, return_dict=True)
        intensity_pred = float(out.intensity_pred.detach().cpu().numpy()[0])

    text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]

    measured = lufs_level(x, sr) if args.intensity_method=="lufs" else rms_dbfs(x)

    print(json.dumps({
        "transcript": text,
        "predicted_intensity": intensity_pred,
        "measured_intensity": measured,
        "units": "dB (LUFS)" if args.intensity_method=="lufs" else "dBFS (RMS)"
    }, indent=2))

if __name__ == "__main__":
    main()
