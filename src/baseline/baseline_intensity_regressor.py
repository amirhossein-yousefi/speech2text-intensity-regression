import argparse, numpy as np
from datasets import load_dataset
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import librosa

def feats(x, sr):
    zcr = librosa.feature.zero_crossing_rate(x).mean()
    S = np.abs(librosa.stft(x, n_fft=512, hop_length=160))
    mel = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel + 1e-9)
    f = np.concatenate([[zcr, np.mean(x**2)], mel_db.mean(axis=1), mel_db.std(axis=1)])
    return f.astype(np.float32)

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
    ap.add_argument("--dataset", choices=["librispeech","common_voice"], default="librispeech")
    ap.add_argument("--language", default="en")
    ap.add_argument("--intensity", choices=["rms","lufs"], default="rms")
    args = ap.parse_args()

    if args.dataset == "librispeech":
        ds_tr = load_dataset("openslr/librispeech_asr","clean",split="train.100")
        ds_te = load_dataset("openslr/librispeech_asr","clean",split="test")
    else:
        ds_tr = load_dataset("mozilla-foundation/common_voice_11_0",args.language,split="train+validation")
        ds_te = load_dataset("mozilla-foundation/common_voice_11_0",args.language,split="test")

    def compute_target(arr, sr):
        return lufs_level(arr, sr) if args.intensity=="lufs" else rms_dbfs(arr)

    X_tr, y_tr = [], []
    for e in ds_tr:
        arr = e["audio"]["array"].astype(np.float32)
        sr = e["audio"]["sampling_rate"]
        X_tr.append(feats(arr, sr)); y_tr.append(compute_target(arr, sr))
    X_tr = np.stack(X_tr); y_tr = np.array(y_tr)

    X_te, y_te = [], []
    for e in ds_te:
        arr = e["audio"]["array"].astype(np.float32); sr = e["audio"]["sampling_rate"]
        X_te.append(feats(arr, sr)); y_te.append(compute_target(arr, sr))
    X_te = np.stack(X_te); y_te = np.array(y_te)

    model = Ridge(alpha=1.0).fit(X_tr, y_tr)
    pred = model.predict(X_te)
    rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
    print({"baseline_intensity_rmse": rmse, "n_train": len(y_tr), "n_test": len(y_te)})

if __name__ == "__main__":
    main()
