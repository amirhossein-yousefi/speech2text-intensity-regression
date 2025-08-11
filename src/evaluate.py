import argparse, json, numpy as np, torch
from datasets import load_dataset, DatasetDict
from transformers import WhisperProcessor, GenerationConfig
from jiwer import wer
from src.models.multitask_whisper import WhisperForASRAndIntensity

def rms_dbfs(x, eps: float = 1e-12) -> float:
    import numpy as np
    x = np.asarray(x, dtype=np.float32)
    r = np.sqrt(np.maximum(np.mean(x ** 2), eps))
    db = 20.0 * np.log10(r + eps)
    return float(np.clip(db, -60.0, 0.0))

def lufs_level(x, sr: int):
    try:
        import numpy as np
        import pyloudnorm as pyln
        meter = pyln.Meter(sr)
        lufs = meter.integrated_loudness(np.asarray(x, dtype=np.float32))
        return float(np.clip(lufs, -70.0, 0.0))
    except Exception:
        return rms_dbfs(x)

def normalize_text(example):
    t = example["text"]
    example["text"] = " ".join(str(t).strip().split())
    return example

def postprocess_text(s: str) -> str:
    return " ".join(s.strip().split()).lower()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--dataset", choices=["librispeech","common_voice"], default="librispeech")
    ap.add_argument("--librispeech_config", type=str, default="clean")
    ap.add_argument("--test_split", type=str, default="test")
    ap.add_argument("--language", type=str, default="en")
    ap.add_argument("--intensity_method", choices=["rms","lufs"], default="rms")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhisperForASRAndIntensity.from_pretrained(args.ckpt).to(device).eval()
    processor = WhisperProcessor.from_pretrained(args.ckpt)

    sr = processor.feature_extractor.sampling_rate
    gen = GenerationConfig.from_pretrained(args.ckpt)
    gen.language = args.language
    gen.task = "transcribe"

    # Load dataset (test)
    if args.dataset == "librispeech":
        ds = load_dataset("openslr/librispeech_asr", args.librispeech_config, split=args.test_split)
    else:
        ds = load_dataset("mozilla-foundation/common_voice_11_0", args.language, split="test")

    ds = ds.map(normalize_text)

    pred_texts, ref_texts = [], []
    preds_intensity, refs_intensity = [], []

    for e in ds:
        x = e["audio"]["array"]
        # intensity target
        target = lufs_level(x, sr) if args.intensity_method == "lufs" else rms_dbfs(x)
        refs_intensity.append(target)

        inputs = processor(x, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=225, generation_config=gen)
            out = model(**inputs, return_dict=True)
        text = processor.batch_decode(ids, skip_special_tokens=True)[0]
        pred_texts.append(" ".join(text.strip().split()).lower())
        ref_texts.append(" ".join(str(e["text"]).strip().split()).lower())
        preds_intensity.append(float(out.intensity_pred.detach().cpu().numpy()[0]))

    W = wer(ref_texts, pred_texts)
    rmse = float(np.sqrt(np.mean((np.array(preds_intensity) - np.array(refs_intensity))**2)))
    print(json.dumps({"test_wer": W, "test_intensity_rmse": rmse, "n": len(ds)}, indent=2))

if __name__ == "__main__":
    main()
