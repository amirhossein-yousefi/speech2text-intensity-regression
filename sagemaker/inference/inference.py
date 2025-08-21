# sagemaker/inference.py
import os, io, json, base64
import numpy as np
import torch
from typing import Any, Dict
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from scipy.io import wavfile

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LANGUAGE = os.getenv("LANGUAGE", "en")
TASK = os.getenv("TASK", "transcribe")
BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "openai/whisper-small")  # model used if processor files aren’t in model_dir

# --- Helpers -----------------------------------------------------------------

def _to_mono_float32(wave: np.ndarray) -> np.ndarray:
    """Ensure mono, float32, range [-1,1]."""
    if wave.ndim == 2:
        wave = wave.mean(axis=1)
    # Normalize from int to float
    if wave.dtype == np.int16:
        wave = wave.astype(np.float32) / 32768.0
    elif wave.dtype == np.int32:
        wave = wave.astype(np.float32) / 2147483648.0
    elif wave.dtype == np.uint8:
        wave = (wave.astype(np.float32) - 128.0) / 128.0
    else:
        wave = wave.astype(np.float32)
    return wave

def _rms_dbfs(wave: np.ndarray, eps: float = 1e-9) -> float:
    """Simple, robust intensity baseline (RMS→dBFS)."""
    rms = np.sqrt((wave ** 2).mean() + eps)
    return float(20 * np.log10(rms + eps))

# --- SageMaker-required functions --------------------------------------------

def model_fn(model_dir: str) -> Dict[str, Any]:
    """
    Load artifacts. We try to load a fine-tuned checkpoint from `model_dir`.
    If a custom head exists in the repo (src/models/multitask_whisper.py),
    we import and use it; otherwise we fall back to plain Whisper + RMS intensity.
    """
    # Load processor (prefer artifacts saved with your fine-tuned checkpoint)
    try:
        processor = WhisperProcessor.from_pretrained(model_dir, language=LANGUAGE, task=TASK)
    except Exception:
        processor = WhisperProcessor.from_pretrained(BASE_MODEL_ID, language=LANGUAGE, task=TASK)

    # Try custom multi-task model first (if present in the repo)
    model = None
    try:
        from src.models.multitask_whisper import WhisperForASRAndIntensity  # your custom head
        try:
            model = WhisperForASRAndIntensity.from_pretrained(model_dir)
        except Exception:
            model = WhisperForASRAndIntensity.from_pretrained(BASE_MODEL_ID)
            # If a state dict was saved in model_dir, load it loosely
            sd_path = os.path.join(model_dir, "pytorch_model.bin")
            if os.path.exists(sd_path):
                sd = torch.load(sd_path, map_location="cpu")
                model.load_state_dict(sd, strict=False)
        using_custom_head = True
    except Exception:
        # Fallback: vanilla Whisper for ASR
        model = WhisperForConditionalGeneration.from_pretrained(model_dir if os.path.exists(os.path.join(model_dir, "config.json")) else BASE_MODEL_ID)
        using_custom_head = False

    model.to(DEVICE).eval()
    return {"model": model, "processor": processor, "using_custom_head": using_custom_head}


def input_fn(request_body: bytes, content_type: str) -> Dict[str, Any]:
    """
    Accepts either:
      - application/json: {"audio_base64": "..."} (wav PCM16) or {"waveform": [...], "sr": 16000}
      - audio/wav: raw wav bytes
    """
    if content_type == "application/json":
        payload = json.loads(request_body.decode("utf-8"))
        if "audio_base64" in payload:
            audio_bytes = base64.b64decode(payload["audio_base64"])
            sr, wave = wavfile.read(io.BytesIO(audio_bytes))
            wave = _to_mono_float32(wave)
            return {"wave": wave, "sr": int(sr)}
        elif "waveform" in payload and "sr" in payload:
            wave = np.array(payload["waveform"], dtype=np.float32)
            return {"wave": _to_mono_float32(wave), "sr": int(payload["sr"])}
        else:
            raise ValueError("JSON must include 'audio_base64' or both 'waveform' and 'sr'.")
    elif content_type in ("audio/wav", "audio/x-wav"):
        sr, wave = wavfile.read(io.BytesIO(request_body))
        wave = _to_mono_float32(wave)
        return {"wave": wave, "sr": int(sr)}
    else:
        raise ValueError(f"Unsupported content_type: {content_type}")


@torch.inference_mode()
def predict_fn(inputs: Dict[str, Any], model_bundle: Dict[str, Any], context=None) -> Dict[str, Any]:
    model = model_bundle["model"]
    processor = model_bundle["processor"]
    using_custom_head = model_bundle["using_custom_head"]

    wave, sr = inputs["wave"], inputs["sr"]

    # Feature extraction for Whisper
    processed = processor(audio=wave, sampling_rate=sr, return_tensors="pt")
    input_features = processed.input_features.to(DEVICE)

    # Forced decoder prompt (language + task)
    forced = processor.get_decoder_prompt_ids(language=LANGUAGE, task=TASK)

    # Transcript
    generated_ids = model.generate(input_features, forced_decoder_ids=forced, max_new_tokens=225)
    transcript = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Intensity (custom head if available; else RMS baseline)
    intensity_dbfs = _rms_dbfs(wave)
    if using_custom_head:
        try:
            # Many custom heads expose an 'intensity' in the forward return dict
            outputs = model(input_features=input_features, return_dict=True)
            if hasattr(outputs, "intensity"):
                val = outputs.intensity
                if isinstance(val, (list, tuple)):
                    val = val[0]
                intensity_dbfs = float(val.detach().cpu().view(-1)[0].item())
        except Exception:
            pass

    return {"transcript": transcript, "intensity_dbfs": intensity_dbfs}


def output_fn(prediction: Dict[str, Any], accept: str) -> bytes:
    body = json.dumps(prediction, ensure_ascii=False)
    return body.encode("utf-8")
