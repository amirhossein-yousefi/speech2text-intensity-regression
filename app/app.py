import os, numpy as np, gradio as gr, soundfile as sf, torch
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

CKPT = os.environ.get("CHECKPOINT", "openai/whisper-small")
##set the checkpoint path here
CKPT = "C:\\Users\\amiru\\Downloads\\speech-intensity-multitask\\src\\checkpoints\\mtl_whisper_ckpt"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = WhisperForASRAndIntensity.from_pretrained(CKPT).to(device).eval()
processor = WhisperProcessor.from_pretrained(CKPT)
SR = processor.feature_extractor.sampling_rate

def transcribe_and_intensity(audio, method):
    # audio is a tuple (sr, data) or filepath depending on Gradio version
    if isinstance(audio, tuple) and len(audio) == 2:
        sr, x = audio
    else:
        # handle filepath
        x, sr = sf.read(audio)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        import librosa
        x = librosa.resample(x.astype(np.float32), orig_sr=sr, target_sr=SR)
        sr = SR

    inputs = processor(x, sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        pred_ids = model.generate(**inputs, max_new_tokens=225)
        out = model(**inputs, return_dict=True)
        pred_intensity = float(out.intensity_pred.detach().cpu().numpy()[0])

    text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
    measured = lufs_level(x, sr) if method == "lufs" else rms_dbfs(x)

    return {
        "Transcript": text,
        "Predicted Intensity": f"{pred_intensity:.2f} ({'LUFS' if method=='lufs' else 'dBFS'})",
        "Measured (from signal)": f"{measured:.2f} ({'LUFS' if method=='lufs' else 'dBFS'})"
    }

with gr.Blocks(title="ASR + Intensity (Whisper Multi-Task)") as demo:
    gr.Markdown("# 🎙️ ASR + Voice Intensity\nUpload audio to transcribe and estimate loudness.")
    with gr.Row():
        with gr.Column():
            audio = gr.Audio(type="filepath", label="Upload .wav/.flac")
            method = gr.Radio(choices=["rms", "lufs"], value="rms", label="Intensity Method")
            btn = gr.Button("Transcribe + Predict Intensity")
        with gr.Column():
            out = gr.JSON(label="Results")
    btn.click(transcribe_and_intensity, inputs=[audio, method], outputs=out)

if __name__ == "__main__":
    demo.launch()
