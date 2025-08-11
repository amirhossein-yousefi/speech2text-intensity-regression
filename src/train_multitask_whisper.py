import argparse, os, json, numpy as np, torch, torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List
from datasets import load_dataset, DatasetDict
from transformers import (
    WhisperProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments, GenerationConfig
)
from jiwer import wer
from src.models.multitask_whisper import WhisperForASRAndIntensity

# --------------------
# Loudness utilities
# --------------------
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

# --------------------
# Data collator
# --------------------
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    sampling_rate: int

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        audios = [f["audio"]["array"] for f in features]
        inputs = self.processor.feature_extractor(
            audios, sampling_rate=self.sampling_rate, return_tensors="pt"
        )
        texts = [f["text"] for f in features]
        label_ids = self.processor.tokenizer(texts, padding=True, return_tensors="pt").input_ids
        label_ids[label_ids == self.processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = label_ids

        intensity_vals = [f["intensity_target"] for f in features]
        inputs["intensity_labels"] = torch.tensor(intensity_vals, dtype=torch.float32)
        return inputs

# --------------------
# Preprocessing & metrics
# --------------------
def compute_intensity_target(example, method: str, sr: int):
    x = example["audio"]["array"]
    if method == "lufs":
        example["intensity_target"] = lufs_level(x, sr)
    else:
        example["intensity_target"] = rms_dbfs(x)
    return example

def normalize_text(example):
    t = example["text"]
    example["text"] = " ".join(str(t).strip().split())
    return example

def postprocess_text(s: str) -> str:
    return " ".join(s.strip().split()).lower()

def build_compute_metrics(processor: WhisperProcessor):
    def _compute(eval_pred):
        pred_ids = eval_pred.predictions
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)

        label_ids = eval_pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        pred_str = [postprocess_text(s) for s in pred_str]
        label_str = [postprocess_text(s) for s in label_str]

        w = wer(label_str, pred_str)
        return {"wer": w}
    return _compute

# --------------------
# Custom Trainer for intensity RMSE
# --------------------
class MultiTaskTrainer(Seq2SeqTrainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        gen_kwargs = {"max_new_tokens": 225}
        return super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        out = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        # Compute intensity RMSE
        import numpy as np
        dl = self.get_eval_dataloader(eval_dataset)
        preds, gts = [], []
        model = self.model
        model.eval()
        for batch in dl:
            batch = {k: v.to(self.args.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch, output_hidden_states=False, return_dict=True)
                pred = outputs.intensity_pred.detach().cpu().float().numpy()
            gt = batch["intensity_labels"].detach().cpu().float().numpy()
            preds.append(pred); gts.append(gt)
        preds = np.concatenate(preds); gts = np.concatenate(gts)
        rmse = float(np.sqrt(np.mean((preds - gts) ** 2)))
        out[f"{metric_key_prefix}_intensity_rmse"] = rmse
        self.log({f"{metric_key_prefix}_intensity_rmse": rmse})
        return out

# --------------------
# Main
# --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default="openai/whisper-small")
    ap.add_argument("--dataset", choices=["librispeech","common_voice"], default="librispeech")
    ap.add_argument("--librispeech_config", type=str, default="clean")
    ap.add_argument("--train_split", type=str, default="train.100")
    ap.add_argument("--eval_split", type=str, default="validation")
    ap.add_argument("--test_split", type=str, default="test")
    ap.add_argument("--language", type=str, default="en")
    ap.add_argument("--intensity_method", choices=["rms","lufs"], default="rms")
    ap.add_argument("--output_dir", type=str, default="./checkpoints/mtl_whisper_ckpt")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lambda_intensity", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np = __import__("numpy")

    processor = WhisperProcessor.from_pretrained(args.model_id)
    sr = processor.feature_extractor.sampling_rate
    gen_config = GenerationConfig.from_pretrained(args.model_id)
    gen_config.language = args.language
    gen_config.task = "transcribe"
    # Set forced decoder prompt ids for stability
    forced = None
    try:
        forced = processor.get_decoder_prompt_ids(language=args.language, task="transcribe")
    except Exception:
        try:
            forced = processor.tokenizer.get_decoder_prompt_ids(language=args.language, task="transcribe")
        except Exception:
            forced = None
    if forced is not None:
        gen_config.forced_decoder_ids = forced

    # Load dataset
    if args.dataset == "librispeech":
        ds = DatasetDict()
        ds["train"] = load_dataset("openslr/librispeech_asr", args.librispeech_config, split=args.train_split)
        ds["validation"] = load_dataset("openslr/librispeech_asr", args.librispeech_config, split=args.eval_split)
        ds["test"] = load_dataset("openslr/librispeech_asr", args.librispeech_config, split=args.test_split)
    else:
        lang = args.language
        ds = DatasetDict()
        ds["train"] = load_dataset("mozilla-foundation/common_voice_11_0", lang, split="train+validation")
        ds["validation"] = load_dataset("mozilla-foundation/common_voice_11_0", lang, split="test")
        ds["test"] = ds["validation"]

    # Clean text and attach intensity
    ds = ds.map(normalize_text)
    ds = ds.cast_column("audio", ds["train"].features["audio"])
    method = "lufs" if args.intensity_method == "lufs" else "rms"
    ds = ds.map(lambda e: compute_intensity_target(e, method, sr), num_proc=1)

    # Model
    model = WhisperForASRAndIntensity.from_pretrained(args.model_id, lambda_intensity=args.lambda_intensity)
    model.generation_config = gen_config
    # if hasattr(model, "gradient_checkpointing_enable"):
    #     model.gradient_checkpointing_enable()

    # Collator
    collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, sampling_rate=sr)

    # Training args
    targs = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        fp16=args.fp16,
        logging_steps=50,
        report_to=["none"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_wer",
        greater_is_better=False,
        remove_unused_columns=False,

    )

    trainer = MultiTaskTrainer(
        model=model,
        args=targs,
        data_collator=collator,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=processor,  # processor has save_pretrained
        compute_metrics=build_compute_metrics(processor),
    )

    trainer.train()

    # Test evaluation
    test_metrics = trainer.evaluate(eval_dataset=ds["test"], metric_key_prefix="test")
    print("TEST:", test_metrics)

    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as f:
        json.dump({k: float(v) for k, v in test_metrics.items()}, f, indent=2)

if __name__ == "__main__":
    import torch
    assert torch.cuda.is_available() is True
    main()
