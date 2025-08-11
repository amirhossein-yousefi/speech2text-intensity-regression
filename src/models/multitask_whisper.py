import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput
from typing import Union

class WhisperForASRAndIntensity(WhisperForConditionalGeneration):
    """
    Whisper + a regression head on the encoder's pooled representation.
    Loss = ASR CE + lambda_intensity * MSE.
    """
    def __init__(self, config, lambda_intensity: float = 1.0):
        super().__init__(config)
        hidden = config.d_model
        self.lambda_intensity = lambda_intensity
        self.intensity_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        input_features=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
        intensity_labels=None,
        output_hidden_states=True,
        return_dict=True,
        **kwargs,
    ) -> Union[Seq2SeqLMOutput, tuple]:
        kwargs.pop("num_items_in_batch", None)
        outputs = super().forward(
            input_features=input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )

        enc_last = outputs.encoder_last_hidden_state  # (B, T, D)
        pooled = enc_last.mean(dim=1)                 # (B, D)
        intensity_pred = self.intensity_head(pooled).squeeze(-1)  # (B,)
        loss = outputs.loss
        intensity_loss = None
        if intensity_labels is not None:
            # keep dtypes aligned (helps if you later train/eval with fp16)
            intensity_labels = intensity_labels.to(intensity_pred.dtype)
            intensity_loss = F.mse_loss(intensity_pred, intensity_labels)
            if loss is None:
                # generation/eval path: no CE loss, so use only intensity term
                loss = self.lambda_intensity * intensity_loss
            else:
                # training (or any path with labels present): combine both
                loss = loss + self.lambda_intensity * intensity_loss

        if not return_dict:
            return (loss, outputs.logits, intensity_pred)

        result = Seq2SeqLMOutput(
            loss=loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=enc_last,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
        # Attach for convenience
        result.intensity_pred = intensity_pred  # type: ignore
        result.intensity_loss = intensity_loss  # type: ignore
        return result
