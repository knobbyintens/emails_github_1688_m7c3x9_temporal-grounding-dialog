"""Temporal grounding encoder for multimodal dialogue."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class TemporalGroundingConfig:
    text_dim: int = 768
    vision_dim: int = 512
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1
    max_turns: int = 24


class TemporalAttention(nn.Module):
    """Cross-modal attention that scores text tokens against visual events."""

    def __init__(self, config: TemporalGroundingConfig) -> None:
        super().__init__()
        self.text_proj = nn.Linear(config.text_dim, config.hidden_dim)
        self.vision_proj = nn.Linear(config.vision_dim, config.hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(config.hidden_dim)

    def forward(
        self,
        text_states: torch.Tensor,
        vision_states: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        text = self.text_proj(text_states)
        vision = self.vision_proj(vision_states)
        grounded, weights = self.attention(
            query=text,
            key=vision,
            value=vision,
            key_padding_mask=text_mask,
            need_weights=True,
        )
        grounded = self.norm(grounded + text)
        return grounded, weights


class TemporalGroundingModel(nn.Module):
    """Stacked temporal grounding layers with a dialogue-state readout."""

    def __init__(self, config: Optional[TemporalGroundingConfig] = None) -> None:
        super().__init__()
        self.config = config or TemporalGroundingConfig()
        self.layers = nn.ModuleList(
            [TemporalAttention(self.config) for _ in range(self.config.num_layers)]
        )
        self.dialogue_head = nn.Linear(self.config.hidden_dim, 2)
        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, batch: dict) -> dict:
        text = batch["text_states"]
        vision = batch["vision_states"]
        mask = batch.get("text_mask")
        alignments = []
        for layer in self.layers:
            text, weights = layer(text, vision, mask)
            alignments.append(weights)
        pooled = self.dropout(text.mean(dim=1))
        logits = self.dialogue_head(pooled)
        return {"logits": logits, "alignments": alignments}
