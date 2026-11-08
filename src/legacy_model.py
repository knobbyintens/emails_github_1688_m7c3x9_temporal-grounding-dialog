"""Archived temporal grounding model used in earlier experiments."""

import torch
import torch.nn as nn


class TemporalGroundingEncoder(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, dialogue_states, visual_states):
        grounded, _ = self.attention(dialogue_states, visual_states, visual_states)
        return self.norm(dialogue_states + grounded)
