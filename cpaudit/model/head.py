"""The learned score of the Section III controls (torch required).

    s(x, y_k) = s_base(x, y_k) + g( concat(z_x, e_{y_k}) )

g is a small MLP and the only trainable part; the frozen model's score s_base,
image state z_x and option embedding e_y are cached inputs.
`bench.supervised_control.fit_ce` trains it by cross-entropy, with every input
(the full-feature head) or with z_x and s_base zeroed (the answer-only head).
"""
from __future__ import annotations

import numpy as np


def build_head(d_in: int, hidden=(512, 256)):
    import torch.nn as nn
    layers = []
    prev = d_in
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers += [nn.Linear(prev, 1)]
    return nn.Sequential(*layers)


class ScoringHead:
    """Thin wrapper: s = s_base + g(concat(z_x, e_y)).

    Inputs are the cached, frozen features. Only g has gradients.
    """

    def __init__(self, d_z: int, d_e: int, hidden=(512, 256), device="cuda"):
        import torch
        self.torch = torch
        self.device = device
        self.d_z, self.d_e = d_z, d_e     # block boundaries, for weight-norm probes
        self.g = build_head(d_z + d_e, hidden).to(device)

    def scores(self, s_base, z_x, e_y, mask):
        """s_base (B,K); z_x (B,dz); e_y (B,K,de); mask (B,K) -> s (B,K).

        Padded slots are forced to +inf so they never enter a prediction set.
        """
        torch = self.torch
        B, K = s_base.shape
        z_rep = z_x.unsqueeze(1).expand(B, K, z_x.shape[-1])
        feat = torch.cat([z_rep, e_y], dim=-1)          # (B,K,dz+de)
        g = self.g(feat).squeeze(-1)                    # (B,K)
        s = s_base + g
        return s.masked_fill(~mask, float("inf"))

    def parameters(self):
        return self.g.parameters()

    def state_dict(self):
        return self.g.state_dict()


def _pack(sub, device, image_blind):
    """A cache subset as tensors; `image_blind` zeroes z_x."""
    import torch
    zx = np.zeros_like(sub.z_x) if image_blind else sub.z_x
    return dict(
        s_base=torch.tensor(sub.s_base, device=device),
        z_x=torch.tensor(zx, device=device),
        e_y=torch.tensor(sub.e_y, device=device),
        mask=torch.tensor(sub.mask, device=device),
        true=torch.tensor(sub.answer_idx, device=device, dtype=torch.long))
