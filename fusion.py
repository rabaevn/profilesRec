from __future__ import annotations

import torch
from torch import nn


class FusionHead(nn.Module):
    """Maps concat(query_emb, profile_emb) -> R^out_dim, L2-normalized.

    Both inputs are expected to be L2-normalized already (the encoder pipeline
    normalizes); concatenation gives a 2*dim vector that the head re-projects
    back to the original embedding dimensionality so the output can be scored
    by cosine similarity against the frozen item embeddings.
    """

    def __init__(
        self,
        embed_dim: int,
        head_type: str = "linear",
        mlp_hidden: int = 512,
    ) -> None:
        super().__init__()
        if head_type == "linear":
            self.net: nn.Module = nn.Linear(2 * embed_dim, embed_dim, bias=False)
        elif head_type == "mlp":
            self.net = nn.Sequential(
                nn.Linear(2 * embed_dim, mlp_hidden),
                nn.GELU(),
                nn.Linear(mlp_hidden, embed_dim, bias=False),
            )
        else:
            raise ValueError(f"Unknown head_type={head_type!r}")
        self.embed_dim = embed_dim
        self.head_type = head_type

    def forward(self, query_emb: torch.Tensor, profile_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([query_emb, profile_emb], dim=-1)
        out = self.net(x)
        return torch.nn.functional.normalize(out, dim=-1)


class CfFusionHead(nn.Module):
    """Maps (text_query_emb [D_text], cf_user_emb [D_cf]) -> R^D_text, L2-normalized.

    head_type variants:
      - linear:           Linear([text;cf] -> text), random init.
      - mlp:              Linear -> GELU -> Linear, random init.
      - identity_linear:  Linear([text;cf] -> text) initialized as [I, 0] so the
                          model starts at the text-only baseline and only learns
                          deviations from it.
      - residual_gate:    out = text + alpha * proj(cf), alpha learnable scalar
                          starting at 0. Same "start at baseline" property as
                          identity_linear, but the residual structure is explicit.
                          alpha is unbounded — it can go negative.
    """

    def __init__(
        self,
        text_dim: int,
        cf_dim: int,
        head_type: str = "linear",
        mlp_hidden: int = 512,
    ) -> None:
        super().__init__()
        in_dim = text_dim + cf_dim
        if head_type == "linear":
            self.net: nn.Module = nn.Linear(in_dim, text_dim, bias=False)
        elif head_type == "mlp":
            self.net = nn.Sequential(
                nn.Linear(in_dim, mlp_hidden),
                nn.GELU(),
                nn.Linear(mlp_hidden, text_dim, bias=False),
            )
        elif head_type == "identity_linear":
            self.net = nn.Linear(in_dim, text_dim, bias=False)
            with torch.no_grad():
                self.net.weight.zero_()
                self.net.weight[:, :text_dim] = torch.eye(text_dim)
        elif head_type == "residual_gate":
            self.proj = nn.Linear(cf_dim, text_dim, bias=False)
            self.alpha = nn.Parameter(torch.zeros(1))
        else:
            raise ValueError(f"Unknown head_type={head_type!r}")
        self.text_dim = text_dim
        self.cf_dim = cf_dim
        self.head_type = head_type

    def forward(self, text_emb: torch.Tensor, cf_emb: torch.Tensor) -> torch.Tensor:
        if self.head_type == "residual_gate":
            out = text_emb + self.alpha * self.proj(cf_emb)
        else:
            x = torch.cat([text_emb, cf_emb], dim=-1)
            out = self.net(x)
        return torch.nn.functional.normalize(out, dim=-1)


class LearnableCfTable(nn.Module):
    """Per-user CF embedding table, seeded from SVD and trainable.

    Holds a (num_users, cf_dim) table plus a (num_users,) boolean buffer
    `has_cf` that masks out users with no training co-purchases. Lookups
    return both the embedding and the per-row mask so the caller can zero
    contributions from cold users.
    """

    def __init__(self, init_embeddings: torch.Tensor, has_cf: torch.Tensor) -> None:
        super().__init__()
        assert init_embeddings.dim() == 2
        assert has_cf.shape[0] == init_embeddings.shape[0]
        self.num_users, self.cf_dim = init_embeddings.shape
        self.weight = nn.Parameter(init_embeddings.clone().float())
        self.register_buffer("has_cf", has_cf.to(torch.bool))
        self.register_buffer("init_weight", init_embeddings.clone().float())

    def forward(self, user_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.weight.index_select(0, user_idx)
        mask = self.has_cf.index_select(0, user_idx).float().unsqueeze(-1)
        return emb * mask, mask


class LearnableCfFusionHead(nn.Module):
    """Residual-gate CF fusion with a trainable per-user CF table.

    Forward: out = text_emb + sigmoid(gate_logit) * has_cf_mask * proj(cf_table[user_idx])
    Then L2-normalize. Properties:
      - gate is in [0, 1] (cannot invert the signal)
      - cold users (has_cf=False) contribute zero regardless of gate
      - gate_logit init at large negative -> fusion starts indistinguishable
        from text-only baseline
      - cf_table is trainable, so noisy SVD rows can be refined during training
    """

    def __init__(
        self,
        text_dim: int,
        init_cf_embeddings: torch.Tensor,
        has_cf: torch.Tensor,
        gate_logit_init: float = -6.0,
    ) -> None:
        super().__init__()
        self.text_dim = text_dim
        self.cf_table = LearnableCfTable(init_cf_embeddings, has_cf)
        self.proj = nn.Linear(self.cf_table.cf_dim, text_dim, bias=False)
        # Hard init invariant: at step 0, proj(cf) = 0 so out = text_emb exactly
        # regardless of gate_logit. Gradients still flow through proj from step 1.
        with torch.no_grad():
            self.proj.weight.zero_()
        self.gate_logit = nn.Parameter(torch.tensor([float(gate_logit_init)]))
        self.head_type = "learnable_residual_gate"
        self.cf_dim = self.cf_table.cf_dim

    def gate(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logit)

    def forward(self, text_emb: torch.Tensor, user_idx: torch.Tensor) -> torch.Tensor:
        cf_emb, mask = self.cf_table(user_idx)
        gated = self.gate() * mask * self.proj(cf_emb)
        out = text_emb + gated
        return torch.nn.functional.normalize(out, dim=-1)

    def anchor_loss(self) -> torch.Tensor:
        """L2 distance from SVD init, restricted to users with has_cf=True."""
        diff = self.cf_table.weight - self.cf_table.init_weight
        mask = self.cf_table.has_cf.float().unsqueeze(-1)
        return ((diff * mask) ** 2).sum() / mask.sum().clamp(min=1.0)


class GatedProfileFusionHead(nn.Module):
    """Residual-gate profile fusion with a per-example sigmoid gate over user features.

    Forward: out = q + sigmoid(gate_mlp(LN(features)) + bias) * has_profile * proj(p)
    Then L2-normalize.

    Invariants (mirror LearnableCfFusionHead):
      - proj zero-initialized -> at step 0, out == normalize(q) exactly, regardless of
        gate value or profile content. Gradients still flow from step 1.
      - gate in [0, 1] (cannot invert signal).
      - has_profile mask zeros the residual for examples with no cached profile,
        regardless of gate.
      - retrieval stays a single dot product: fused @ catalog.T.

    Gate features are caller-provided (B, F). Typical columns:
      [log1p(history_len), has_profile, mean_history_log_pop, cos(q, p) * has_profile]
    """

    def __init__(
        self,
        embed_dim: int,
        num_gate_features: int,
        gate_mlp_hidden: int = 16,
        gate_logit_init: float = -6.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_gate_features = num_gate_features
        self.gate_mlp_hidden = gate_mlp_hidden
        self.gate_logit_init = float(gate_logit_init)

        self.feature_norm = nn.LayerNorm(num_gate_features)
        self.gate_mlp = nn.Sequential(
            nn.Linear(num_gate_features, gate_mlp_hidden),
            nn.GELU(),
            nn.Linear(gate_mlp_hidden, 1),
        )
        # Bias-init the final logit so sigmoid starts near 0 (off by default).
        with torch.no_grad():
            self.gate_mlp[-1].bias.fill_(self.gate_logit_init)

        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)
        with torch.no_grad():
            self.proj.weight.zero_()

        self.head_type = "gated_profile"

    def gate_logit(self, features: torch.Tensor) -> torch.Tensor:
        """Return pre-sigmoid gate logit (B, 1). No has_profile mask — used as
        the prediction target of the oracle-supervised BCE auxiliary loss."""
        return self.gate_mlp(self.feature_norm(features))

    def gate(self, features: torch.Tensor, has_profile: torch.Tensor) -> torch.Tensor:
        """Return per-example gate in [0, 1] (B, 1). Zeros where has_profile == 0."""
        return torch.sigmoid(self.gate_logit(features)) * has_profile

    def forward(
        self,
        query_emb: torch.Tensor,
        profile_emb: torch.Tensor,
        has_profile: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        g = self.gate(features, has_profile)  # (B, 1)
        out = query_emb + g * self.proj(profile_emb)
        return torch.nn.functional.normalize(out, dim=-1)


class GatedFusionDataset(torch.utils.data.Dataset):
    """Pre-computed (query, profile, target, hist_feats, has_profile, weight) for gated fusion."""

    def __init__(
        self,
        query_embs: torch.Tensor,
        profile_embs: torch.Tensor,
        target_embs: torch.Tensor,
        hist_feats: torch.Tensor,
        has_profile: torch.Tensor,
        weights: torch.Tensor | None = None,
        y_oracle: torch.Tensor | None = None,
    ) -> None:
        n = query_embs.shape[0]
        assert profile_embs.shape[0] == n
        assert target_embs.shape[0] == n
        assert hist_feats.shape[0] == n
        assert has_profile.shape[0] == n
        self.query_embs = query_embs
        self.profile_embs = profile_embs
        self.target_embs = target_embs
        self.hist_feats = hist_feats
        self.has_profile = has_profile
        if weights is None:
            weights = torch.ones(n, dtype=torch.float32)
        assert weights.shape[0] == n
        self.weights = weights
        if y_oracle is None:
            y_oracle = torch.full((n,), float("nan"), dtype=torch.float32)
        assert y_oracle.shape[0] == n
        self.y_oracle = y_oracle

    def __len__(self) -> int:
        return self.query_embs.shape[0]

    def __getitem__(self, idx: int):
        return (
            self.query_embs[idx],
            self.profile_embs[idx],
            self.target_embs[idx],
            self.hist_feats[idx],
            self.has_profile[idx],
            self.weights[idx],
            self.y_oracle[idx],
        )


class FusionTripleDataset(torch.utils.data.Dataset):
    """Pre-computed (query_emb, profile_emb, target_emb) triples for fusion training."""

    def __init__(
        self,
        query_embs: torch.Tensor,
        profile_embs: torch.Tensor,
        target_embs: torch.Tensor,
    ) -> None:
        assert query_embs.shape[0] == profile_embs.shape[0] == target_embs.shape[0]
        self.query_embs = query_embs
        self.profile_embs = profile_embs
        self.target_embs = target_embs

    def __len__(self) -> int:
        return self.query_embs.shape[0]

    def __getitem__(self, idx: int):
        return (
            self.query_embs[idx],
            self.profile_embs[idx],
            self.target_embs[idx],
        )


def info_nce_loss(
    fused: torch.Tensor,
    target_emb: torch.Tensor,
    temperature: float,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """InfoNCE with in-batch negatives. fused / target_emb are (B, D) L2-normalized.

    When `weights` (B,) is provided, returns weighted-mean per-example cross-entropy
    (numerator only — negative distribution untouched). Weights need not sum to 1;
    they're normalized internally so absolute LR scale is unchanged.
    """
    logits = fused @ target_emb.T  # (B, B)
    labels = torch.arange(fused.shape[0], device=fused.device)
    if weights is None:
        return torch.nn.functional.cross_entropy(logits / temperature, labels)
    per_ex = torch.nn.functional.cross_entropy(
        logits / temperature, labels, reduction="none"
    )
    return (weights * per_ex).sum() / weights.sum().clamp(min=1.0)
