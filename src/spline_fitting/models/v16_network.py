"""V16: dense proposals, structured one-shot selection and conditioned fitting.

No solver or discrete search is hidden in ``forward_deployment``.  New runs may
use the curve-adaptive ``beta + probability-mass Top-K`` policy recovered from
v10--v15; historical v16 checkpoints retain their literal 0.5 policy through
explicit constructor defaults.  A selected subset conditions both the parameter
gaps and the surviving knot locations.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .candidate_knot_head import CandidateKnotHead
from .geometry_encoder import GeometryEncoder
from .knot_head import KnotHead
from .parameter_head import ParameterHead


class _SelectionBlock(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.self_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(3)])
        self.feed_forward = nn.Sequential(
            nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width)
        )

    def forward(self, tokens: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        update, _ = self.self_attention(tokens, tokens, tokens, need_weights=False)
        tokens = self.norms[0](tokens + update)
        update, _ = self.cross_attention(tokens, memory, memory, need_weights=False)
        tokens = self.norms[1](tokens + update)
        return self.norms[2](tokens + self.feed_forward(tokens))


class V16CandidateSelectionNetwork(nn.Module):
    """One proposal pass, one structured mask and one joint subset decoder."""

    def __init__(
        self,
        point_dim: int = 2,
        degree: int = 3,
        hidden_dim: int = 128,
        encoder_layers: int = 3,
        max_internal_knots: int = 56,
        min_parameter_gap: float = 1e-5,
        min_knot_gap: float = 1e-4,
        attention_heads: int = 4,
        selector_layers: int = 2,
        mse_tolerance: float = 2.5e-5,
        parameter_residual_limit: float = 0.5,
        subset_parameter_residual_limit: float = 0.25,
        relocation_blend: float = 0.25,
        one_shot_selection_policy: str = "threshold",
        one_shot_adaptive_threshold: bool = False,
        one_shot_safety_sigma: float = 0.0,
        one_shot_safety_knots: int = 0,
        one_shot_coverage_bins: int = 0,
        min_selected_knots: int = 0,
        initial_keep_fraction: float = 0.95,
        structure_mode: str = "candidate_pruning_one_shot",
    ) -> None:
        super().__init__()
        if point_dim < 1 or degree < 1 or hidden_dim < 4:
            raise ValueError("point_dim/degree must be positive and hidden_dim >= 4")
        if attention_heads < 1 or hidden_dim % attention_heads:
            raise ValueError("attention_heads must divide hidden_dim")
        if encoder_layers < 1 or selector_layers < 1 or max_internal_knots < 1:
            raise ValueError("layer and candidate counts must be positive")
        for name, value in {
            "min_parameter_gap": min_parameter_gap,
            "min_knot_gap": min_knot_gap,
            "mse_tolerance": mse_tolerance,
            "parameter_residual_limit": parameter_residual_limit,
            "subset_parameter_residual_limit": subset_parameter_residual_limit,
        }.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if min_knot_gap * (max_internal_knots + 1) >= 1:
            raise ValueError("min_knot_gap leaves no free interval budget")
        if not math.isfinite(relocation_blend) or not 0 <= relocation_blend <= 1:
            raise ValueError("relocation_blend must be in [0,1]")
        if one_shot_selection_policy not in {"threshold", "mass_topk"}:
            raise ValueError(
                "one_shot_selection_policy must be 'threshold' or 'mass_topk'"
            )
        if not isinstance(one_shot_adaptive_threshold, bool):
            raise ValueError("one_shot_adaptive_threshold must be Boolean")
        if not math.isfinite(one_shot_safety_sigma) or one_shot_safety_sigma < 0:
            raise ValueError("one_shot_safety_sigma must be finite and non-negative")
        if (
            not math.isfinite(initial_keep_fraction)
            or not 0.0 < initial_keep_fraction < 1.0
        ):
            raise ValueError("initial_keep_fraction must lie strictly inside (0,1)")
        for name, value in {
            "one_shot_safety_knots": one_shot_safety_knots,
            "one_shot_coverage_bins": one_shot_coverage_bins,
            "min_selected_knots": min_selected_knots,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if min_selected_knots > max_internal_knots:
            raise ValueError("min_selected_knots cannot exceed max_internal_knots")
        if one_shot_coverage_bins > max_internal_knots:
            raise ValueError("one_shot_coverage_bins cannot exceed max_internal_knots")
        if structure_mode != "candidate_pruning_one_shot":
            raise ValueError("v16 requires candidate_pruning_one_shot structure")
        self._config = dict(
            point_dim=point_dim, degree=degree, hidden_dim=hidden_dim,
            encoder_layers=encoder_layers, max_internal_knots=max_internal_knots,
            min_parameter_gap=min_parameter_gap, min_knot_gap=min_knot_gap,
            attention_heads=attention_heads, selector_layers=selector_layers,
            mse_tolerance=mse_tolerance,
            parameter_residual_limit=parameter_residual_limit,
            subset_parameter_residual_limit=subset_parameter_residual_limit,
            relocation_blend=relocation_blend,
            one_shot_selection_policy=one_shot_selection_policy,
            one_shot_adaptive_threshold=one_shot_adaptive_threshold,
            one_shot_safety_sigma=one_shot_safety_sigma,
            one_shot_safety_knots=one_shot_safety_knots,
            one_shot_coverage_bins=one_shot_coverage_bins,
            min_selected_knots=min_selected_knots,
            initial_keep_fraction=initial_keep_fraction,
            structure_mode=structure_mode,
        )
        self.point_dim, self.degree, self.hidden_dim = point_dim, degree, hidden_dim
        self.max_internal_knots = max_internal_knots
        self.structure_mode, self.mse_tolerance = structure_mode, mse_tolerance
        self.min_parameter_gap, self.min_knot_gap = min_parameter_gap, min_knot_gap
        self.subset_parameter_residual_limit = subset_parameter_residual_limit
        self.one_shot_selection_policy = one_shot_selection_policy
        self.one_shot_adaptive_threshold = one_shot_adaptive_threshold
        self.one_shot_safety_sigma = float(one_shot_safety_sigma)
        self.one_shot_safety_knots = int(one_shot_safety_knots)
        self.one_shot_coverage_bins = int(one_shot_coverage_bins)
        self.min_selected_knots = int(min_selected_knots)
        self.initial_keep_fraction = float(initial_keep_fraction)
        self.encoder = GeometryEncoder(point_dim, hidden_dim, encoder_layers)
        self.parameter_head = ParameterHead(
            hidden_dim, min_parameter_gap, "strict", "chord_residual",
            parameter_residual_limit,
        )
        self.candidate_head = CandidateKnotHead(
            hidden_dim, max_internal_knots, min_gap=min_knot_gap,
            attention_heads=attention_heads, local_attention_bandwidth=0.08,
            position_parameterization="bounded_anchor_residual",
        )
        self.tolerance_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.coverage_embedding = nn.Linear(4, hidden_dim)
        self.selection_blocks = nn.ModuleList([
            _SelectionBlock(hidden_dim, attention_heads) for _ in range(selector_layers)
        ])
        self.keep_head = nn.Linear(hidden_dim, 1)
        nn.init.normal_(self.keep_head.weight, std=0.005)
        nn.init.constant_(self.keep_head.bias, math.log(9.0))
        if one_shot_adaptive_threshold:
            self.adaptive_threshold_norm = nn.LayerNorm(hidden_dim)
            self.adaptive_threshold_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
            )
            nn.init.normal_(self.adaptive_threshold_head[-1].weight, std=0.01)
            # Raw scores are centred across candidates below, so beta alone
            # controls the initial cardinality.  The legacy p=0.95 prior is
            # retained by the constructor default for checkpoint compatibility;
            # new trainers can start near the expected source complexity instead
            # of spending many joint epochs backing away from all-keep.
            initial_beta = math.log(
                (1.0 - self.initial_keep_fraction) / self.initial_keep_fraction
            )
            nn.init.constant_(self.adaptive_threshold_head[-1].bias, initial_beta)
        self.subset_geometry = nn.Linear(4, hidden_dim)
        self.survivor_attention = nn.MultiheadAttention(
            hidden_dim, attention_heads, batch_first=True
        )
        self.parameter_attention = nn.MultiheadAttention(
            hidden_dim, attention_heads, batch_first=True
        )
        self.survivor_norm = nn.LayerNorm(hidden_dim)
        self.parameter_norm = nn.LayerNorm(hidden_dim)
        self.parameter_update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )
        self.relocation_update = nn.Linear(hidden_dim, 2)
        nn.init.zeros_(self.parameter_update[-1].weight)
        nn.init.zeros_(self.parameter_update[-1].bias)
        nn.init.zeros_(self.relocation_update.weight)
        nn.init.zeros_(self.relocation_update.bias)
        clipped_blend = min(max(relocation_blend, 1e-6), 1 - 1e-6)
        self.relocation_blend_logit = nn.Parameter(torch.tensor(
            math.log(clipped_blend / (1 - clipped_blend))
        ))

    def get_config(self) -> dict:
        # Selection safety is validation-controlled during joint training.  A
        # checkpoint must serialize the safety values with which it was
        # actually evaluated, rather than the constructor's initial values.
        config = dict(self._config)
        config.update(
            one_shot_safety_sigma=self.one_shot_safety_sigma,
            one_shot_safety_knots=self.one_shot_safety_knots,
        )
        return config

    def set_selection_safety(self, *, sigma: float, knots: int) -> None:
        """Set the one-shot safety reserve without changing network weights."""
        if not math.isfinite(sigma) or sigma < 0:
            raise ValueError("selection safety sigma must be finite and non-negative")
        if isinstance(knots, bool) or not isinstance(knots, int) or knots < 0:
            raise ValueError("selection safety knots must be a non-negative integer")
        self.one_shot_safety_sigma = float(sigma)
        self.one_shot_safety_knots = int(knots)

    def _tolerance(self, points: torch.Tensor, value) -> torch.Tensor:
        tolerance = torch.as_tensor(
            self.mse_tolerance if value is None else value,
            dtype=points.dtype, device=points.device,
        )
        if tolerance.ndim == 0:
            tolerance = tolerance.expand(points.shape[0])
        if tolerance.shape != (points.shape[0],):
            raise ValueError("mse_tolerance must be a scalar or have shape [B]")
        if not torch.isfinite(tolerance).all() or (tolerance <= 0).any():
            raise ValueError("mse_tolerance must be finite and positive")
        return tolerance

    def _check_points(self, points: torch.Tensor) -> None:
        if points.ndim != 3 or points.shape[2] != self.point_dim:
            raise ValueError("points must have shape [B,M,point_dim]")
        if points.shape[0] == 0 or points.shape[1] <= self.degree:
            raise ValueError("a nonempty batch with M > degree is required")
        weight = next(self.parameters())
        if not points.is_floating_point() or points.dtype != weight.dtype:
            raise ValueError("points and model must have the same floating dtype")
        if points.device != weight.device:
            raise ValueError("points and model must share a device")
        if not torch.isfinite(points).all():
            raise ValueError("points must be finite")
        if self.min_parameter_gap * (points.shape[1] - 1) >= 1:
            raise ValueError("min_parameter_gap leaves no free interval budget")

    def encode_candidates(self, points: torch.Tensor, mse_tolerance=None) -> dict:
        self._check_points(points)
        tolerance = self._tolerance(points, mse_tolerance)
        local, global_features = self.encoder(points)
        lengths = (points[:, 1:] - points[:, :-1]).norm(dim=-1)
        lengths = lengths.clamp_min(torch.finfo(points.dtype).eps)
        chord = torch.cat([
            lengths.new_zeros(lengths.shape[0], 1),
            (lengths / lengths.sum(-1, keepdim=True)).cumsum(-1),
        ], dim=-1)
        chord = torch.cat([chord[:, :-1], chord.new_ones(chord.shape[0], 1)], -1)
        params = self.parameter_head(local, global_features, reference_params=chord)["params"]
        proposal = self.candidate_head(global_features, local, params)
        knots = proposal["candidate_knots"]
        boundaries = torch.cat([knots.new_zeros(knots.shape[0], 1), knots,
                                knots.new_ones(knots.shape[0], 1)], -1)
        gaps = boundaries.diff(dim=-1)
        nearest = (knots.unsqueeze(-1) - params.unsqueeze(1)).abs().amin(-1)
        coverage = torch.stack([knots, gaps[:, :-1], gaps[:, 1:], nearest], -1)
        tolerance_features = self.tolerance_embedding(
            (tolerance / self.mse_tolerance).log().unsqueeze(-1)
        )
        memory = local + KnotHead._sinusoidal_position_encoding(params, self.hidden_dim)
        tokens = (proposal["candidate_tokens"] + self.coverage_embedding(coverage)
                  + tolerance_features.unsqueeze(1))
        for block in self.selection_blocks:
            tokens = block(tokens, memory)
        raw_importance = self.keep_head(tokens).squeeze(-1)
        if self.one_shot_adaptive_threshold:
            centered_importance = raw_importance - raw_importance.mean(
                dim=-1, keepdim=True
            )
            threshold_context = self.adaptive_threshold_norm(
                tokens.mean(dim=1) + global_features + tolerance_features
            )
            adaptive_threshold = self.adaptive_threshold_head(
                threshold_context
            ).squeeze(-1)
        else:
            centered_importance = raw_importance
            adaptive_threshold = raw_importance.new_zeros(raw_importance.shape[0])
        logits = centered_importance - adaptive_threshold.unsqueeze(-1)
        probabilities = logits.sigmoid()
        uncertainty = (
            probabilities.mul(1.0 - probabilities).sum(dim=-1).clamp_min(0.0).sqrt()
        )
        requested_count_score = (
            probabilities.sum(dim=-1)
            + self.one_shot_safety_sigma * uncertainty
            + self.one_shot_safety_knots
        )
        return dict(
            points=points, proposal_params=params, proposal_internal_knots=knots,
            candidate_tokens=tokens, local_features=local,
            global_features=global_features, raw_keep_importance=raw_importance,
            centered_keep_importance=centered_importance,
            adaptive_keep_threshold=adaptive_threshold,
            adaptive_keep_logit_threshold=adaptive_threshold,
            keep_logits=logits, keep_probabilities=probabilities,
            one_shot_probability_mass=probabilities.sum(dim=-1),
            one_shot_selection_uncertainty=uncertainty,
            one_shot_requested_count_score=requested_count_score,
            tolerance=tolerance,
            tolerance_features=tolerance_features,
        )

    def _coverage_anchors(self, context: dict) -> torch.Tensor:
        probabilities = context["keep_probabilities"]
        anchors = torch.zeros_like(probabilities, dtype=torch.bool)
        if not self.one_shot_coverage_bins:
            return anchors
        positions = context["proposal_internal_knots"]
        for bin_index in range(self.one_shot_coverage_bins):
            lower = bin_index / self.one_shot_coverage_bins
            upper = (bin_index + 1) / self.one_shot_coverage_bins
            in_bin = (positions >= lower) & (
                positions < upper if bin_index + 1 < self.one_shot_coverage_bins
                else positions <= upper
            )
            best = probabilities.masked_fill(~in_bin, float("-inf")).argmax(-1)
            # Scatter into a fresh tensor, then OR.  Writing ``False`` directly
            # into ``anchors`` for an empty bin could erase a valid anchor from
            # an earlier bin when argmax defaults to candidate zero.
            addition = torch.zeros_like(anchors)
            addition.scatter_(
                1, best.unsqueeze(-1), in_bin.any(-1).unsqueeze(-1)
            )
            anchors |= addition
        return anchors

    def select_mask_at_count(
        self, context: dict, requested_count: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the exact deployed ranking/coverage rule at a supplied count."""
        probabilities = context["keep_probabilities"]
        if not isinstance(requested_count, torch.Tensor):
            raise ValueError("requested_count must be a tensor")
        if requested_count.shape != probabilities.shape[:1]:
            raise ValueError("requested_count must have shape [B]")
        if requested_count.device != probabilities.device:
            raise ValueError("requested_count and probabilities must share a device")
        counts = requested_count.to(torch.long).clamp(
            min=self.min_selected_knots, max=probabilities.shape[1]
        )
        # Top-K is performed once. Optional spatial anchors only change the
        # ordering; they do not invoke a spline solve or threshold sweep.
        anchors = self._coverage_anchors(context)
        affordable = anchors.sum(-1) <= counts
        anchors &= affordable.unsqueeze(-1)
        selection_score = probabilities + 2.0 * anchors.to(probabilities.dtype)
        order = torch.argsort(selection_score, dim=-1, descending=True, stable=True)
        rank = torch.empty_like(order)
        rank.scatter_(
            1,
            order,
            torch.arange(order.shape[1], device=order.device).expand_as(order),
        )
        return rank < counts.unsqueeze(-1)

    def constrain_selection_mask(
        self, context: dict, mask: torch.Tensor,
    ) -> torch.Tensor:
        """Repair a training mask so it belongs to the deployed mask family."""
        probabilities = context["keep_probabilities"]
        if mask.shape != probabilities.shape or mask.dtype != torch.bool:
            raise ValueError("mask must be boolean with the candidate [B,K] shape")
        if mask.device != probabilities.device:
            raise ValueError("mask and probabilities must share a device")
        repaired = mask | self._coverage_anchors(context)
        missing = (self.min_selected_knots - repaired.sum(-1)).clamp_min(0)
        if bool((missing > 0).any()):
            order = torch.argsort(
                probabilities.masked_fill(repaired, float("-inf")),
                dim=-1, descending=True, stable=True,
            )
            rank = torch.empty_like(order)
            rank.scatter_(
                1, order,
                torch.arange(order.shape[1], device=order.device).expand_as(order),
            )
            repaired |= rank < missing.unsqueeze(-1)
        return repaired

    def select_mask(self, context: dict) -> torch.Tensor:
        probabilities = context["keep_probabilities"]
        if self.one_shot_selection_policy == "threshold":
            return probabilities >= 0.5

        requested_score = context.get("one_shot_requested_count_score")
        if requested_score is None:
            uncertainty = (
                probabilities.mul(1.0 - probabilities)
                .sum(dim=-1)
                .clamp_min(0.0)
                .sqrt()
            )
            requested_score = (
                probabilities.sum(dim=-1)
                + self.one_shot_safety_sigma * uncertainty
                + self.one_shot_safety_knots
            )
        requested_count = torch.where(
            requested_score < 0.5,
            torch.zeros_like(requested_score, dtype=torch.long),
            torch.ceil(requested_score).to(torch.long),
        )
        return self.select_mask_at_count(context, requested_count)

    @staticmethod
    def _neighbors(values: torch.Tensor, mask: torch.Tensor) -> tuple:
        """Nearest *selected* neighbors, including endpoint boundaries."""
        batch, count = values.shape
        ids = torch.arange(count, device=values.device).expand(batch, -1)
        previous = torch.where(mask, ids + 1, 0).cummax(-1).values
        previous = torch.cat([previous.new_zeros(batch, 1), previous[:, :-1]], -1)
        following = torch.where(mask, ids + 1, count + 1).flip(-1).cummin(-1).values.flip(-1)
        following = torch.cat([following[:, 1:], following.new_full((batch, 1), count + 1)], -1)
        boundaries = torch.cat([values.new_zeros(batch, 1), values, values.new_ones(batch, 1)], -1)
        return boundaries.gather(1, previous), boundaries.gather(1, following)

    @staticmethod
    def _warp_knots(knots: torch.Tensor, old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
        right = torch.searchsorted(old.contiguous(), knots.contiguous(), right=True)
        right = right.clamp(1, old.shape[1] - 1)
        left = right - 1
        old_left, old_right = old.gather(1, left), old.gather(1, right)
        fraction = (knots - old_left) / (old_right - old_left)
        return new.gather(1, left) + fraction * (new.gather(1, right) - new.gather(1, left))

    def decode_subset(self, context: dict, keep_mask: torch.Tensor) -> dict:
        knots, tokens = context["proposal_internal_knots"], context["candidate_tokens"]
        if keep_mask.shape != knots.shape or keep_mask.dtype != torch.bool:
            raise ValueError("keep_mask must be boolean with shape [B,Kc]")
        if keep_mask.device != knots.device:
            raise ValueError("keep_mask and context must share a device")
        batch, count = knots.shape
        kept_count = keep_mask.sum(-1)
        rank = keep_mask.cumsum(-1).to(knots.dtype) / (kept_count.unsqueeze(-1) + 1)
        left, right = self._neighbors(knots, keep_mask)
        geometry = torch.stack([
            knots - left, right - knots, rank,
            (kept_count.to(knots.dtype) / count).unsqueeze(-1).expand_as(knots),
        ], -1)
        subset_tokens = tokens + self.subset_geometry(geometry)
        # The sentinel is visible ONLY for empty subsets. Unselected candidate
        # tokens can never serve as keys/values for the actual subset decoder.
        memory = torch.cat([subset_tokens, subset_tokens.new_zeros(batch, 1, self.hidden_dim)], 1)
        padding = torch.cat([~keep_mask, (kept_count > 0).unsqueeze(-1)], -1)
        survivor_update, _ = self.survivor_attention(
            subset_tokens, memory, memory, key_padding_mask=padding, need_weights=False
        )
        survivors = self.survivor_norm(subset_tokens + survivor_update)
        old_params = context["proposal_params"]
        point_queries = (context["local_features"] + context["tolerance_features"].unsqueeze(1)
                         + KnotHead._sinusoidal_position_encoding(old_params, self.hidden_dim))
        point_update, _ = self.parameter_attention(
            point_queries, memory, memory, key_padding_mask=padding, need_weights=False
        )
        raw = self.parameter_update(self.parameter_norm(point_queries + point_update))[:, :-1, 0]
        limit = self.subset_parameter_residual_limit
        correction = limit * torch.tanh((raw - raw.mean(-1, keepdim=True)) / limit)
        free = (old_params.diff(dim=-1) - self.min_parameter_gap).clamp_min(0)
        weights = free * correction.exp()
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
        gaps = self.min_parameter_gap + (1 - self.min_parameter_gap * free.shape[1]) * weights
        params = torch.cat([gaps.new_zeros(batch, 1), gaps.cumsum(-1)[:, :-1], gaps.new_ones(batch, 1)], -1)
        warped = self._warp_knots(knots, old_params, params)
        residual = self.relocation_update(survivors)
        blend = (self.relocation_blend_logit + residual[..., 1]).sigmoid()
        base = warped * (1 - blend) + rank * blend
        left, right = self._neighbors(base, keep_mask)
        relocated = base + 0.45 * (right - left).clamp_min(0) * residual[..., 0].tanh()
        # Project only selected intervals onto their feasible simplex. The
        # available intervals span removed cells, not the original anchor cells.
        rows = []
        for row in range(batch):
            ids = keep_mask[row].nonzero(as_tuple=False).flatten()
            chosen = relocated[row, ids]
            boundaries = torch.cat([chosen.new_zeros(1), chosen, chosen.new_ones(1)])
            gap_free = (boundaries.diff() - self.min_knot_gap).clamp_min(0)
            normalized = gap_free / gap_free.sum().clamp_min(torch.finfo(knots.dtype).tiny)
            selected_gaps = self.min_knot_gap + (1 - self.min_knot_gap * (ids.numel() + 1)) * normalized
            chosen = selected_gaps.cumsum(0)[:-1]
            rows.append(warped[row].scatter(0, ids, chosen))
        internal = torch.stack(rows)
        probabilities = context["keep_probabilities"]
        return dict(
            params=params, internal_knots=internal, learned_keep_mask=keep_mask,
            final_hard_keep_mask=keep_mask,
            knot_mask=keep_mask, proposal_params=old_params,
            proposal_internal_knots=knots, activity=probabilities,
            keep_probabilities=probabilities, keep_probability=probabilities,
            activity_probability_logits=context["keep_logits"],
            predicted_knot_count=kept_count,
            expected_knot_count=probabilities.sum(-1),
            raw_keep_importance=context["raw_keep_importance"],
            adaptive_keep_threshold=context["adaptive_keep_threshold"],
            adaptive_keep_logit_threshold=context["adaptive_keep_threshold"],
            one_shot_selected_count=kept_count,
            one_shot_probability_mass=context["one_shot_probability_mass"],
            one_shot_selection_uncertainty=context["one_shot_selection_uncertainty"],
            one_shot_requested_count_score=context["one_shot_requested_count_score"],
            mse_tolerance=context["tolerance"],
            parameter_gaps=gaps, warped_proposal_internal_knots=warped,
            relocation_blend=blend,
        )

    def forward_deployment(self, points: torch.Tensor, mse_tolerance=None) -> dict:
        context = self.encode_candidates(points, mse_tolerance=mse_tolerance)
        return self.decode_subset(context, self.select_mask(context))

    def forward(self, points: torch.Tensor, mse_tolerance=None) -> dict:
        return self.forward_deployment(points, mse_tolerance=mse_tolerance)
