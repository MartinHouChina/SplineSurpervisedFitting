from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .knot_head import KnotHead


class InteractivePruningHead(nn.Module):
    """Select and refine candidate knots using a curve-adaptive threshold.

    Each candidate receives its truncated-power coefficient energy, the
    measured/approximated objective increase after deletion, two-sided spacing,
    and local residual features.  Self-attention lets candidates reason about
    redundancy jointly instead of making independent threshold decisions.

    A single global threshold is predicted from the pooled candidate state.  A
    knot is kept according to ``sigmoid(raw_importance - threshold)`` rather
    than an unrelated fixed 0.5 classifier threshold.  The resulting soft keep
    state is pooled back into every candidate before position refinement, which
    couples selection and location in one head.

    Position updates remain bounded to less than half of the available
    neighboring slack.  Consequently ordered inputs remain strictly ordered
    without sorting, preserving candidate/output correspondence.

    With ``one_shot_survivor_relocation`` enabled, the final discrete subset
    receives one additional fixed-depth relaxation pass.  Only retained knots
    participate as attention keys/values, and every retained token receives
    its adjacent-survivor spacing and compact rank/count.  This couples the
    final deletion pattern to the deployed locations without an iterative
    deployment loop.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        *,
        residual_feature_dim: int = 1,
        attention_heads: int = 4,
        min_gap: float = 1e-3,
        max_position_fraction: float = 0.45,
        initial_keep_probability: float = 0.9,
        one_shot_adaptive: bool = False,
        one_shot_fixed_proposal_geometry: bool = False,
        one_shot_selection_policy: str = "threshold",
        one_shot_safety_sigma: float = 0.0,
        one_shot_selector_layers: int = 1,
        one_shot_coverage_bins: int = 0,
        one_shot_joint_position_refinement: bool = False,
        one_shot_survivor_relocation: bool = False,
        one_shot_max_position_shift: float = 0.05,
    ) -> None:
        super().__init__()
        if attention_heads <= 0 or hidden_dim % attention_heads != 0:
            raise ValueError("attention_heads must divide hidden_dim")
        if residual_feature_dim <= 0:
            raise ValueError("residual_feature_dim must be positive")
        if min_gap < 0.0:
            raise ValueError("min_gap must be non-negative")
        if not 0.0 < max_position_fraction < 0.5:
            raise ValueError(
                "max_position_fraction must lie strictly between 0 and 0.5"
            )
        if not 0.0 < initial_keep_probability < 1.0:
            raise ValueError("initial_keep_probability must lie in (0,1)")
        if one_shot_selection_policy not in {"threshold", "mass_topk"}:
            raise ValueError(
                "one_shot_selection_policy must be 'threshold' or 'mass_topk'"
            )
        if not math.isfinite(one_shot_safety_sigma) or one_shot_safety_sigma < 0.0:
            raise ValueError("one_shot_safety_sigma must be finite and non-negative")
        if one_shot_selector_layers < 1:
            raise ValueError("one_shot_selector_layers must be positive")
        if one_shot_coverage_bins < 0:
            raise ValueError("one_shot_coverage_bins must be non-negative")
        if (
            not math.isfinite(one_shot_max_position_shift)
            or one_shot_max_position_shift <= 0.0
            or one_shot_max_position_shift >= 0.5
        ):
            raise ValueError("one_shot_max_position_shift must lie in (0,0.5)")

        self.hidden_dim = int(hidden_dim)
        self.residual_feature_dim = int(residual_feature_dim)
        self.min_gap = float(min_gap)
        self.max_position_fraction = float(max_position_fraction)
        self.one_shot_adaptive = bool(one_shot_adaptive)
        self.one_shot_fixed_proposal_geometry = bool(one_shot_fixed_proposal_geometry)
        self.one_shot_selection_policy = str(one_shot_selection_policy)
        self.one_shot_safety_sigma = float(one_shot_safety_sigma)
        self.one_shot_selector_layers = int(one_shot_selector_layers)
        self.one_shot_coverage_bins = int(one_shot_coverage_bins)
        self.one_shot_joint_position_refinement = bool(
            one_shot_joint_position_refinement
        )
        self.one_shot_survivor_relocation = bool(one_shot_survivor_relocation)
        self.one_shot_max_position_shift = float(one_shot_max_position_shift)
        if self.one_shot_fixed_proposal_geometry and not self.one_shot_adaptive:
            raise ValueError(
                "fixed one-shot proposal geometry requires one_shot_adaptive"
            )
        if self.one_shot_joint_position_refinement and not (
            self.one_shot_fixed_proposal_geometry and self.one_shot_adaptive
        ):
            raise ValueError(
                "joint one-shot position refinement requires fixed adaptive proposals"
            )
        if self.one_shot_survivor_relocation and not (
            self.one_shot_joint_position_refinement
            and self.one_shot_fixed_proposal_geometry
            and self.one_shot_adaptive
        ):
            raise ValueError(
                "survivor relocation requires joint fixed-proposal one-shot "
                "position refinement"
            )

        # position + coefficient energy + deletion delta + left/right spacing
        # + the configurable local residual descriptor.
        analytic_dim = 5 + self.residual_feature_dim
        self.analytic_projection = nn.Sequential(
            nn.Linear(analytic_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.input_norm = nn.LayerNorm(self.hidden_dim)
        self.self_attention = nn.MultiheadAttention(
            self.hidden_dim,
            attention_heads,
            batch_first=True,
        )
        self.self_norm = nn.LayerNorm(self.hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
            nn.GELU(),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)

        self.keep_head = nn.Linear(self.hidden_dim, 1)
        nn.init.normal_(self.keep_head.weight, std=0.01)
        nn.init.constant_(
            self.keep_head.bias,
            math.log(initial_keep_probability / (1.0 - initial_keep_probability)),
        )
        if self.one_shot_adaptive:
            self.adaptive_threshold_head = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, 1),
            )
            nn.init.normal_(self.adaptive_threshold_head[-1].weight, std=0.01)
            nn.init.zeros_(self.adaptive_threshold_head[-1].bias)
            self.keep_context_projection = nn.Sequential(
                nn.Linear(3 * self.hidden_dim + 1, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            self.keep_context_norm = nn.LayerNorm(self.hidden_dim)
            # A separately named position -> selection path lets staged
            # training unfreeze the bidirectional feedback without also
            # changing the candidate encoder or the position decoder.  It is
            # intentionally created only for v8 one-shot heads: legacy v7
            # state_dict layouts remain byte-for-byte unchanged.
            self.position_to_keep_feedback = nn.Sequential(
                nn.Linear(3 * self.hidden_dim + 1, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            nn.init.normal_(self.position_to_keep_feedback[-1].weight, std=0.01)
            nn.init.zeros_(self.position_to_keep_feedback[-1].bias)
            if self.one_shot_fixed_proposal_geometry:
                # v9 selection capacity is deliberately isolated from the
                # proposal/location path.  These layers may be distilled from
                # the offline Hard-RMS teacher without changing the candidate
                # positions to which that teacher cache is bound.
                self.one_shot_selector_attention = nn.MultiheadAttention(
                    self.hidden_dim,
                    attention_heads,
                    batch_first=True,
                )
                self.one_shot_selector_attention_norm = nn.LayerNorm(self.hidden_dim)
                self.one_shot_selector_feed_forward = nn.Sequential(
                    nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
                    nn.GELU(),
                    nn.Linear(2 * self.hidden_dim, self.hidden_dim),
                )
                self.one_shot_selector_output_norm = nn.LayerNorm(self.hidden_dim)
                extra_layers = self.one_shot_selector_layers - 1
                self.one_shot_selector_extra_attention = nn.ModuleList(
                    nn.MultiheadAttention(
                        self.hidden_dim,
                        attention_heads,
                        batch_first=True,
                    )
                    for _ in range(extra_layers)
                )
                self.one_shot_selector_extra_attention_norm = nn.ModuleList(
                    nn.LayerNorm(self.hidden_dim) for _ in range(extra_layers)
                )
                self.one_shot_selector_extra_feed_forward = nn.ModuleList(
                    nn.Sequential(
                        nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
                        nn.GELU(),
                        nn.Linear(2 * self.hidden_dim, self.hidden_dim),
                    )
                    for _ in range(extra_layers)
                )
                self.one_shot_selector_extra_output_norm = nn.ModuleList(
                    nn.LayerNorm(self.hidden_dim) for _ in range(extra_layers)
                )
                if self.one_shot_joint_position_refinement:
                    # v11 uses a fixed two-pass interaction inside one network
                    # forward: preliminary Keep -> position -> final Keep ->
                    # final position.  Zero-initialised terminal layers make a
                    # v10 proposal an exact neutral initialiser.
                    self.joint_preliminary_position_projection = nn.Sequential(
                        nn.Linear(3 * self.hidden_dim + 1, self.hidden_dim),
                        nn.GELU(),
                        nn.Linear(self.hidden_dim, self.hidden_dim),
                    )
                    self.joint_preliminary_position_norm = nn.LayerNorm(self.hidden_dim)
                    self.joint_preliminary_position_head = nn.Linear(self.hidden_dim, 1)
                    self.joint_position_to_keep_feedback = nn.Sequential(
                        nn.Linear(3 * self.hidden_dim + 2, self.hidden_dim),
                        nn.GELU(),
                        nn.Linear(self.hidden_dim, self.hidden_dim),
                    )
                    self.joint_position_to_keep_norm = nn.LayerNorm(self.hidden_dim)
                    self.joint_final_position_projection = nn.Sequential(
                        nn.Linear(3 * self.hidden_dim + 1, self.hidden_dim),
                        nn.GELU(),
                        nn.Linear(self.hidden_dim, self.hidden_dim),
                    )
                    self.joint_final_position_norm = nn.LayerNorm(self.hidden_dim)
                    self.joint_final_position_head = nn.Linear(self.hidden_dim, 1)
                    for module in (
                        self.joint_preliminary_position_head,
                        self.joint_position_to_keep_feedback[-1],
                        self.joint_final_position_head,
                    ):
                        nn.init.zeros_(module.weight)
                        nn.init.zeros_(module.bias)
                    if self.one_shot_survivor_relocation:
                        # v12 adds a final survivor-only relaxation after the
                        # v11 Keep -> position -> Keep -> position passes.  Its
                        # keys/values contain only the final hard survivors,
                        # while explicit relative geometry tells every
                        # survivor where it lies inside the compact retained
                        # sequence.  The zero-initialised output keeps a v11
                        # checkpoint exactly neutral until this block is
                        # trained against relocated teacher knots.
                        survivor_feature_dim = 8
                        self.survivor_relative_projection = nn.Sequential(
                            nn.Linear(survivor_feature_dim, self.hidden_dim),
                            nn.GELU(),
                            nn.Linear(self.hidden_dim, self.hidden_dim),
                        )
                        self.survivor_relocation_input_norm = nn.LayerNorm(
                            self.hidden_dim
                        )
                        self.survivor_relocation_attention = nn.MultiheadAttention(
                            self.hidden_dim,
                            attention_heads,
                            batch_first=True,
                        )
                        self.survivor_relocation_attention_norm = nn.LayerNorm(
                            self.hidden_dim
                        )
                        self.survivor_relocation_feed_forward = nn.Sequential(
                            nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
                            nn.GELU(),
                            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
                        )
                        self.survivor_relocation_output_norm = nn.LayerNorm(
                            self.hidden_dim
                        )
                        self.survivor_relocation_head = nn.Linear(self.hidden_dim, 1)
                        nn.init.zeros_(self.survivor_relocation_head.weight)
                        nn.init.zeros_(self.survivor_relocation_head.bias)
        self.deletion_cost_head = nn.Linear(self.hidden_dim, 1)
        nn.init.normal_(self.deletion_cost_head.weight, std=0.01)
        nn.init.constant_(self.deletion_cost_head.bias, -4.0)
        self.remove_action_head = nn.Linear(self.hidden_dim, 1)
        nn.init.normal_(self.remove_action_head.weight, std=0.01)
        nn.init.zeros_(self.remove_action_head.bias)
        self.stop_action_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.stop_action_head[-1].weight)
        nn.init.constant_(self.stop_action_head[-1].bias, -2.0)
        self.position_residual_head = nn.Linear(self.hidden_dim, 1)
        # Begin from the high-recall candidates; position refinement is learned
        # only when its supervision supplies evidence for a move.
        nn.init.zeros_(self.position_residual_head.weight)
        nn.init.zeros_(self.position_residual_head.bias)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        # Early one-shot v8 checkpoints predate the explicit position->keep
        # feedback block.  Supplying this block's constructor initialization
        # keeps strict loading usable while still loading every historical
        # tensor exactly.  Legacy v7 heads never construct the block, so their
        # 30-key layout and strict-loading behavior are untouched.
        if self.one_shot_adaptive:
            feedback_prefix = prefix + "position_to_keep_feedback."
            for name, value in self.position_to_keep_feedback.state_dict().items():
                key = feedback_prefix + name
                if key not in state_dict:
                    state_dict[key] = value.detach().clone()
        # A v9 proposal/final checkpoint can initialize the deeper structured
        # selector used by the feasibility-enhanced objective.  Only the new
        # selector blocks are filled from their constructor initialization;
        # every proposal/location tensor still loads strictly from the source.
        if self.one_shot_fixed_proposal_geometry:
            for module_name in (
                "one_shot_selector_extra_attention",
                "one_shot_selector_extra_attention_norm",
                "one_shot_selector_extra_feed_forward",
                "one_shot_selector_extra_output_norm",
            ):
                module = getattr(self, module_name, None)
                if module is None:
                    continue
                module_prefix = prefix + module_name + "."
                for name, value in module.state_dict().items():
                    key = module_prefix + name
                    if key not in state_dict:
                        state_dict[key] = value.detach().clone()
            if self.one_shot_joint_position_refinement:
                for module_name in (
                    "joint_preliminary_position_projection",
                    "joint_preliminary_position_norm",
                    "joint_preliminary_position_head",
                    "joint_position_to_keep_feedback",
                    "joint_position_to_keep_norm",
                    "joint_final_position_projection",
                    "joint_final_position_norm",
                    "joint_final_position_head",
                ):
                    module = getattr(self, module_name)
                    module_prefix = prefix + module_name + "."
                    for name, value in module.state_dict().items():
                        key = module_prefix + name
                        if key not in state_dict:
                            state_dict[key] = value.detach().clone()
                if self.one_shot_survivor_relocation:
                    # Strictly load a v11 model into v12.  In particular, the
                    # zero relocation head makes the newly created branch an
                    # exact identity before v12 training starts.
                    for module_name in (
                        "survivor_relative_projection",
                        "survivor_relocation_input_norm",
                        "survivor_relocation_attention",
                        "survivor_relocation_attention_norm",
                        "survivor_relocation_feed_forward",
                        "survivor_relocation_output_norm",
                        "survivor_relocation_head",
                    ):
                        module = getattr(self, module_name)
                        module_prefix = prefix + module_name + "."
                        for name, value in module.state_dict().items():
                            key = module_prefix + name
                            if key not in state_dict:
                                state_dict[key] = value.detach().clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @staticmethod
    def _stable_signed_log1p(values: torch.Tensor) -> torch.Tensor:
        values = torch.nan_to_num(values, nan=0.0, posinf=1e6, neginf=-1e6)
        values = values.clamp(min=-1e6, max=1e6)
        return values.sign() * torch.log1p(values.abs())

    @staticmethod
    def _two_sided_spacing(candidate_positions: torch.Tensor) -> torch.Tensor:
        batch = candidate_positions.shape[0]
        boundaries = torch.cat(
            [
                candidate_positions.new_zeros(batch, 1),
                candidate_positions,
                candidate_positions.new_ones(batch, 1),
            ],
            dim=-1,
        )
        intervals = boundaries[:, 1:] - boundaries[:, :-1]
        return torch.stack([intervals[:, :-1], intervals[:, 1:]], dim=-1)

    @staticmethod
    def _straight_through_keep_gate(
        keep_probabilities: torch.Tensor,
        hard_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        return (
            hard_mask.to(keep_probabilities.dtype)
            + keep_probabilities
            - keep_probabilities.detach()
        ) * candidate_mask.to(keep_probabilities.dtype)

    @staticmethod
    def _hard_st_context(
        tokens: torch.Tensor,
        hard_mask: torch.Tensor,
        straight_through_gate: torch.Tensor,
    ) -> torch.Tensor:
        hard_count = hard_mask.to(tokens.dtype).sum(dim=1, keepdim=True)
        st_count = straight_through_gate.sum(dim=1, keepdim=True)
        denominator = hard_count.clamp_min(1.0) + st_count - st_count.detach()
        return (tokens * straight_through_gate.unsqueeze(-1)).sum(dim=1) / denominator

    def _bounded_all_candidate_residual(
        self,
        candidate_positions: torch.Tensor,
        raw_signal: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Move every slot without changing the global candidate ordering."""

        spacing = self._two_sided_spacing(candidate_positions)
        left_slack = (spacing[..., 0] - self.min_gap).clamp_min(0.0)
        right_slack = (spacing[..., 1] - self.min_gap).clamp_min(0.0)
        residual = self.max_position_fraction * torch.where(
            raw_signal >= 0.0,
            raw_signal * right_slack,
            raw_signal * left_slack,
        )
        residual = residual.clamp(
            min=-self.one_shot_max_position_shift,
            max=self.one_shot_max_position_shift,
        )
        return residual.masked_fill(~candidate_mask, 0.0)

    def _bounded_selected_residual(
        self,
        candidate_positions: torch.Tensor,
        raw_signal: torch.Tensor,
        selected_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
        *,
        reference_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Bound motion by the adjacent *selected* knots, not dense proposals.

        The selected subsequence can therefore move substantially farther than
        half a dense proposal interval, while the ``< 0.5`` motion fraction
        preserves its strict order and minimum gap in one fixed tensor pass.
        Unselected slots remain at their proposal locations.
        """

        active = selected_mask & candidate_mask
        batch, candidate_count = active.shape
        indices = torch.arange(candidate_count, device=active.device).unsqueeze(0)
        indices = indices.expand(batch, -1)

        previous_inclusive = torch.where(active, indices, -1).cummax(dim=-1).values
        previous_exclusive = torch.cat(
            [
                torch.full_like(previous_inclusive[:, :1], -1),
                previous_inclusive[:, :-1],
            ],
            dim=-1,
        )
        next_inclusive = torch.flip(
            torch.flip(torch.where(active, indices, candidate_count), dims=(-1,))
            .cummin(dim=-1)
            .values,
            dims=(-1,),
        )
        next_exclusive = torch.cat(
            [
                next_inclusive[:, 1:],
                torch.full_like(next_inclusive[:, :1], candidate_count),
            ],
            dim=-1,
        )

        previous_position = candidate_positions.gather(
            1, previous_exclusive.clamp(min=0)
        )
        previous_position = torch.where(
            previous_exclusive >= 0,
            previous_position,
            torch.zeros_like(previous_position),
        )
        next_position = candidate_positions.gather(
            1, next_exclusive.clamp(max=candidate_count - 1)
        )
        next_position = torch.where(
            next_exclusive < candidate_count,
            next_position,
            torch.ones_like(next_position),
        )
        left_slack = (candidate_positions - previous_position - self.min_gap).clamp_min(
            0.0
        )
        right_slack = (next_position - candidate_positions - self.min_gap).clamp_min(
            0.0
        )
        residual = self.max_position_fraction * torch.where(
            raw_signal >= 0.0,
            raw_signal * right_slack,
            raw_signal * left_slack,
        )
        if reference_positions is None:
            residual = residual.clamp(
                min=-self.one_shot_max_position_shift,
                max=self.one_shot_max_position_shift,
            )
        else:
            if reference_positions.shape != candidate_positions.shape:
                raise ValueError("reference_positions must share candidate shape")
            consumed_budget = candidate_positions - reference_positions
            minimum_residual = -self.one_shot_max_position_shift - consumed_budget
            maximum_residual = self.one_shot_max_position_shift - consumed_budget
            residual = torch.maximum(
                minimum_residual,
                torch.minimum(residual, maximum_residual),
            )
        return residual.masked_fill(~active, 0.0)

    def _project_ordered_positions(
        self,
        target_positions: torch.Tensor,
        active_mask: torch.Tensor,
        *,
        reference_positions: torch.Tensor,
        max_shift: float | None = None,
    ) -> torch.Tensor:
        """Project active slots to a feasible ordered parameter subsequence.

        Selected-only relocation deliberately lets a survivor move across
        dormant proposal slots.  Such a tensor is ordered on the survivor
        subsequence, but not necessarily across all ``Kc`` storage slots.  A
        later all-candidate pass must not silently assume otherwise.  This
        fixed two-sweep projection repairs exactly that boundary condition.

        The projection preserves slot identity, enforces the endpoint and
        adjacent-active ``min_gap`` constraints, and limits every active slot
        to ``max_shift`` from an ordered reference.  A backward sweep first
        propagates future upper bounds; a forward sweep then clamps each slot
        without an optimizer or a data-dependent iteration.  Gradients flow
        through unclamped targets and through the active clamp boundary.

        Inactive slots are returned unchanged.  Consequently callers may use
        all valid candidates as ``active_mask`` before re-selection, or only
        the final KeepMask when validating the deployed survivor sequence.
        """

        if target_positions.ndim != 2:
            raise ValueError("target_positions must have shape [B,K]")
        if reference_positions.shape != target_positions.shape:
            raise ValueError("reference_positions must share target shape")
        if active_mask.shape != target_positions.shape or active_mask.dtype != torch.bool:
            raise ValueError("active_mask must be boolean with shape [B,K]")
        if not target_positions.is_floating_point():
            raise ValueError("target_positions must be floating-point")
        if reference_positions.dtype != target_positions.dtype:
            raise ValueError("reference_positions must share target dtype")
        if reference_positions.device != target_positions.device:
            raise ValueError("reference_positions must share target device")

        shift = self.one_shot_max_position_shift if max_shift is None else max_shift
        if not math.isfinite(shift) or shift < 0.0 or shift >= 0.5:
            raise ValueError("max_shift must be finite and lie in [0, 0.5)")

        dtype = target_positions.dtype
        active_float = active_mask.to(dtype)
        active_count = active_float.sum(dim=-1, keepdim=True)
        active_rank = active_float.cumsum(dim=-1) - active_float

        # Internal knots also keep ``min_gap`` from both domain endpoints.
        domain_lower = (active_rank + 1.0) * self.min_gap
        remaining = active_count - active_rank
        domain_upper = 1.0 - remaining * self.min_gap
        lower = torch.maximum(
            domain_lower,
            reference_positions - float(shift),
        )
        upper = torch.minimum(
            domain_upper,
            reference_positions + float(shift),
        )

        # Propagate the tightest future upper bound to every earlier active
        # slot.  This prevents a greedy left-to-right clamp from consuming the
        # space required by a later survivor.
        batch, candidate_count = target_positions.shape
        running_upper = target_positions.new_ones(batch)
        has_next = torch.zeros(batch, dtype=torch.bool, device=target_positions.device)
        feasible_upper_columns: list[torch.Tensor] = []
        for index in range(candidate_count - 1, -1, -1):
            is_active = active_mask[:, index]
            candidate_upper = torch.where(
                has_next,
                torch.minimum(upper[:, index], running_upper - self.min_gap),
                upper[:, index],
            )
            running_upper = torch.where(is_active, candidate_upper, running_upper)
            has_next = has_next | is_active
            feasible_upper_columns.append(candidate_upper)
        feasible_upper = torch.stack(
            list(reversed(feasible_upper_columns)),
            dim=-1,
        )

        fallback = torch.where(
            torch.isfinite(reference_positions),
            reference_positions,
            0.5 * (lower + feasible_upper),
        )
        finite_target = torch.where(
            torch.isfinite(target_positions),
            target_positions,
            fallback,
        )
        previous = target_positions.new_zeros(batch)
        has_previous = torch.zeros(
            batch,
            dtype=torch.bool,
            device=target_positions.device,
        )
        projected_columns: list[torch.Tensor] = []
        for index in range(candidate_count):
            is_active = active_mask[:, index]
            feasible_lower = torch.where(
                has_previous,
                torch.maximum(lower[:, index], previous + self.min_gap),
                lower[:, index],
            )
            # A legal ordered reference makes this interval nonempty.  The
            # minimum below is a round-off guard for float32 at the boundary.
            feasible_lower = torch.minimum(
                feasible_lower,
                feasible_upper[:, index],
            )
            projected = torch.maximum(
                feasible_lower,
                torch.minimum(finite_target[:, index], feasible_upper[:, index]),
            )
            output_column = torch.where(
                is_active,
                projected,
                target_positions[:, index],
            )
            projected_columns.append(output_column)
            previous = torch.where(is_active, projected, previous)
            has_previous = has_previous | is_active

        return torch.stack(projected_columns, dim=-1)

    @staticmethod
    def _fill_inactive_ordered_slots(
        selected_positions: torch.Tensor,
        selected_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Canonicalize dormant storage slots without moving survivors.

        A survivor is allowed to cross a deleted proposal slot, because the
        latter contributes no spline basis column.  For downstream code that
        consumes the complete fixed-width tensor, place each dormant slot by
        linear interpolation between its adjacent survivors (or endpoints).
        The selected geometry and KeepMask correspondence stay unchanged,
        while every valid storage row becomes strictly left-to-right ordered.
        """

        if selected_positions.ndim != 2:
            raise ValueError("selected_positions must have shape [B,K]")
        if selected_mask.shape != selected_positions.shape or selected_mask.dtype != torch.bool:
            raise ValueError("selected_mask must be boolean with shape [B,K]")
        if candidate_mask.shape != selected_positions.shape or candidate_mask.dtype != torch.bool:
            raise ValueError("candidate_mask must be boolean with shape [B,K]")

        active = selected_mask & candidate_mask
        batch, candidate_count = active.shape
        indices = torch.arange(candidate_count, device=active.device).unsqueeze(0)
        indices = indices.expand(batch, -1)
        previous_index = torch.where(active, indices, -1).cummax(dim=-1).values
        next_index = torch.flip(
            torch.flip(
                torch.where(active, indices, candidate_count),
                dims=(-1,),
            ).cummin(dim=-1).values,
            dims=(-1,),
        )

        previous_position = selected_positions.gather(
            1,
            previous_index.clamp(min=0),
        )
        previous_position = torch.where(
            previous_index >= 0,
            previous_position,
            torch.zeros_like(previous_position),
        )
        next_position = selected_positions.gather(
            1,
            next_index.clamp(max=candidate_count - 1),
        )
        next_position = torch.where(
            next_index < candidate_count,
            next_position,
            torch.ones_like(next_position),
        )
        denominator = (next_index - previous_index).clamp_min(1).to(
            selected_positions.dtype
        )
        fraction = (indices - previous_index).to(selected_positions.dtype) / denominator
        filled = previous_position + fraction * (next_position - previous_position)
        filled = torch.where(active, selected_positions, filled)

        # With no survivor there is no authoritative geometry to interpolate
        # around; retain the already ordered all-candidate input unchanged.
        has_survivor = active.any(dim=-1, keepdim=True)
        filled = torch.where(has_survivor, filled, selected_positions)
        return torch.where(candidate_mask, filled, selected_positions)

    def _survivor_relative_geometry(
        self,
        candidate_positions: torch.Tensor,
        selected_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
        straight_through_gate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Describe each knot relative to the final retained subsequence.

        Dense proposal spacing is the wrong geometry after pruning: two
        adjacent survivors may have many deleted slots between them.  This
        descriptor therefore uses the previous/next *selected* knot (or a
        domain endpoint), together with retained rank and retained count.
        Rank/count have hard values in the forward pass and sigmoid gradients
        in the backward pass through ``straight_through_gate``.
        """

        if selected_mask.shape != candidate_positions.shape:
            raise ValueError("selected_mask must share candidate shape")
        if candidate_mask.shape != candidate_positions.shape:
            raise ValueError("candidate_mask must share candidate shape")
        if straight_through_gate.shape != candidate_positions.shape:
            raise ValueError("straight_through_gate must share candidate shape")

        active = selected_mask & candidate_mask
        batch, candidate_count = active.shape
        indices = torch.arange(candidate_count, device=active.device).unsqueeze(0)
        indices = indices.expand(batch, -1)

        previous_inclusive = torch.where(active, indices, -1).cummax(dim=-1).values
        previous_exclusive = torch.cat(
            [
                torch.full_like(previous_inclusive[:, :1], -1),
                previous_inclusive[:, :-1],
            ],
            dim=-1,
        )
        next_inclusive = torch.flip(
            torch.flip(
                torch.where(active, indices, candidate_count),
                dims=(-1,),
            )
            .cummin(dim=-1)
            .values,
            dims=(-1,),
        )
        next_exclusive = torch.cat(
            [
                next_inclusive[:, 1:],
                torch.full_like(next_inclusive[:, :1], candidate_count),
            ],
            dim=-1,
        )

        previous_position = candidate_positions.gather(
            1, previous_exclusive.clamp(min=0)
        )
        previous_position = torch.where(
            previous_exclusive >= 0,
            previous_position,
            torch.zeros_like(previous_position),
        )
        next_position = candidate_positions.gather(
            1, next_exclusive.clamp(max=candidate_count - 1)
        )
        next_position = torch.where(
            next_exclusive < candidate_count,
            next_position,
            torch.ones_like(next_position),
        )

        left_gap = (candidate_positions - previous_position).clamp_min(0.0)
        right_gap = (next_position - candidate_positions).clamp_min(0.0)
        survivor_cell_span = (next_position - previous_position).clamp_min(1e-6)
        local_coordinate = (left_gap / survivor_cell_span).clamp(0.0, 1.0)

        hard_count = active.to(candidate_positions.dtype).sum(dim=-1, keepdim=True)
        st_count = straight_through_gate.sum(dim=-1, keepdim=True)
        differentiable_count = hard_count + st_count - st_count.detach()
        hard_rank = active.to(candidate_positions.dtype).cumsum(dim=-1)
        st_rank = straight_through_gate.cumsum(dim=-1)
        differentiable_rank = hard_rank + st_rank - st_rank.detach()
        relative_rank = differentiable_rank / (differentiable_count + 1.0)
        valid_count = candidate_mask.to(candidate_positions.dtype).sum(
            dim=-1, keepdim=True
        )
        count_fraction = differentiable_count / valid_count.clamp_min(1.0)
        count_fraction = count_fraction.expand_as(candidate_positions)
        coverage_offset = relative_rank - candidate_positions

        relative_features = torch.stack(
            [
                left_gap,
                right_gap,
                survivor_cell_span,
                local_coordinate,
                relative_rank,
                count_fraction,
                coverage_offset,
                straight_through_gate,
            ],
            dim=-1,
        )
        relative_features = relative_features * active.to(
            relative_features.dtype
        ).unsqueeze(-1)
        return relative_features, previous_position, next_position

    def _relocate_survivors(
        self,
        decision_tokens: torch.Tensor,
        candidate_positions: torch.Tensor,
        selected_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
        straight_through_gate: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run one selected-only attention pass for final knot relocation."""

        (
            relative_features,
            previous_position,
            next_position,
        ) = self._survivor_relative_geometry(
            candidate_positions,
            selected_mask,
            candidate_mask,
            straight_through_gate,
        )
        relocation_position_encoding = KnotHead._sinusoidal_position_encoding(
            candidate_positions,
            self.hidden_dim,
        )
        relocation_inputs = self.survivor_relocation_input_norm(
            decision_tokens
            + relocation_position_encoding
            + self.survivor_relative_projection(relative_features)
        )

        active = selected_mask & candidate_mask
        # MultiheadAttention cannot consume an all-masked key row.  A zero-
        # valued fallback key makes K=0 finite; the final active mask still
        # forces every relocation residual to exactly zero for that sample.
        empty = ~active.any(dim=-1)
        fallback_index = candidate_mask.to(torch.long).argmax(dim=-1, keepdim=True)
        fallback_mask = torch.zeros_like(active).scatter(
            1,
            fallback_index,
            True,
        )
        safe_key_mask = active | (fallback_mask & empty.unsqueeze(-1))
        selected_values = relocation_inputs * straight_through_gate.unsqueeze(-1)
        interacted, attention_weights = self.survivor_relocation_attention(
            relocation_inputs,
            selected_values,
            selected_values,
            key_padding_mask=~safe_key_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        relocation_tokens = self.survivor_relocation_attention_norm(
            relocation_inputs + interacted
        )
        relocation_tokens = self.survivor_relocation_output_norm(
            relocation_tokens + self.survivor_relocation_feed_forward(relocation_tokens)
        )
        raw_signal = torch.tanh(
            self.survivor_relocation_head(relocation_tokens).squeeze(-1)
        )
        diagnostic_attention = (
            attention_weights
            * active.to(attention_weights.dtype).unsqueeze(-1)
            * active.to(attention_weights.dtype).unsqueeze(1)
        )
        return {
            "relative_features": relative_features,
            "previous_position": previous_position,
            "next_position": next_position,
            "input_tokens": relocation_inputs,
            "tokens": relocation_tokens,
            "attention_weights": diagnostic_attention,
            "raw_signal": raw_signal,
        }

    def _select_hard_keep_mask(
        self,
        keep_probabilities: torch.Tensor,
        candidate_mask: torch.Tensor,
        candidate_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create the discrete one-shot subset and expose its count statistics.

        ``mass_topk`` interprets ``sum(p)`` as the learned cardinality and adds
        an optional Bernoulli uncertainty reserve before choosing the globally
        highest-scoring candidates.  It is still a single mask construction:
        no spline solve, threshold sweep, or iterative deletion is performed.
        """

        probability = keep_probabilities * candidate_mask.to(keep_probabilities.dtype)
        probability_mass = probability.sum(dim=-1)
        uncertainty = (
            (probability * (1.0 - probability) * candidate_mask.to(probability.dtype))
            .sum(dim=-1)
            .clamp_min(0.0)
            .sqrt()
        )
        if self.one_shot_selection_policy == "threshold":
            hard_mask = (keep_probabilities >= 0.5) & candidate_mask
            selected_count = hard_mask.sum(dim=-1).to(torch.long)
            return hard_mask, selected_count, uncertainty

        requested_count_score = (
            probability_mass + self.one_shot_safety_sigma * uncertainty
        )
        requested_count = torch.where(
            requested_count_score < 0.5,
            torch.zeros_like(requested_count_score, dtype=torch.long),
            torch.ceil(requested_count_score).to(torch.long),
        )
        valid_count = candidate_mask.sum(dim=-1).to(torch.long)
        requested_count = torch.minimum(requested_count.clamp_min(0), valid_count)
        selection_score = keep_probabilities.masked_fill(~candidate_mask, float("-inf"))
        if self.one_shot_coverage_bins > 0:
            # Reserve one high-probability anchor in each non-empty parameter
            # interval whenever the selected budget can afford all anchors.
            # Giving anchors a constant score offset makes the final operation
            # one global Top-K rather than a data-dependent refit/search loop.
            anchors = torch.zeros_like(candidate_mask)
            for bin_index in range(self.one_shot_coverage_bins):
                lower = bin_index / self.one_shot_coverage_bins
                upper = (bin_index + 1) / self.one_shot_coverage_bins
                in_bin = (
                    (candidate_positions >= lower)
                    & (candidate_positions < upper)
                    & candidate_mask
                )
                has_candidate = in_bin.any(dim=-1)
                best_index = keep_probabilities.masked_fill(
                    ~in_bin, float("-inf")
                ).argmax(dim=-1)
                anchors.scatter_(
                    1, best_index.unsqueeze(-1), has_candidate.unsqueeze(-1)
                )
            anchor_count = anchors.sum(dim=-1)
            affordable = anchor_count <= requested_count
            anchors = anchors & affordable.unsqueeze(-1)
            selection_score = selection_score + anchors.to(selection_score.dtype) * 2.0
        order = torch.argsort(
            selection_score,
            dim=-1,
            descending=True,
            stable=True,
        )
        rank = torch.empty_like(order)
        rank.scatter_(
            1,
            order,
            torch.arange(order.shape[1], device=order.device)
            .unsqueeze(0)
            .expand_as(order),
        )
        hard_mask = (rank < requested_count.unsqueeze(-1)) & candidate_mask
        return hard_mask, requested_count, uncertainty

    def _validate_inputs(
        self,
        candidate_tokens: torch.Tensor,
        candidate_positions: torch.Tensor,
        coefficient_energy: torch.Tensor,
        deletion_delta: torch.Tensor,
        spacing: torch.Tensor,
        residual_features: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> None:
        if candidate_tokens.ndim != 3 or candidate_positions.ndim != 2:
            raise ValueError(
                "candidate_tokens and candidate_positions must be [B,K,H] and [B,K]"
            )
        if candidate_tokens.shape[:2] != candidate_positions.shape:
            raise ValueError("candidate tokens and positions must share [B,K]")
        if candidate_tokens.shape[-1] != self.hidden_dim:
            raise ValueError("candidate token dimension does not match hidden_dim")
        expected_scalar_shape = candidate_positions.shape
        if coefficient_energy.shape != expected_scalar_shape:
            raise ValueError("coefficient_energy must have shape [B,K]")
        if deletion_delta.shape != expected_scalar_shape:
            raise ValueError("deletion_delta must have shape [B,K]")
        if spacing.shape != (*expected_scalar_shape, 2):
            raise ValueError("spacing must have shape [B,K,2]")
        if residual_features.shape != (
            *expected_scalar_shape,
            self.residual_feature_dim,
        ):
            raise ValueError(
                "residual_features must have shape [B,K,residual_feature_dim]"
            )
        if (
            candidate_mask.shape != expected_scalar_shape
            or candidate_mask.dtype != torch.bool
        ):
            raise ValueError("candidate_mask must be boolean with shape [B,K]")
        if candidate_positions.shape[1] == 0:
            raise ValueError("at least one candidate is required")
        if torch.any(~candidate_mask.any(dim=-1)):
            raise ValueError("every sample must contain at least one active candidate")

        # The bounded refinement guarantee assumes a valid ordered input.  The
        # small tolerance avoids false failures from floating-point round-off.
        full_spacing = self._two_sided_spacing(candidate_positions)
        if torch.any(full_spacing < self.min_gap - 1e-6):
            raise ValueError("candidate_positions must be ordered and respect min_gap")

    def forward(
        self,
        candidate_tokens: torch.Tensor,
        candidate_positions: torch.Tensor,
        *,
        coefficient_energy: torch.Tensor,
        deletion_delta: torch.Tensor,
        residual_features: torch.Tensor,
        spacing: torch.Tensor | None = None,
        candidate_mask: torch.Tensor | None = None,
        global_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Evaluate and refine a fixed candidate set.

        ``spacing`` contains left/right gaps and defaults to the exact gaps
        derived from ``candidate_positions``.  A scalar residual descriptor can
        be passed as ``[B,K]`` when ``residual_feature_dim == 1``.
        """

        if candidate_positions.ndim != 2:
            raise ValueError("candidate_positions must have shape [B,K]")
        if spacing is None:
            spacing = self._two_sided_spacing(candidate_positions)
        if residual_features.ndim == 2 and self.residual_feature_dim == 1:
            residual_features = residual_features.unsqueeze(-1)
        if candidate_mask is None:
            candidate_mask = torch.ones_like(candidate_positions, dtype=torch.bool)
        if global_features is not None and global_features.shape != (
            candidate_positions.shape[0],
            self.hidden_dim,
        ):
            raise ValueError("global_features must have shape [B,H]")

        self._validate_inputs(
            candidate_tokens,
            candidate_positions,
            coefficient_energy,
            deletion_delta,
            spacing,
            residual_features,
            candidate_mask,
        )

        coefficient_feature = torch.log1p(
            torch.nan_to_num(
                coefficient_energy,
                nan=0.0,
                posinf=1e6,
                neginf=0.0,
            ).clamp(min=0.0, max=1e6)
        )
        deletion_feature = self._stable_signed_log1p(deletion_delta)
        spacing_feature = torch.nan_to_num(
            spacing,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(min=0.0, max=1.0)
        residual_feature = self._stable_signed_log1p(residual_features)
        analytic_features = torch.cat(
            [
                candidate_positions.unsqueeze(-1),
                coefficient_feature.unsqueeze(-1),
                deletion_feature.unsqueeze(-1),
                spacing_feature,
                residual_feature,
            ],
            dim=-1,
        )

        position_encoding = KnotHead._sinusoidal_position_encoding(
            candidate_positions,
            self.hidden_dim,
        )
        tokens = self.input_norm(
            candidate_tokens
            + position_encoding
            + self.analytic_projection(analytic_features)
        )
        interacted, attention_weights = self.self_attention(
            tokens,
            tokens,
            tokens,
            key_padding_mask=~candidate_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        tokens = self.self_norm(tokens + interacted)
        tokens = self.output_norm(tokens + self.feed_forward(tokens))

        valid_weight = candidate_mask.to(tokens.dtype).unsqueeze(-1)
        pooled = (tokens * valid_weight).sum(dim=1) / valid_weight.sum(dim=1).clamp_min(
            1.0
        )
        fixed_proposal_position_residual = None
        fixed_proposal_candidates = None
        if self.one_shot_fixed_proposal_geometry:
            # v9: refine the proposal once using proposal-only tokens.  No
            # keep probability or hard mask enters this path, so the geometry
            # used to build the offline teacher remains exactly fixed during
            # selector distillation and deployment.
            two_sided_spacing = self._two_sided_spacing(candidate_positions)
            left_slack = (two_sided_spacing[..., 0] - self.min_gap).clamp_min(0.0)
            right_slack = (two_sided_spacing[..., 1] - self.min_gap).clamp_min(0.0)
            fixed_proposal_signal = torch.tanh(
                self.position_residual_head(tokens).squeeze(-1)
            )
            fixed_proposal_position_residual = self.max_position_fraction * torch.where(
                fixed_proposal_signal >= 0.0,
                fixed_proposal_signal * right_slack,
                fixed_proposal_signal * left_slack,
            )
            fixed_proposal_position_residual = (
                fixed_proposal_position_residual.masked_fill(~candidate_mask, 0.0)
            )
            fixed_proposal_candidates = (
                candidate_positions + fixed_proposal_position_residual
            )

            # The selector sees the actual fixed proposal locations and then
            # performs its own candidate interaction.  This increases KeepMask
            # capacity without sharing parameters with the position decoder.
            fixed_position_encoding = KnotHead._sinusoidal_position_encoding(
                fixed_proposal_candidates,
                self.hidden_dim,
            )
            selector_tokens = tokens + fixed_position_encoding - position_encoding
            selector_interacted, _ = self.one_shot_selector_attention(
                selector_tokens,
                selector_tokens,
                selector_tokens,
                key_padding_mask=~candidate_mask,
                need_weights=False,
            )
            selector_tokens = self.one_shot_selector_attention_norm(
                selector_tokens + selector_interacted
            )
            final_decision_tokens = self.one_shot_selector_output_norm(
                selector_tokens + self.one_shot_selector_feed_forward(selector_tokens)
            )
            for attention, attention_norm, feed_forward, output_norm in zip(
                self.one_shot_selector_extra_attention,
                self.one_shot_selector_extra_attention_norm,
                self.one_shot_selector_extra_feed_forward,
                self.one_shot_selector_extra_output_norm,
            ):
                extra_interacted, _ = attention(
                    final_decision_tokens,
                    final_decision_tokens,
                    final_decision_tokens,
                    key_padding_mask=~candidate_mask,
                    need_weights=False,
                )
                final_decision_tokens = attention_norm(
                    final_decision_tokens + extra_interacted
                )
                final_decision_tokens = output_norm(
                    final_decision_tokens + feed_forward(final_decision_tokens)
                )
            normalized_global = None
            if global_features is not None:
                normalized_global = F.layer_norm(
                    torch.nan_to_num(global_features),
                    (self.hidden_dim,),
                )
            selector_decision_tokens = final_decision_tokens
            selector_pooled = (selector_decision_tokens * valid_weight).sum(
                dim=1
            ) / valid_weight.sum(dim=1).clamp_min(1.0)
            preliminary_threshold_context = selector_pooled
            if normalized_global is not None:
                preliminary_threshold_context = 0.5 * (
                    selector_pooled + normalized_global
                )
            preliminary_raw_importance = self.keep_head(
                selector_decision_tokens
            ).squeeze(-1)
            preliminary_raw_importance = preliminary_raw_importance.masked_fill(
                ~candidate_mask,
                torch.finfo(preliminary_raw_importance.dtype).min,
            )
            preliminary_adaptive_keep_threshold = self.adaptive_threshold_head(
                preliminary_threshold_context
            ).squeeze(-1)
            preliminary_keep_logits = (
                preliminary_raw_importance
                - preliminary_adaptive_keep_threshold.unsqueeze(-1)
            )
            preliminary_keep_probabilities = torch.sigmoid(preliminary_keep_logits)
            preliminary_soft_keep_weight = (
                preliminary_keep_probabilities * candidate_mask.to(tokens.dtype)
            )
            preliminary_soft_keep_context = (
                selector_decision_tokens * preliminary_soft_keep_weight.unsqueeze(-1)
            ).sum(dim=1) / preliminary_soft_keep_weight.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-6)
            if self.one_shot_joint_position_refinement:
                preliminary_hard_keep_mask, _, _ = self._select_hard_keep_mask(
                    preliminary_keep_probabilities,
                    candidate_mask,
                    fixed_proposal_candidates,
                )
                preliminary_hard_st_keep_gate = self._straight_through_keep_gate(
                    preliminary_keep_probabilities,
                    preliminary_hard_keep_mask,
                    candidate_mask,
                )
                preliminary_hard_st_keep_context = self._hard_st_context(
                    selector_decision_tokens,
                    preliminary_hard_keep_mask,
                    preliminary_hard_st_keep_gate,
                )
                preliminary_context = preliminary_hard_st_keep_context.unsqueeze(
                    1
                ).expand_as(selector_decision_tokens)
                preliminary_position_interaction = torch.cat(
                    [
                        selector_decision_tokens,
                        preliminary_context,
                        selector_decision_tokens * torch.tanh(preliminary_context),
                        preliminary_keep_probabilities.unsqueeze(-1),
                    ],
                    dim=-1,
                )
                provisional_position_refinement_tokens = (
                    self.joint_preliminary_position_norm(
                        selector_decision_tokens
                        + self.joint_preliminary_position_projection(
                            preliminary_position_interaction
                        )
                    )
                )
                provisional_raw_position_residual = torch.tanh(
                    self.joint_preliminary_position_head(
                        provisional_position_refinement_tokens
                    ).squeeze(-1)
                )
                # The preliminary move is not multiplied by p0: a provisional
                # false negative can still move and feed evidence back into p1.
                provisional_keep_conditioned_position_signal = (
                    provisional_raw_position_residual
                )
                provisional_position_residual = self._bounded_all_candidate_residual(
                    fixed_proposal_candidates,
                    provisional_raw_position_residual,
                    candidate_mask,
                )
                provisional_candidates = (
                    fixed_proposal_candidates + provisional_position_residual
                )
                provisional_position_encoding = KnotHead._sinusoidal_position_encoding(
                    provisional_candidates,
                    self.hidden_dim,
                )
                position_feedback_encoding = (
                    provisional_position_encoding - fixed_position_encoding
                )
                position_feedback_state = (
                    provisional_position_refinement_tokens + position_feedback_encoding
                )
                normalized_motion = (
                    provisional_position_residual / self.one_shot_max_position_shift
                ).unsqueeze(-1)
                feedback_interaction = torch.cat(
                    [
                        selector_decision_tokens,
                        position_feedback_state,
                        selector_decision_tokens * torch.tanh(position_feedback_state),
                        preliminary_keep_probabilities.unsqueeze(-1),
                        normalized_motion,
                    ],
                    dim=-1,
                )
                final_decision_tokens = (
                    selector_decision_tokens
                    + self.joint_position_to_keep_norm(
                        self.joint_position_to_keep_feedback(feedback_interaction)
                    )
                )
                final_pooled = (final_decision_tokens * valid_weight).sum(
                    dim=1
                ) / valid_weight.sum(dim=1).clamp_min(1.0)
                threshold_context = final_pooled
                if normalized_global is not None:
                    threshold_context = 0.5 * (final_pooled + normalized_global)
                raw_importance = self.keep_head(final_decision_tokens).squeeze(-1)
                raw_importance = raw_importance.masked_fill(
                    ~candidate_mask,
                    torch.finfo(raw_importance.dtype).min,
                )
                adaptive_keep_threshold = self.adaptive_threshold_head(
                    threshold_context
                ).squeeze(-1)
                keep_logits = raw_importance - adaptive_keep_threshold.unsqueeze(-1)
            else:
                # Exact v9/v10 behavior: one selector pass and fixed proposal
                # positions.  The preliminary aliases intentionally equal the
                # final decision for historical diagnostics.
                raw_importance = preliminary_raw_importance
                adaptive_keep_threshold = preliminary_adaptive_keep_threshold
                keep_logits = preliminary_keep_logits
                provisional_position_refinement_tokens = tokens
                provisional_raw_position_residual = fixed_proposal_signal
                provisional_keep_conditioned_position_signal = fixed_proposal_signal
                provisional_position_residual = fixed_proposal_position_residual
                provisional_candidates = fixed_proposal_candidates
                position_feedback_encoding = fixed_position_encoding - position_encoding
                position_feedback_state = final_decision_tokens
        elif self.one_shot_adaptive:
            # Fixed-pass bidirectional interaction (there is deliberately no
            # data-dependent loop):
            #   keep_0 -> provisional position -> keep_1 -> final position.
            preliminary_raw_importance = self.keep_head(tokens).squeeze(-1)
            preliminary_raw_importance = preliminary_raw_importance.masked_fill(
                ~candidate_mask,
                torch.finfo(preliminary_raw_importance.dtype).min,
            )
            preliminary_threshold_context = pooled
            normalized_global = None
            if global_features is not None:
                normalized_global = F.layer_norm(
                    torch.nan_to_num(global_features),
                    (self.hidden_dim,),
                )
                preliminary_threshold_context = 0.5 * (pooled + normalized_global)
            preliminary_adaptive_keep_threshold = self.adaptive_threshold_head(
                preliminary_threshold_context
            ).squeeze(-1)
            preliminary_keep_logits = (
                preliminary_raw_importance
                - preliminary_adaptive_keep_threshold.unsqueeze(-1)
            )
            preliminary_keep_logits = preliminary_keep_logits.masked_fill(
                ~candidate_mask,
                torch.finfo(preliminary_keep_logits.dtype).min,
            )
            preliminary_keep_probabilities = torch.sigmoid(preliminary_keep_logits)

            two_sided_spacing = self._two_sided_spacing(candidate_positions)
            left_slack = (two_sided_spacing[..., 0] - self.min_gap).clamp_min(0.0)
            right_slack = (two_sided_spacing[..., 1] - self.min_gap).clamp_min(0.0)

            preliminary_soft_keep_weight = (
                preliminary_keep_probabilities * candidate_mask.to(tokens.dtype)
            )
            preliminary_soft_keep_context = (
                tokens * preliminary_soft_keep_weight.unsqueeze(-1)
            ).sum(dim=1) / preliminary_soft_keep_weight.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-6)
            provisional_context = preliminary_soft_keep_context.unsqueeze(1).expand_as(
                tokens
            )
            provisional_keep_interaction = torch.cat(
                [
                    tokens,
                    provisional_context,
                    tokens * torch.tanh(provisional_context),
                    preliminary_keep_probabilities.unsqueeze(-1),
                ],
                dim=-1,
            )
            provisional_position_refinement_tokens = self.keep_context_norm(
                tokens + self.keep_context_projection(provisional_keep_interaction)
            )
            provisional_raw_position_residual = torch.tanh(
                self.position_residual_head(
                    provisional_position_refinement_tokens
                ).squeeze(-1)
            )
            provisional_keep_conditioned_position_signal = (
                provisional_raw_position_residual * preliminary_keep_probabilities
            )
            provisional_position_residual = self.max_position_fraction * torch.where(
                provisional_keep_conditioned_position_signal >= 0.0,
                provisional_keep_conditioned_position_signal * right_slack,
                provisional_keep_conditioned_position_signal * left_slack,
            )
            provisional_position_residual = provisional_position_residual.masked_fill(
                ~candidate_mask,
                0.0,
            )
            provisional_candidates = candidate_positions + provisional_position_residual

            # Feed the *actual provisional locations* back into the keep
            # decision.  Subtracting the original encoding isolates the
            # proposed motion while preserving the local provisional token.
            provisional_position_encoding = KnotHead._sinusoidal_position_encoding(
                provisional_candidates,
                self.hidden_dim,
            )
            position_feedback_encoding = (
                provisional_position_encoding - position_encoding
            )
            position_feedback_state = (
                provisional_position_refinement_tokens + position_feedback_encoding
            )
            position_feedback_interaction = torch.cat(
                [
                    tokens,
                    position_feedback_state,
                    tokens * torch.tanh(position_feedback_state),
                    preliminary_keep_probabilities.unsqueeze(-1),
                ],
                dim=-1,
            )
            final_decision_tokens = self.keep_context_norm(
                tokens + self.position_to_keep_feedback(position_feedback_interaction)
            )
            final_pooled = (final_decision_tokens * valid_weight).sum(
                dim=1
            ) / valid_weight.sum(dim=1).clamp_min(1.0)
            raw_importance = self.keep_head(final_decision_tokens).squeeze(-1)
            raw_importance = raw_importance.masked_fill(
                ~candidate_mask,
                torch.finfo(raw_importance.dtype).min,
            )
            threshold_context = final_pooled
            if normalized_global is not None:
                threshold_context = 0.5 * (final_pooled + normalized_global)
            adaptive_keep_threshold = self.adaptive_threshold_head(
                threshold_context
            ).squeeze(-1)
            keep_logits = raw_importance - adaptive_keep_threshold.unsqueeze(-1)
        else:
            # Exact v7 behavior: keep_head directly emits the Bernoulli logits.
            raw_importance = self.keep_head(tokens).squeeze(-1)
            raw_importance = raw_importance.masked_fill(
                ~candidate_mask,
                torch.finfo(raw_importance.dtype).min,
            )
            adaptive_keep_threshold = raw_importance.new_zeros(raw_importance.shape[0])
            keep_logits = raw_importance
            preliminary_raw_importance = raw_importance
            preliminary_adaptive_keep_threshold = adaptive_keep_threshold
            preliminary_keep_logits = keep_logits
            final_decision_tokens = tokens
        keep_logits = keep_logits.masked_fill(
            ~candidate_mask,
            torch.finfo(keep_logits.dtype).min,
        )
        keep_probabilities = torch.sigmoid(keep_logits)
        if not self.one_shot_adaptive:
            preliminary_keep_probabilities = keep_probabilities

        predicted_deletion_cost = F.softplus(
            self.deletion_cost_head(tokens).squeeze(-1)
        )
        predicted_deletion_cost = predicted_deletion_cost.masked_fill(
            ~candidate_mask,
            0.0,
        )

        # Sequential pruning uses one action per state.  The last category is
        # an explicit STOP action, so K=0 and "no safe deletion" are first-
        # class outcomes rather than a special count-head convention.
        remove_logits = self.remove_action_head(tokens).squeeze(-1)
        remove_logits = remove_logits.masked_fill(
            ~candidate_mask,
            torch.finfo(remove_logits.dtype).min,
        )
        stop_logit = self.stop_action_head(pooled).squeeze(-1)
        remove_stop_logits = torch.cat(
            [remove_logits, stop_logit.unsqueeze(-1)], dim=-1
        )
        predicted_log_deletion_cost = torch.log(
            predicted_deletion_cost.clamp_min(
                torch.finfo(predicted_deletion_cost.dtype).tiny
            )
        )

        if self.one_shot_fixed_proposal_geometry:
            final_soft_keep_weight = keep_probabilities * candidate_mask.to(
                tokens.dtype
            )
            soft_keep_context = (
                final_decision_tokens * final_soft_keep_weight.unsqueeze(-1)
            ).sum(dim=1) / final_soft_keep_weight.sum(dim=1, keepdim=True).clamp_min(
                1e-6
            )
            (
                final_hard_keep_mask,
                one_shot_selected_count,
                one_shot_selection_uncertainty,
            ) = self._select_hard_keep_mask(
                keep_probabilities,
                candidate_mask,
                (
                    provisional_candidates
                    if self.one_shot_joint_position_refinement
                    else fixed_proposal_candidates
                ),
            )
            final_hard_st_keep_gate = self._straight_through_keep_gate(
                keep_probabilities,
                final_hard_keep_mask,
                candidate_mask,
            )
            final_hard_st_keep_context = self._hard_st_context(
                final_decision_tokens,
                final_hard_keep_mask,
                final_hard_st_keep_gate,
            )
            if self.one_shot_joint_position_refinement:
                final_context = final_hard_st_keep_context.unsqueeze(1).expand_as(
                    final_decision_tokens
                )
                final_position_interaction = torch.cat(
                    [
                        final_decision_tokens,
                        final_context,
                        final_decision_tokens * torch.tanh(final_context),
                        final_hard_st_keep_gate.unsqueeze(-1),
                    ],
                    dim=-1,
                )
                position_refinement_tokens = self.joint_final_position_norm(
                    final_decision_tokens
                    + self.joint_final_position_projection(final_position_interaction)
                )
                residual_signal = torch.tanh(
                    self.joint_final_position_head(position_refinement_tokens).squeeze(
                        -1
                    )
                )
                keep_conditioned_signal = residual_signal * final_hard_st_keep_gate
            else:
                position_refinement_tokens = tokens
                residual_signal = fixed_proposal_signal
                keep_conditioned_signal = fixed_proposal_signal
        elif self.one_shot_adaptive:
            # ``soft_keep_context`` remains a useful diagnostic of the final
            # probabilities.  The actual final location path below does not
            # use it: its forward value contains hard-kept candidates only.
            final_soft_keep_weight = keep_probabilities * candidate_mask.to(
                tokens.dtype
            )
            soft_keep_context = (
                final_decision_tokens * final_soft_keep_weight.unsqueeze(-1)
            ).sum(dim=1) / final_soft_keep_weight.sum(dim=1, keepdim=True).clamp_min(
                1e-6
            )

            final_hard_keep_mask = (keep_probabilities >= 0.5) & candidate_mask
            one_shot_selected_count = final_hard_keep_mask.sum(dim=-1).to(torch.long)
            one_shot_selection_uncertainty = (
                (
                    keep_probabilities
                    * (1.0 - keep_probabilities)
                    * candidate_mask.to(keep_probabilities.dtype)
                )
                .sum(dim=-1)
                .clamp_min(0.0)
                .sqrt()
            )
            hard_keep_weight = final_hard_keep_mask.to(tokens.dtype)
            # Straight-through KeepMask: exactly hard in the forward pass,
            # with the sigmoid probability supplying the backward derivative.
            final_hard_st_keep_gate = (
                hard_keep_weight + keep_probabilities - keep_probabilities.detach()
            ) * candidate_mask.to(tokens.dtype)
            hard_count = hard_keep_weight.sum(dim=1, keepdim=True)
            st_count = final_hard_st_keep_gate.sum(dim=1, keepdim=True)
            safe_context_denominator = (
                hard_count.clamp_min(1.0) + st_count - st_count.detach()
            )
            final_hard_st_keep_context = (
                final_decision_tokens * final_hard_st_keep_gate.unsqueeze(-1)
            ).sum(dim=1) / safe_context_denominator
            final_context = final_hard_st_keep_context.unsqueeze(1).expand_as(
                final_decision_tokens
            )
            final_keep_interaction = torch.cat(
                [
                    final_decision_tokens,
                    final_context,
                    final_decision_tokens * torch.tanh(final_context),
                    final_hard_st_keep_gate.unsqueeze(-1),
                ],
                dim=-1,
            )
            position_refinement_tokens = self.keep_context_norm(
                final_decision_tokens
                + self.keep_context_projection(final_keep_interaction)
            )
            residual_signal = torch.tanh(
                self.position_residual_head(position_refinement_tokens).squeeze(-1)
            )
            keep_conditioned_signal = residual_signal * final_hard_st_keep_gate
        else:
            # Keep the old position path bit-for-bit: no selection context was
            # used by v7 checkpoints.
            two_sided_spacing = self._two_sided_spacing(candidate_positions)
            left_slack = (two_sided_spacing[..., 0] - self.min_gap).clamp_min(0.0)
            right_slack = (two_sided_spacing[..., 1] - self.min_gap).clamp_min(0.0)
            soft_keep_weight = keep_probabilities * candidate_mask.to(tokens.dtype)
            soft_keep_context = (tokens * soft_keep_weight.unsqueeze(-1)).sum(
                dim=1
            ) / soft_keep_weight.sum(dim=1, keepdim=True).clamp_min(1e-6)
            position_refinement_tokens = tokens
            residual_signal = torch.tanh(
                self.position_residual_head(position_refinement_tokens).squeeze(-1)
            )
            keep_conditioned_signal = residual_signal
        if self.one_shot_fixed_proposal_geometry:
            if self.one_shot_joint_position_refinement:
                final_position_residual = self._bounded_selected_residual(
                    provisional_candidates,
                    residual_signal,
                    final_hard_keep_mask,
                    candidate_mask,
                    reference_positions=fixed_proposal_candidates,
                )
                refined_candidates = provisional_candidates + final_position_residual
                position_residual = refined_candidates - fixed_proposal_candidates
            else:
                final_position_residual = fixed_proposal_position_residual.new_zeros(
                    fixed_proposal_position_residual.shape
                )
                position_residual = fixed_proposal_position_residual
                refined_candidates = fixed_proposal_candidates
        else:
            final_position_residual = candidate_positions.new_zeros(
                candidate_positions.shape
            )
            position_residual = self.max_position_fraction * torch.where(
                keep_conditioned_signal >= 0.0,
                keep_conditioned_signal * right_slack,
                keep_conditioned_signal * left_slack,
            )
            position_residual = position_residual.masked_fill(~candidate_mask, 0.0)
            refined_candidates = candidate_positions + position_residual

        # v12: the discrete subset is now known.  Relax the retained geometry
        # once more using only final survivors as attention keys/values and
        # using their compact-sequence neighbor/rank/count features.  This is
        # deliberately an optional additive stage: its zero-initialised head
        # makes a strict v11 -> v12 load exactly neutral.
        pre_relocation_candidates = refined_candidates
        relocation_relative_features = candidate_positions.new_zeros(
            (*candidate_positions.shape, 8)
        )
        relocation_previous_position = candidate_positions.new_zeros(
            candidate_positions.shape
        )
        relocation_next_position = candidate_positions.new_ones(
            candidate_positions.shape
        )
        relocation_input_tokens = position_refinement_tokens
        relocation_tokens = position_refinement_tokens
        relocation_attention_weights = candidate_positions.new_zeros(
            (
                candidate_positions.shape[0],
                candidate_positions.shape[1],
                candidate_positions.shape[1],
            )
        )
        relocation_raw_signal = candidate_positions.new_zeros(candidate_positions.shape)
        relocation_position_residual = candidate_positions.new_zeros(
            candidate_positions.shape
        )
        if self.one_shot_survivor_relocation:
            relocation_output = self._relocate_survivors(
                final_decision_tokens,
                pre_relocation_candidates,
                final_hard_keep_mask,
                candidate_mask,
                final_hard_st_keep_gate,
            )
            relocation_relative_features = relocation_output["relative_features"]
            relocation_previous_position = relocation_output["previous_position"]
            relocation_next_position = relocation_output["next_position"]
            relocation_input_tokens = relocation_output["input_tokens"]
            relocation_tokens = relocation_output["tokens"]
            relocation_attention_weights = relocation_output["attention_weights"]
            relocation_raw_signal = relocation_output["raw_signal"]
            relocation_position_residual = self._bounded_selected_residual(
                pre_relocation_candidates,
                relocation_raw_signal,
                final_hard_keep_mask,
                candidate_mask,
                reference_positions=fixed_proposal_candidates,
            )
            refined_candidates = (
                pre_relocation_candidates + relocation_position_residual
            )
            position_residual = refined_candidates - fixed_proposal_candidates

        if not self.one_shot_adaptive:
            # Diagnostic aliases only.  They add no parameters and leave all
            # legacy output tensors and computations unchanged.
            preliminary_soft_keep_context = soft_keep_context
            provisional_position_refinement_tokens = position_refinement_tokens
            provisional_raw_position_residual = residual_signal
            provisional_keep_conditioned_position_signal = keep_conditioned_signal
            provisional_position_residual = position_residual
            provisional_candidates = refined_candidates
            position_feedback_encoding = torch.zeros_like(tokens)
            position_feedback_state = tokens
            final_hard_keep_mask = (keep_probabilities >= 0.5) & candidate_mask
            one_shot_selected_count = final_hard_keep_mask.sum(dim=-1).to(torch.long)
            one_shot_selection_uncertainty = (
                (
                    keep_probabilities
                    * (1.0 - keep_probabilities)
                    * candidate_mask.to(keep_probabilities.dtype)
                )
                .sum(dim=-1)
                .clamp_min(0.0)
                .sqrt()
            )
            final_hard_st_keep_gate = (
                final_hard_keep_mask.to(tokens.dtype)
                + keep_probabilities
                - keep_probabilities.detach()
            ) * candidate_mask.to(tokens.dtype)
            hard_count = final_hard_keep_mask.to(tokens.dtype).sum(dim=1, keepdim=True)
            st_count = final_hard_st_keep_gate.sum(dim=1, keepdim=True)
            safe_context_denominator = (
                hard_count.clamp_min(1.0) + st_count - st_count.detach()
            )
            final_hard_st_keep_context = (
                tokens * final_hard_st_keep_gate.unsqueeze(-1)
            ).sum(dim=1) / safe_context_denominator

        one_shot_probability_mass = (
            keep_probabilities * candidate_mask.to(keep_probabilities.dtype)
        ).sum(dim=-1)
        one_shot_requested_count_score = (
            one_shot_probability_mass
            + self.one_shot_safety_sigma * one_shot_selection_uncertainty
        )
        proposal_candidates = (
            fixed_proposal_candidates
            if fixed_proposal_candidates is not None
            else candidate_positions
        )

        return {
            "pruning_tokens": tokens,
            "analytic_contribution_features": analytic_features,
            "pruning_attention_weights": attention_weights,
            "preliminary_raw_importance": preliminary_raw_importance,
            "preliminary_adaptive_keep_threshold": (
                preliminary_adaptive_keep_threshold
            ),
            "preliminary_keep_logits": preliminary_keep_logits,
            "preliminary_keep_probabilities": preliminary_keep_probabilities,
            "preliminary_keep_probability": preliminary_keep_probabilities,
            "preliminary_soft_keep_context": preliminary_soft_keep_context,
            "provisional_position_refinement_tokens": (
                provisional_position_refinement_tokens
            ),
            "provisional_raw_position_residual": (provisional_raw_position_residual),
            "provisional_keep_conditioned_position_signal": (
                provisional_keep_conditioned_position_signal
            ),
            "provisional_position_residual": provisional_position_residual,
            "provisional_candidate_positions": provisional_candidates,
            "provisional_candidate_knots": provisional_candidates,
            "position_feedback_encoding": position_feedback_encoding,
            "position_feedback_tokens": position_feedback_state,
            "final_decision_tokens": final_decision_tokens,
            "raw_importance": raw_importance,
            "raw_keep_importance": raw_importance,
            "final_raw_importance": raw_importance,
            "adaptive_keep_threshold": adaptive_keep_threshold,
            "adaptive_keep_logit_threshold": adaptive_keep_threshold,
            "adaptive_raw_importance_probability_threshold": torch.sigmoid(
                adaptive_keep_threshold
            ),
            "keep_threshold": adaptive_keep_threshold,
            "keep_logits": keep_logits,
            "keep_probabilities": keep_probabilities,
            "keep_probability": keep_probabilities,
            "final_keep_logits": keep_logits,
            "final_keep_probabilities": keep_probabilities,
            "final_keep_probability": keep_probabilities,
            "final_hard_keep_mask": final_hard_keep_mask,
            "one_shot_selected_count": one_shot_selected_count,
            "one_shot_selection_uncertainty": one_shot_selection_uncertainty,
            "one_shot_probability_mass": one_shot_probability_mass,
            "one_shot_requested_count_score": one_shot_requested_count_score,
            "final_hard_st_keep_gate": final_hard_st_keep_gate,
            "final_hard_st_keep_context": final_hard_st_keep_context,
            "predicted_deletion_cost": predicted_deletion_cost,
            "remove_logits": remove_logits,
            "stop_logit": stop_logit,
            "remove_stop_logits": remove_stop_logits,
            "remove_stop_probability": F.softmax(remove_stop_logits, dim=-1),
            "predicted_log_deletion_cost": predicted_log_deletion_cost,
            "soft_keep_context": soft_keep_context,
            "position_refinement_tokens": position_refinement_tokens,
            "raw_position_residual": residual_signal,
            "keep_conditioned_position_signal": keep_conditioned_signal,
            "position_residual": position_residual,
            "final_position_residual": final_position_residual,
            "pre_relocation_candidate_positions": pre_relocation_candidates,
            "pre_relocation_candidate_knots": pre_relocation_candidates,
            "relocation_relative_features": relocation_relative_features,
            "relocation_previous_survivor_position": (relocation_previous_position),
            "relocation_next_survivor_position": relocation_next_position,
            "relocation_input_tokens": relocation_input_tokens,
            "relocation_tokens": relocation_tokens,
            "relocation_attention_weights": relocation_attention_weights,
            "relocation_raw_position_signal": relocation_raw_signal,
            "relocation_position_residual": relocation_position_residual,
            "relocation_selected_mask": final_hard_keep_mask,
            "proposal_candidate_positions": proposal_candidates,
            "proposal_candidate_knots": proposal_candidates,
            "refined_candidate_positions": refined_candidates,
            "refined_candidate_knots": refined_candidates,
            "deployment_candidate_positions": refined_candidates,
            "deployment_candidate_knots": refined_candidates,
            "candidate_mask": candidate_mask,
        }
