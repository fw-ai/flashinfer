"""Blackwell acceptance tests for the private b552 SiTU MoE runner.

These tests intentionally require the matched private cubin and JIT-cache
wheels.  They never JIT compile or download artifacts.  CPU-only and other GPU
jobs skip before loading the private AOT module.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
import gc
import math
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Literal

import pytest
import torch
from torch.nn import functional as F

from flashinfer import (
    ActivationType,
    RoutingMethodType,
    mxfp8_dequantize_host,
    mxfp8_quantize,
)
from flashinfer.autotuner import AutoTuner
from flashinfer.fused_moe import WeightLayout
from flashinfer.fused_moe._situ_b552 import (
    trtllm_fp4_block_scale_situ_moe,
    trtllm_fp4_block_scale_situ_routed_moe,
)
from flashinfer.fused_moe.core import (
    Fp8QuantizationType,
    get_trtllm_moe_situ_b552_module,
)
from flashinfer.jit.situ_b552 import (
    gen_trtllm_gen_fused_moe_situ_b552_module,
    get_situ_b552_artifact_root,
    load_trtllm_gen_fused_moe_situ_b552_module,
    verify_situ_b552_artifacts,
)
from flashinfer.tllm_enums import DtypeTrtllmGen
from flashinfer.utils import device_support_pdl, get_compute_capability

from .test_trtllm_gen_fused_moe import (
    FP4Moe,
    QuantMode,
    e2m1_and_ufp8_scale_batches,
    moe_args,
)
from .test_trtllm_gen_moe_autotune_tactics import (
    _force_tactic_in_autotuner_cache,
    _last_positive_power_of_2,
    _moe_profile_shapes,
)

_PRIVATE_CUSTOM_OP = (
    "flashinfer::trtllm_fp4_block_scale_moe_situ_b552::unpackedprecomputed"
)
_PRIVATE_LOG_KEY = (_PRIVATE_CUSTOM_OP, "MoERunner")
_EXPECTED_FC1_CUBINS = 102
_EXPECTED_FC2_CUBINS = 101
_METADATA_SYMBOL = re.compile(r'"(bmm_[^"]+)"')
_TILE_N = re.compile(r"_t\d+x(\d+)x\d+(?:u\d+)?_")
_K3_NUM_EXPERTS = 896
_K3_TOP_K = 16
_K3_COMMON_EXPERTS = (0, 7, 31, 42, 63, 127, 255, 383, 511, 639, 767, 895)


@dataclass(frozen=True, order=True)
class _Shape:
    num_tokens: int
    hidden_size: int
    intermediate_size: int
    num_experts: int = 8
    top_k: int = 2


@dataclass
class _SituCase:
    shape: _Shape
    logits: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    packed_topk: torch.Tensor
    hidden_states: torch.Tensor
    hidden_states_scale: torch.Tensor
    hidden_states_dequant: torch.Tensor
    gemm1_weights: torch.Tensor
    gemm1_weights_scale: torch.Tensor
    gemm2_weights: torch.Tensor
    gemm2_weights_scale: torch.Tensor
    gemm1_weights_dequant: torch.Tensor
    gemm2_weights_dequant: torch.Tensor
    output1_scale_scalar: torch.Tensor
    output1_scale_gate_scalar: torch.Tensor
    output2_scale_scalar: torch.Tensor


@dataclass(frozen=True)
class _ReferenceStats:
    max_abs_up: float
    max_abs_gate: float
    max_abs_up_minus_gate: float


class _PrivateMetadataIndex:
    """Map a public ``[tile_N, MoE-config]`` tactic to its FC1 cubin.

    The immutable b552 ``MoE::Runner`` builds ``mPassingConfigs`` as the
    Cartesian product of the matching FC1 indices (outer loop) and FC2 indices
    (inner loop).  Both lists preserve ``flashinferMetaInfo.h`` order.  That
    makes the FC1 position ``config // len(fc2_for_tile)``.
    """

    def __init__(self, artifact_root: Path):
        metainfo = artifact_root / "include/flashinferMetaInfo.h"
        symbols = _METADATA_SYMBOL.findall(metainfo.read_text())
        fc1_symbols = [symbol for symbol in symbols if "_situ_" in symbol]
        fc2_symbols = [
            symbol
            for symbol in symbols
            if symbol.startswith("bmm_Bfloat16_MxE2m1MxE4m3_Fp32_bA32_bB32_")
        ]

        cubin_symbols = {
            "bmm_" + path.stem.removeprefix("Bmm_")
            for path in artifact_root.glob("Bmm_*_situ_*.cubin")
        }
        assert len(fc1_symbols) == _EXPECTED_FC1_CUBINS
        assert len(cubin_symbols) == _EXPECTED_FC1_CUBINS
        assert set(fc1_symbols) == cubin_symbols
        assert len(fc2_symbols) == _EXPECTED_FC2_CUBINS

        self.expected_fc1_symbols = frozenset(fc1_symbols)
        self.fc1_by_tile: dict[int, list[str]] = defaultdict(list)
        self.fc2_by_tile: dict[int, list[str]] = defaultdict(list)
        for symbol in fc1_symbols:
            self.fc1_by_tile[self._tile_n(symbol)].append(symbol)
        for symbol in fc2_symbols:
            self.fc2_by_tile[self._tile_n(symbol)].append(symbol)
        assert set(self.fc1_by_tile) == {8, 16, 32, 64, 128, 256}
        assert set(self.fc1_by_tile) == set(self.fc2_by_tile)

    @staticmethod
    def _tile_n(symbol: str) -> int:
        match = _TILE_N.search(symbol)
        assert match is not None, f"cannot parse tile_N from {symbol}"
        return int(match.group(1))

    def fc1_symbol(self, tactic: tuple[int, int]) -> str:
        tile_n, config_index = tactic
        fc1 = self.fc1_by_tile[tile_n]
        fc2 = self.fc2_by_tile[tile_n]
        assert 0 <= config_index < len(fc1) * len(fc2), (
            f"private tactic {tactic} exceeds the b552 FC1xFC2 table "
            f"({len(fc1)}x{len(fc2)})"
        )
        return fc1[config_index // len(fc2)]


def _require_matched_private_blackwell() -> Path:
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    capability = get_compute_capability(torch.device("cuda:0"))
    if capability not in ((10, 0), (10, 3)):
        pytest.skip("requires SM100/B200 or SM103/B300")

    artifact_root = get_situ_b552_artifact_root()
    if not artifact_root.is_dir():
        pytest.skip("matched private SiTU b552 cubin wheel is not installed")
    if (
        not (artifact_root / "checksums.txt").is_file()
        or not (artifact_root / "manifest.sha256").is_file()
    ):
        pytest.skip("matched private SiTU b552 manifest is not installed")

    # Once a private bundle is present, corruption or an ABI mismatch is a hard
    # test failure rather than a skip.
    verify_situ_b552_artifacts()
    spec = gen_trtllm_gen_fused_moe_situ_b552_module()
    if not spec.is_aot:
        raise RuntimeError(
            "private SiTU cubins are installed but the matched AOT module is missing"
        )
    load_trtllm_gen_fused_moe_situ_b552_module()
    return artifact_root


def _routing_inputs(shape: _Shape, device: torch.device):
    logits = torch.full(
        (shape.num_tokens, shape.num_experts),
        -12.0,
        dtype=torch.float32,
        device=device,
    )
    # Expert 0 receives every token, expert 1 receives all but one, expert 2
    # receives one, and the remaining experts are empty.  Small row-dependent
    # offsets avoid ties while preserving the deliberately imbalanced routing.
    row_offset = torch.arange(shape.num_tokens, device=device, dtype=torch.float32)
    row_offset = row_offset / max(shape.num_tokens, 1) / 8.0
    logits[:, 0] = 6.0 + row_offset
    logits[:, 1] = 2.0 - row_offset
    logits[-1, 2] = 4.0
    if shape.top_k > 2:
        rows = torch.arange(shape.num_tokens, device=device)
        token_experts = (
            shape.num_experts
            - 1
            - (rows % max(min(shape.num_experts - 3, shape.num_tokens), 1))
        )
        logits[rows, token_experts] = 1.0 - row_offset

    topk_logits, topk_ids = torch.topk(logits.float(), shape.top_k, dim=-1)
    topk_ids = topk_ids.to(torch.int32)
    topk_weights = torch.softmax(topk_logits, dim=-1).to(torch.bfloat16)
    packed_topk = (topk_ids << 16) | topk_weights.view(torch.int16).to(torch.int32)

    counts = torch.bincount(topk_ids.reshape(-1), minlength=shape.num_experts).cpu()
    # Expert 0/1 plus at most ``top_k - 2`` tied tail experts are selected on
    # every row; one token-varying expert per row makes the routing imbalanced.
    assert int((counts == 0).sum()) >= shape.num_experts - (
        shape.top_k + shape.num_tokens + 1
    )
    nonempty = counts[counts > 0]
    assert int(nonempty.max()) > int(nonempty.min())
    return logits, topk_ids, topk_weights, packed_topk


def _build_case(shape: _Shape, seed: int = 17) -> _SituCase:
    device = torch.device("cuda:0")
    torch.manual_seed(seed)

    hidden_orig = (
        torch.randn(
            shape.num_tokens,
            shape.hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.35
    )
    hidden_orig[0].fill_(1.0)
    hidden_orig[1].fill_(-1.0)
    hidden_orig[2] = torch.linspace(
        -1.5,
        2.0,
        shape.hidden_size,
        device=device,
        dtype=torch.float32,
    ).to(torch.bfloat16)

    gemm1_orig = torch.randn(
        shape.num_experts,
        2 * shape.intermediate_size,
        shape.hidden_size,
        device=device,
        dtype=torch.bfloat16,
    ) / math.sqrt(shape.hidden_size)
    # The first rows of each [up, gate] half deliberately produce different,
    # saturated values for the all-ones rows.  The remaining random rows retain
    # broad sign and magnitude coverage.
    saturation_rows = min(128, shape.intermediate_size // 4)
    for expert in range(shape.num_experts):
        up_sign = 1.0 if expert % 2 == 0 else -1.0
        gemm1_orig[expert, :saturation_rows].fill_(up_sign * 0.0625)
        gate_start = shape.intermediate_size
        gemm1_orig[expert, gate_start : gate_start + saturation_rows].fill_(0.03125)
    gemm2_orig = torch.randn(
        shape.num_experts,
        shape.hidden_size,
        shape.intermediate_size,
        device=device,
        dtype=torch.bfloat16,
    ) / math.sqrt(shape.intermediate_size)

    moe_impl = FP4Moe(QuantMode.FP4_MXFP4_MXFP8)
    moe_impl._cache_permute_indices = {}
    weights = moe_impl.quantize_weights(gemm1_orig, gemm2_orig, hidden_orig)
    inputs = moe_impl.quantize_inputs(
        hidden_orig, weights["hidden_states_scale_global"], is_swizzling=False
    )
    assert inputs["hidden_states"].dtype == torch.float8_e4m3fn
    assert inputs["hidden_states_scale"].shape == (
        shape.num_tokens,
        shape.hidden_size // 32,
    )
    args = moe_args(
        shape.num_tokens,
        shape.num_experts,
        shape.hidden_size,
        shape.intermediate_size,
        shape.top_k,
        8,
        inputs["hidden_states"],
        inputs["hidden_states_scale"],
        weights["hidden_states_scale_global"],
        torch.empty(0, device=device, dtype=torch.bfloat16),
        weights["gemm1_weights"],
        weights["gemm1_scales"],
        weights["gemm1_scales_global"],
        weights["gemm2_weights"],
        weights["gemm2_scales"],
        weights["gemm2_scales_global"],
        {},
        False,
        ActivationType.Swiglu,
    )
    static = moe_impl.prepare_static_weights_for_kernel(
        SimpleNamespace(c_global_sf=1.0),
        args,
        gemm1_orig,
        gemm2_orig,
        shape.hidden_size,
        shape.intermediate_size,
        shape.num_experts,
        {},
    )

    hidden_dequant = mxfp8_dequantize_host(
        inputs["hidden_states"].cpu().view(torch.uint8),
        inputs["hidden_states_scale"].cpu().view(torch.uint8).reshape(-1),
        False,
    ).to(device=device, dtype=torch.float32)
    gemm1_dequant = e2m1_and_ufp8_scale_batches(
        weights["gemm1_weights"],
        weights["gemm1_scales"],
        1.0 / weights["gemm1_scales_global"],
        32,
        0,
    ).to(device=device, dtype=torch.float32)
    gemm2_dequant = e2m1_and_ufp8_scale_batches(
        weights["gemm2_weights"],
        weights["gemm2_scales"],
        1.0 / weights["gemm2_scales_global"],
        32,
        0,
    ).to(device=device, dtype=torch.float32)
    logits, topk_ids, topk_weights, packed_topk = _routing_inputs(shape, device)

    case = _SituCase(
        shape=shape,
        logits=logits,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        packed_topk=packed_topk,
        hidden_states=inputs["hidden_states"],
        hidden_states_scale=inputs["hidden_states_scale"],
        hidden_states_dequant=hidden_dequant,
        gemm1_weights=static["gemm1_weights_fp4_shuffled"],
        gemm1_weights_scale=static["gemm1_scales_fp4_shuffled"],
        gemm2_weights=static["gemm2_weights_fp4_shuffled"],
        gemm2_weights_scale=static["gemm2_scales_fp4_shuffled"],
        gemm1_weights_dequant=gemm1_dequant,
        gemm2_weights_dequant=gemm2_dequant,
        output1_scale_scalar=static["scale_c_fc1"].to(device),
        output1_scale_gate_scalar=static["scale_gate_fc1"].to(device),
        output2_scale_scalar=static["scale_c_fc2"].to(device),
    )
    del hidden_orig, gemm1_orig, gemm2_orig, weights, args, static, inputs, moe_impl
    return case


def _situ_and_mul(
    up: torch.Tensor,
    gate: torch.Tensor,
    beta: float,
    linear_beta: float | None,
) -> torch.Tensor:
    gate_term = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    up_term = up if linear_beta is None else linear_beta * torch.tanh(up / linear_beta)
    return gate_term * up_term


def _k3_routing_reference(
    num_tokens: int,
    device: torch.device,
    logits_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a tie-free K3 no-aux routing case and its PyTorch reference.

    K3 has a single expert group, so DeepSeek-v3 routing reduces to selecting
    top-16 from ``sigmoid(logits) + correction_bias`` and normalizing the
    *unbiased* sigmoid scores.  Twelve experts are shared by every token and
    four are token-specific, deliberately creating hot, cold, and empty
    experts.  FP32 correction bias also excludes high-logit decoys, ensuring
    the test would fail if the runner ignored the bias.
    """
    logits_fp32 = torch.full(
        (num_tokens, _K3_NUM_EXPERTS),
        -9.0,
        dtype=torch.float32,
        device=device,
    )
    routing_bias = torch.linspace(
        -0.01,
        0.01,
        _K3_NUM_EXPERTS,
        dtype=torch.float32,
        device=device,
    )
    common_scores = torch.linspace(
        2.4,
        1.3,
        len(_K3_COMMON_EXPERTS),
        dtype=torch.float32,
        device=device,
    )
    common_ids = torch.tensor(_K3_COMMON_EXPERTS, dtype=torch.long, device=device)
    unique_scores = torch.tensor(
        [1.25, 1.15, 1.05, 0.95], dtype=torch.float32, device=device
    )

    expected_ids: list[set[int]] = []
    unique_ids_by_token: list[torch.Tensor] = []
    decoy_ids: list[int] = []
    for token in range(num_tokens):
        # Keep at least 850 experts empty even when the swizzled-scale fixture
        # uses 32 rows. Six four-expert groups are reused, while the final row
        # gets a seventh group exactly once to retain a cold expert.
        unique_group = (
            (6 if num_tokens > 7 and token == num_tokens - 1 else token % 6)
            if num_tokens > 7
            else token
        )
        unique_ids = torch.arange(
            700 + 4 * unique_group,
            704 + 4 * unique_group,
            dtype=torch.long,
            device=device,
        )
        decoy_id = 200 + token
        logits_fp32[token, common_ids] = common_scores
        logits_fp32[token, unique_ids] = unique_scores
        logits_fp32[token, decoy_id] = 1.7
        unique_ids_by_token.append(unique_ids)
        decoy_ids.append(decoy_id)
        expected_ids.append(set(_K3_COMMON_EXPERTS) | set(unique_ids.tolist()))

    routing_bias[torch.cat(unique_ids_by_token)] += 0.03
    routing_bias[torch.tensor(decoy_ids, device=device)] = -0.40
    logits = logits_fp32.to(logits_dtype)

    sigmoid_scores = torch.sigmoid(logits.float())
    biased_scores = sigmoid_scores + routing_bias
    _, topk_ids = torch.topk(
        biased_scores, _K3_TOP_K, dim=-1, largest=True, sorted=True
    )
    _, unbiased_topk_ids = torch.topk(
        sigmoid_scores, _K3_TOP_K, dim=-1, largest=True, sorted=True
    )
    selected_scores = sigmoid_scores.gather(1, topk_ids)
    topk_weights = selected_scores / selected_scores.sum(dim=-1, keepdim=True)

    for token in range(num_tokens):
        assert set(topk_ids[token].tolist()) == expected_ids[token]
        assert decoy_ids[token] not in expected_ids[token]
        assert set(unbiased_topk_ids[token].tolist()) != expected_ids[token]
    torch.testing.assert_close(
        topk_weights.sum(dim=-1),
        torch.ones(num_tokens, dtype=torch.float32, device=device),
        rtol=1e-6,
        atol=1e-6,
    )
    return (
        logits,
        routing_bias,
        topk_ids.to(torch.int32),
        topk_weights.to(torch.bfloat16),
    )


def _grouped_deepseek_routing_reference(
    shape: _Shape,
    n_group: int,
    topk_group: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a tie-free grouped DeepSeek-v3 route and its exact result.

    Each selected group has two moderately strong experts, while every pruned
    group has one stronger decoy.  The decoys would win a global top-k, but
    their groups lose the top-two-sum group score.  Correction bias also swaps
    the second selected expert in every kept group, so this case independently
    detects omitted group pruning and omitted routing bias.
    """
    assert n_group > 1
    assert 1 < topk_group < n_group
    assert shape.num_experts % n_group == 0
    assert shape.top_k == 2 * topk_group
    group_size = shape.num_experts // n_group
    assert group_size >= 4

    device = torch.device("cuda:0")
    logits = torch.full(
        (shape.num_tokens, shape.num_experts),
        -6.0,
        dtype=torch.float32,
        device=device,
    )
    routing_bias = torch.zeros(shape.num_experts, dtype=torch.float32, device=device)
    for group in range(n_group):
        base = group * group_size
        routing_bias[base + 1] = -0.04
        routing_bias[base + 2] = 0.08

    expected_groups: list[set[int]] = []
    for token in range(shape.num_tokens):
        kept_groups = {token % n_group, (token + 1) % n_group}
        expected_groups.append(kept_groups)
        row_offset = token / max(shape.num_tokens, 1) / 100.0
        for order, group in enumerate(sorted(kept_groups)):
            base = group * group_size
            group_offset = order / 100.0 + row_offset
            logits[token, base] = 2.00 + group_offset
            logits[token, base + 1] = 1.85 + group_offset
            logits[token, base + 2] = 1.80 + group_offset
        for group in set(range(n_group)) - kept_groups:
            # A strong individual decoy with no strong partner loses the group
            # top-two sum but would survive an ungrouped global top-k.
            logits[token, group * group_size + 3] = 4.0 - group / 100.0

    sigmoid_scores = torch.sigmoid(logits)
    biased_scores = sigmoid_scores + routing_bias
    grouped_scores = biased_scores.view(shape.num_tokens, n_group, group_size)
    group_scores = torch.topk(grouped_scores, 2, dim=-1).values.sum(dim=-1)
    selected_groups = torch.topk(group_scores, topk_group, dim=-1).indices
    group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
    group_mask.scatter_(1, selected_groups, True)
    expert_mask = (
        group_mask.unsqueeze(-1)
        .expand(shape.num_tokens, n_group, group_size)
        .reshape(shape.num_tokens, shape.num_experts)
    )

    pruned_scores = biased_scores.masked_fill(~expert_mask, float("-inf"))
    topk_ids = torch.topk(pruned_scores, shape.top_k, dim=-1).indices
    ungrouped_ids = torch.topk(biased_scores, shape.top_k, dim=-1).indices
    unbiased_grouped_ids = torch.topk(
        sigmoid_scores.masked_fill(~expert_mask, float("-inf")),
        shape.top_k,
        dim=-1,
    ).indices

    for token in range(shape.num_tokens):
        actual_groups = set((selected_groups[token] % n_group).tolist())
        assert actual_groups == expected_groups[token]
        assert set(topk_ids[token].tolist()) != set(ungrouped_ids[token].tolist())
        assert set(topk_ids[token].tolist()) != set(
            unbiased_grouped_ids[token].tolist()
        )

    selected_scores = sigmoid_scores.gather(1, topk_ids)
    topk_weights = selected_scores / selected_scores.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(
        topk_weights.sum(dim=-1),
        torch.ones(shape.num_tokens, dtype=torch.float32, device=device),
        rtol=1e-6,
        atol=1e-6,
    )
    return (
        logits,
        routing_bias,
        topk_ids.to(torch.int32),
        topk_weights.to(torch.bfloat16),
    )


def _explicit_reference(
    case: _SituCase,
    beta: float,
    linear_beta: float | None,
    local_expert_offset: int = 0,
    local_num_experts: int | None = None,
) -> tuple[torch.Tensor, _ReferenceStats]:
    shape = case.shape
    if local_num_experts is None:
        local_num_experts = shape.num_experts
    assert 0 <= local_expert_offset < shape.num_experts
    assert 0 < local_num_experts <= shape.num_experts
    assert local_expert_offset + local_num_experts <= shape.num_experts
    local_expert_stop = local_expert_offset + local_num_experts
    flat_ids = case.topk_ids.reshape(-1).long()
    token_ids = (
        torch.arange(shape.num_tokens, device=flat_ids.device)
        .unsqueeze(1)
        .expand(-1, shape.top_k)
        .reshape(-1)
    )
    activation = torch.zeros(
        flat_ids.numel(),
        shape.intermediate_size,
        dtype=torch.float32,
        device=flat_ids.device,
    )

    max_abs_up = 0.0
    max_abs_gate = 0.0
    max_abs_up_minus_gate = 0.0
    for expert in range(local_expert_offset, local_expert_stop):
        positions = torch.nonzero(flat_ids == expert, as_tuple=False).flatten()
        if positions.numel() == 0:
            continue
        fc1 = (
            case.hidden_states_dequant[token_ids[positions]]
            @ case.gemm1_weights_dequant[expert].t()
        )
        # Fireworks stores FC1 as [up, gate].
        up = fc1[:, : shape.intermediate_size]
        gate = fc1[:, shape.intermediate_size :]
        max_abs_up = max(max_abs_up, float(up.abs().max()))
        max_abs_gate = max(max_abs_gate, float(gate.abs().max()))
        max_abs_up_minus_gate = max(
            max_abs_up_minus_gate, float((up - gate).abs().max())
        )
        activation[positions] = _situ_and_mul(up, gate, beta, linear_beta)

    # FC1's private epilogue quantizes the SiTU result to MXFP8 with block-32
    # scales before FC2.  Round-trip the reference through the same primitive;
    # comparing an unquantized activation would test a different numeric path.
    activation_quant, activation_scale = mxfp8_quantize(
        activation.to(torch.bfloat16), True
    )
    assert activation_quant.dtype == torch.float8_e4m3fn
    padded_scale_rows = ((activation.shape[0] + 127) // 128) * 128
    padded_scale_cols = (((activation.shape[1] // 32) + 3) // 4) * 4
    assert activation_scale.numel() == padded_scale_rows * padded_scale_cols
    activation_dequant = mxfp8_dequantize_host(
        activation_quant.cpu().view(torch.uint8),
        activation_scale.cpu().view(torch.uint8).reshape(-1),
        True,
    ).to(device=activation.device, dtype=torch.float32)

    routed_output = torch.zeros(
        flat_ids.numel(),
        shape.hidden_size,
        dtype=torch.float32,
        device=flat_ids.device,
    )
    for expert in range(local_expert_offset, local_expert_stop):
        positions = torch.nonzero(flat_ids == expert, as_tuple=False).flatten()
        if positions.numel() == 0:
            continue
        routed_output[positions] = (
            activation_dequant[positions] @ case.gemm2_weights_dequant[expert].t()
        )
    routed_output = routed_output.view(shape.num_tokens, shape.top_k, shape.hidden_size)
    finalized = (routed_output * case.topk_weights.float().unsqueeze(-1)).sum(dim=1)
    return finalized, _ReferenceStats(
        max_abs_up=max_abs_up,
        max_abs_gate=max_abs_gate,
        max_abs_up_minus_gate=max_abs_up_minus_gate,
    )


def _kernel_kwargs(
    case: _SituCase,
    linear_beta: float | None,
    do_finalize: bool,
    tune_max_num_tokens: int,
    local_expert_offset: int = 0,
    local_num_experts: int | None = None,
) -> dict:
    shape = case.shape
    device = case.hidden_states.device
    if local_num_experts is None:
        local_num_experts = shape.num_experts
    assert 0 <= local_expert_offset < shape.num_experts
    assert 0 < local_num_experts <= shape.num_experts
    assert local_expert_offset + local_num_experts <= shape.num_experts
    local_slice = slice(local_expert_offset, local_expert_offset + local_num_experts)

    def local(tensor: torch.Tensor) -> torch.Tensor:
        return tensor[local_slice].contiguous()

    return dict(
        routing_bias=None,
        hidden_states=case.hidden_states,
        hidden_states_scale=case.hidden_states_scale,
        gemm1_weights=local(case.gemm1_weights),
        gemm1_weights_scale=local(case.gemm1_weights_scale),
        gemm1_bias=None,
        gemm1_alpha=torch.full(
            (local_num_experts,), 4.0, device=device, dtype=torch.float32
        ),
        gemm1_beta=(
            None
            if linear_beta is None
            else torch.full(
                (local_num_experts,),
                linear_beta,
                device=device,
                dtype=torch.float32,
            )
        ),
        gemm1_clamp_limit=None,
        gemm2_weights=local(case.gemm2_weights),
        gemm2_weights_scale=local(case.gemm2_weights_scale),
        gemm2_bias=None,
        output1_scale_scalar=local(case.output1_scale_scalar),
        output1_scale_gate_scalar=local(case.output1_scale_gate_scalar),
        output2_scale_scalar=local(case.output2_scale_scalar),
        num_experts=shape.num_experts,
        top_k=shape.top_k,
        n_group=None,
        topk_group=None,
        intermediate_size=shape.intermediate_size,
        local_expert_offset=local_expert_offset,
        local_num_experts=local_num_experts,
        routed_scaling_factor=None,
        routing_method_type=RoutingMethodType.Renormalize.value,
        do_finalize=do_finalize,
        enable_pdl=device_support_pdl(device),
        per_token_scale=None,
        tune_max_num_tokens=tune_max_num_tokens,
    )


def _run_private_api(
    case: _SituCase,
    routing_api: Literal["logits", "packed", "unpacked"],
    linear_beta: float | None,
    do_finalize: bool,
    tune_max_num_tokens: int,
) -> list[torch.Tensor]:
    kwargs = _kernel_kwargs(
        case, linear_beta, do_finalize, tune_max_num_tokens=tune_max_num_tokens
    )
    if routing_api == "logits":
        return trtllm_fp4_block_scale_situ_moe(routing_logits=case.logits, **kwargs)
    if routing_api == "packed":
        packed_before = case.packed_topk.clone()
        result = trtllm_fp4_block_scale_situ_routed_moe(
            topk_ids=case.packed_topk, **kwargs
        )
        assert torch.equal(case.packed_topk, packed_before)
        return result
    return trtllm_fp4_block_scale_situ_routed_moe(
        topk_ids=(case.topk_ids, case.topk_weights), **kwargs
    )


def _finalize_result(
    result: list[torch.Tensor], case: _SituCase, do_finalize: bool
) -> torch.Tensor:
    if do_finalize:
        assert len(result) == 1
        return result[0].float()

    assert len(result) == 3
    gemm2_output, expert_weights, expanded_to_permuted = result
    shape = case.shape
    assert expert_weights.shape == (shape.num_tokens, shape.top_k)
    mapping = expanded_to_permuted.reshape(-1)[: shape.num_tokens * shape.top_k]
    mapping = mapping.reshape(shape.num_tokens, shape.top_k).long()
    assert bool((mapping >= 0).all())
    assert int(mapping.max()) < gemm2_output.shape[0]
    routed = gemm2_output.float()[mapping]
    return (routed * expert_weights.float().unsqueeze(-1)).sum(dim=1)


def _assert_quantized_reference_close(
    actual: torch.Tensor, reference: torch.Tensor, context: str
) -> None:
    actual = actual.float()
    reference = reference.float()
    assert torch.isfinite(actual).all(), f"{context}: non-finite kernel output"
    assert torch.isfinite(reference).all(), f"{context}: non-finite reference"

    close_fraction = (
        torch.isclose(actual, reference, rtol=0.85, atol=0.15).float().mean().item()
    )
    cosine = F.cosine_similarity(
        actual.reshape(1, -1), reference.reshape(1, -1), dim=1
    ).item()
    normalized_rmse = (
        torch.linalg.vector_norm(actual - reference)
        / torch.linalg.vector_norm(reference).clamp_min(1e-6)
    ).item()
    assert close_fraction >= 0.90, (
        f"{context}: only {close_fraction:.2%} of values match the quantized "
        f"reference; cosine={cosine:.6f}, normalized_rmse={normalized_rmse:.6f}"
    )
    assert cosine >= 0.95, (
        f"{context}: cosine={cosine:.6f}, normalized_rmse={normalized_rmse:.6f}"
    )
    assert normalized_rmse <= 0.35, (
        f"{context}: normalized_rmse={normalized_rmse:.6f}, cosine={cosine:.6f}"
    )


@pytest.fixture(scope="module")
def _situ_numeric_case():
    _require_matched_private_blackwell()
    case = _build_case(_Shape(9, 1024, 1024))
    yield case
    del case
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def _situ_k3_case():
    _require_matched_private_blackwell()
    # The expert count/top-k/routing contract is exact K3.  Only H and I are
    # reduced to the smallest b552-compatible multiples to keep this acceptance
    # case comfortably below production-model memory requirements.
    case = _build_case(
        _Shape(
            num_tokens=7,
            hidden_size=256,
            intermediate_size=256,
            num_experts=_K3_NUM_EXPERTS,
            top_k=_K3_TOP_K,
        ),
        seed=307,
    )
    assert _valid_private_tactics(get_trtllm_moe_situ_b552_module(), case.shape)
    yield case
    del case
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def _situ_grouped_case():
    _require_matched_private_blackwell()
    # Group routing needs enough experts to make pruned-group decoys
    # unambiguous, but this is deliberately much smaller than the K3 fixture.
    case = _build_case(
        _Shape(
            num_tokens=8,
            hidden_size=256,
            intermediate_size=256,
            num_experts=64,
            top_k=4,
        ),
        seed=409,
    )
    assert _valid_private_tactics(get_trtllm_moe_situ_b552_module(), case.shape)
    yield case
    del case
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.parametrize("routing_api", ["logits", "packed", "unpacked"])
@pytest.mark.parametrize(
    "linear_beta", [25.0, None], ids=["linear_beta_25", "unclamped_up"]
)
@pytest.mark.parametrize("do_finalize", [True, False], ids=["finalized", "raw_fc2"])
def test_situ_b552_public_apis_match_quantized_reference(
    _situ_numeric_case: _SituCase,
    routing_api: Literal["logits", "packed", "unpacked"],
    linear_beta: float | None,
    do_finalize: bool,
):
    case = _situ_numeric_case
    reference, stats = _explicit_reference(case, beta=4.0, linear_beta=linear_beta)
    assert stats.max_abs_gate > 4.0
    assert stats.max_abs_up > 25.0
    assert stats.max_abs_up_minus_gate > 1.0

    result = _run_private_api(
        case,
        routing_api,
        linear_beta,
        do_finalize,
        tune_max_num_tokens=16,
    )
    actual = _finalize_result(result, case, do_finalize)
    _assert_quantized_reference_close(
        actual,
        reference,
        f"api={routing_api}, linear_beta={linear_beta}, finalize={do_finalize}",
    )


def test_situ_b552_nonzero_expert_parallel_offset_matches_local_reference(
    _situ_numeric_case: _SituCase,
):
    """Exercise a middle EP rank with local, nonlocal, and empty experts."""
    base_case = _situ_numeric_case
    local_expert_offset = base_case.shape.num_experts // 2
    local_num_experts = base_case.shape.num_experts // 2
    local_expert_stop = local_expert_offset + local_num_experts

    # Model a real second EP rank. Every token selects one expert from the
    # first shard and one from this shard; three local experts are hot while
    # the fourth remains empty. The shallow replacement reuses all quantized
    # weights without mutating the module-scoped numeric fixture.
    rows = torch.arange(
        base_case.shape.num_tokens, device=base_case.hidden_states.device
    )
    logits = torch.full_like(base_case.logits, -12.0)
    nonlocal_ids = rows % local_expert_offset
    local_ids = local_expert_offset + rows % (local_num_experts - 1)
    logits[rows, nonlocal_ids] = 6.0 + rows.float() / 100.0
    logits[rows, local_ids] = 4.0 - rows.float() / 100.0
    topk_logits, topk_ids = torch.topk(logits, base_case.shape.top_k, dim=-1)
    topk_ids = topk_ids.to(torch.int32)
    topk_weights = torch.softmax(topk_logits, dim=-1).to(torch.bfloat16)
    packed_topk = (topk_ids << 16) | topk_weights.view(torch.int16).to(torch.int32)
    case = replace(
        base_case,
        logits=logits,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        packed_topk=packed_topk,
    )

    selected = case.topk_ids.long()
    local_routes = (selected >= local_expert_offset) & (selected < local_expert_stop)
    assert bool(local_routes.any())
    assert bool((~local_routes).any())
    local_counts = torch.bincount(selected[local_routes], minlength=local_expert_stop)[
        local_expert_offset:local_expert_stop
    ]
    assert bool((local_counts > 0).any())
    assert bool((local_counts == 0).any())

    reference, _ = _explicit_reference(
        case,
        beta=4.0,
        linear_beta=25.0,
        local_expert_offset=local_expert_offset,
        local_num_experts=local_num_experts,
    )
    kwargs = _kernel_kwargs(
        case,
        linear_beta=25.0,
        do_finalize=True,
        tune_max_num_tokens=16,
        local_expert_offset=local_expert_offset,
        local_num_experts=local_num_experts,
    )
    for name in (
        "gemm1_weights",
        "gemm1_weights_scale",
        "gemm2_weights",
        "gemm2_weights_scale",
        "output1_scale_scalar",
        "output1_scale_gate_scalar",
        "output2_scale_scalar",
        "gemm1_alpha",
        "gemm1_beta",
    ):
        assert kwargs[name].shape[0] == local_num_experts

    logits_result = trtllm_fp4_block_scale_situ_moe(
        routing_logits=case.logits, **kwargs
    )
    routed_result = trtllm_fp4_block_scale_situ_routed_moe(
        topk_ids=(case.topk_ids, case.topk_weights), **kwargs
    )
    logits_output = _finalize_result(logits_result, case, do_finalize=True)
    routed_output = _finalize_result(routed_result, case, do_finalize=True)
    _assert_quantized_reference_close(logits_output, reference, "nonzero EP logits")
    _assert_quantized_reference_close(routed_output, reference, "nonzero EP pre-routed")
    torch.testing.assert_close(logits_output, routed_output, rtol=0.05, atol=0.05)


@pytest.mark.parametrize(
    "logits_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"]
)
def test_situ_b552_k3_deepseek_routing_matches_reference(
    _situ_k3_case: _SituCase,
    logits_dtype: torch.dtype,
):
    """Validate the exact K3 E896/K16 routing contract end to end.

    The private b552 ABI does not return expert IDs.  Selection is therefore
    proven by equivalence to a pre-routed launch using the exact PyTorch IDs:
    the logits launch must emit the same normalized weights and final output,
    while both outputs must also match the independently quantized PyTorch MoE
    reference.  Expert-specific random weights make a wrong selected-ID set
    observable even if its normalized routing weights happen to be similar.
    """
    case = _situ_k3_case
    logits, routing_bias, topk_ids, topk_weights = _k3_routing_reference(
        case.shape.num_tokens, case.hidden_states.device, logits_dtype
    )
    case.logits = logits
    case.topk_ids = topk_ids
    case.topk_weights = topk_weights
    case.packed_topk = (topk_ids << 16) | topk_weights.view(torch.int16).to(torch.int32)

    counts = torch.bincount(
        topk_ids.reshape(-1).long(), minlength=_K3_NUM_EXPERTS
    ).cpu()
    assert int((counts == 0).sum()) >= 850
    assert int(counts.max()) == case.shape.num_tokens
    assert int(counts[counts > 0].min()) == 1

    reference, stats = _explicit_reference(case, beta=4.0, linear_beta=25.0)
    assert stats.max_abs_up > 4.0
    assert stats.max_abs_gate > 2.0
    assert stats.max_abs_up_minus_gate > 1.0

    common_kwargs = _kernel_kwargs(
        case, linear_beta=25.0, do_finalize=False, tune_max_num_tokens=16
    )
    common_kwargs.update(
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        routing_method_type=RoutingMethodType.DeepSeekV3.value,
    )
    logits_result = trtllm_fp4_block_scale_situ_moe(
        routing_logits=logits,
        routing_bias=routing_bias,
        **{key: value for key, value in common_kwargs.items() if key != "routing_bias"},
    )
    routed_result = trtllm_fp4_block_scale_situ_routed_moe(
        topk_ids=(topk_ids, topk_weights),
        **common_kwargs,
    )

    # Native b552 stores routing weights as BF16 for both BF16 and FP32 logits.
    actual_weights = torch.sort(logits_result[1].float(), dim=-1).values
    reference_weights = torch.sort(topk_weights.float(), dim=-1).values
    torch.testing.assert_close(actual_weights, reference_weights, rtol=0.02, atol=0.002)
    torch.testing.assert_close(routed_result[1], topk_weights, rtol=0, atol=0)

    logits_output = _finalize_result(logits_result, case, do_finalize=False)
    routed_output = _finalize_result(routed_result, case, do_finalize=False)
    _assert_quantized_reference_close(
        logits_output, reference, f"K3 logits dtype={logits_dtype}"
    )
    _assert_quantized_reference_close(
        routed_output, reference, f"K3 pre-routed dtype={logits_dtype}"
    )
    torch.testing.assert_close(logits_output, routed_output, rtol=0.05, atol=0.05)


def test_situ_b552_grouped_deepseek_routing_matches_reference(
    _situ_grouped_case: _SituCase,
):
    """Cover the n_group>1/topk_group>1 DeepSeek routing pipeline."""
    case = _situ_grouped_case
    n_group = 4
    topk_group = 2
    logits, routing_bias, topk_ids, topk_weights = _grouped_deepseek_routing_reference(
        case.shape, n_group, topk_group
    )
    case.logits = logits
    case.topk_ids = topk_ids
    case.topk_weights = topk_weights
    case.packed_topk = (topk_ids << 16) | topk_weights.view(torch.int16).to(torch.int32)

    counts = torch.bincount(
        topk_ids.reshape(-1).long(), minlength=case.shape.num_experts
    ).cpu()
    assert int((counts == 0).sum()) > case.shape.num_experts // 2
    assert int((counts > 0).sum()) == 2 * n_group

    reference, stats = _explicit_reference(case, beta=4.0, linear_beta=25.0)
    assert stats.max_abs_up > 4.0
    assert stats.max_abs_gate > 2.0

    common_kwargs = _kernel_kwargs(
        case, linear_beta=25.0, do_finalize=False, tune_max_num_tokens=16
    )
    common_kwargs.update(
        routing_bias=routing_bias,
        n_group=n_group,
        topk_group=topk_group,
        routed_scaling_factor=1.0,
        routing_method_type=RoutingMethodType.DeepSeekV3.value,
    )
    logits_result = trtllm_fp4_block_scale_situ_moe(
        routing_logits=logits, **common_kwargs
    )
    routed_result = trtllm_fp4_block_scale_situ_routed_moe(
        topk_ids=(topk_ids, topk_weights), **common_kwargs
    )

    torch.testing.assert_close(
        torch.sort(logits_result[1].float(), dim=-1).values,
        torch.sort(topk_weights.float(), dim=-1).values,
        rtol=0.02,
        atol=0.002,
    )
    torch.testing.assert_close(routed_result[1], topk_weights, rtol=0, atol=0)

    logits_output = _finalize_result(logits_result, case, do_finalize=False)
    routed_output = _finalize_result(routed_result, case, do_finalize=False)
    _assert_quantized_reference_close(
        logits_output, reference, "grouped DeepSeek logits"
    )
    _assert_quantized_reference_close(
        routed_output, reference, "grouped DeepSeek pre-routed"
    )
    torch.testing.assert_close(logits_output, routed_output, rtol=0.05, atol=0.05)


@pytest.mark.parametrize(
    ("routing_api", "do_finalize"),
    [
        pytest.param("logits", True, id="logits_finalized"),
        pytest.param("unpacked", True, id="unpacked_finalized"),
        pytest.param("packed", False, id="packed_raw_fc2"),
    ],
)
def test_situ_b552_private_apis_cuda_graph_capture_replay(
    _situ_numeric_case: _SituCase,
    routing_api: Literal["logits", "packed", "unpacked"],
    do_finalize: bool,
):
    """Capture logits, unpacked, and packed/raw private entry points."""
    case = _situ_numeric_case
    kwargs = _kernel_kwargs(
        case,
        linear_beta=25.0,
        do_finalize=do_finalize,
        tune_max_num_tokens=16,
    )
    output = torch.empty(
        case.shape.num_tokens,
        case.shape.hidden_size,
        dtype=torch.bfloat16,
        device=case.hidden_states.device,
    )
    kwargs["output"] = output
    packed_before = case.packed_topk.clone()

    if routing_api == "logits":

        def launch() -> list[torch.Tensor]:
            return trtllm_fp4_block_scale_situ_moe(routing_logits=case.logits, **kwargs)

    elif routing_api == "packed":

        def launch() -> list[torch.Tensor]:
            return trtllm_fp4_block_scale_situ_routed_moe(
                topk_ids=case.packed_topk, **kwargs
            )

    else:

        def launch() -> list[torch.Tensor]:
            return trtllm_fp4_block_scale_situ_routed_moe(
                topk_ids=(case.topk_ids, case.topk_weights), **kwargs
            )

    # Finish autotuning and lazy CUDA initialization before capture.
    for _ in range(3):
        eager_result = launch()
        if do_finalize:
            assert eager_result[0].data_ptr() == output.data_ptr()
    torch.cuda.synchronize()
    eager_output = _finalize_result(eager_result, case, do_finalize).clone()
    assert torch.equal(case.packed_topk, packed_before)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_result = launch()
    if do_finalize:
        assert captured_result[0].data_ptr() == output.data_ptr()
    else:
        assert len(captured_result) == 3
    captured_ptrs = tuple(tensor.data_ptr() for tensor in captured_result)
    assert all(pointer != 0 for pointer in captured_ptrs)
    assert torch.equal(case.packed_topk, packed_before)

    replays: list[torch.Tensor] = []
    for sentinel in (float("nan"), 0.0):
        if do_finalize:
            output.fill_(sentinel)
        else:
            captured_result[0].fill_(sentinel)
            captured_result[1].fill_(sentinel)
            captured_result[2].fill_(-1)
        graph.replay()
        torch.cuda.synchronize()
        assert tuple(tensor.data_ptr() for tensor in captured_result) == captured_ptrs
        assert torch.equal(case.packed_topk, packed_before)
        replay = _finalize_result(captured_result, case, do_finalize)
        assert torch.isfinite(replay).all()
        replays.append(replay.clone())

    for replay in replays:
        torch.testing.assert_close(replay, eager_output, rtol=0.05, atol=0.05)
    torch.testing.assert_close(replays[0], replays[1], rtol=0.05, atol=0.05)


def _valid_private_tactics(op, shape: _Shape) -> list[tuple[int, int]]:
    tactics = op.trtllm_get_valid_moe_configs(
        DtypeTrtllmGen.MxE4m3,
        DtypeTrtllmGen.MxE2m1,
        Fp8QuantizationType.NoneFp8,
        shape.top_k,
        shape.hidden_size,
        shape.intermediate_size,
        shape.num_experts,
        ActivationType.Swiglu.value,
        True,
        WeightLayout.MajorK.value,
        False,
        shape.num_tokens,
        False,
    )
    return [tuple(int(value) for value in tactic) for tactic in tactics]


def _find_all_fc1_witnesses(
    metadata: _PrivateMetadataIndex,
) -> dict[str, tuple[_Shape, tuple[int, int]]]:
    op = get_trtllm_moe_situ_b552_module()
    witnesses: dict[str, tuple[_Shape, tuple[int, int]]] = {}

    # First try a compact shape family.  Larger K3-like dimensions are fallback
    # witnesses for any config whose validity predicate needs a wider K/N.
    dimensions = [
        (1024, 1024),
        (2048, 2048),
        (4096, 3072),
        (7168, 2048),
    ]
    # With 8 local experts and top-k 2 these produce average expert loads of
    # 8, 16, 32, 64, 128, and 256, covering the complete tile-N ladder.
    token_counts = [32, 64, 128, 256, 512, 1024]
    for hidden_size, intermediate_size in dimensions:
        for num_tokens in token_counts:
            shape = _Shape(num_tokens, hidden_size, intermediate_size)
            for tactic in _valid_private_tactics(op, shape):
                symbol = metadata.fc1_symbol(tactic)
                witnesses.setdefault(symbol, (shape, tactic))
            if witnesses.keys() == metadata.expected_fc1_symbols:
                return witnesses

    missing = metadata.expected_fc1_symbols - witnesses.keys()
    by_tile = defaultdict(int)
    for symbol in missing:
        by_tile[metadata._tile_n(symbol)] += 1
    raise AssertionError(
        f"only {len(witnesses)}/{_EXPECTED_FC1_CUBINS} private FC1 cubins have "
        f"a metadata-valid witness shape; missing by tile_N={dict(sorted(by_tile.items()))}"
    )


def _force_profile(case: _SituCase, tactic: tuple[int, int]) -> int:
    num_tokens = case.shape.num_tokens
    tune_max_num_tokens = max(_last_positive_power_of_2(num_tokens), 16)
    bucket_m = min(_last_positive_power_of_2(num_tokens), tune_max_num_tokens)
    profile_shapes = _moe_profile_shapes(
        {
            "hidden_size": case.shape.hidden_size,
            "packed_topk": case.topk_ids,
            "expert_weights": case.topk_weights,
            "hidden_states": case.hidden_states,
            "hidden_states_scale": case.hidden_states_scale,
        },
        num_tokens,
        bucket_m,
    )
    AutoTuner.get()._logged_file_hits.discard(_PRIVATE_LOG_KEY)
    _force_tactic_in_autotuner_cache(
        profile_shapes, list(tactic), custom_op=_PRIVATE_CUSTOM_OP
    )
    return tune_max_num_tokens


def test_situ_b552_force_every_private_fc1_tactic():
    artifact_root = _require_matched_private_blackwell()
    metadata = _PrivateMetadataIndex(artifact_root)
    witnesses = _find_all_fc1_witnesses(metadata)
    assert len(witnesses) == _EXPECTED_FC1_CUBINS
    assert witnesses.keys() == metadata.expected_fc1_symbols

    by_shape: dict[_Shape, list[tuple[str, tuple[int, int]]]] = defaultdict(list)
    for symbol, (shape, tactic) in witnesses.items():
        by_shape[shape].append((symbol, tactic))

    launched: set[str] = set()
    for shape in sorted(by_shape):
        case = _build_case(shape, seed=101)
        reference, stats = _explicit_reference(case, beta=4.0, linear_beta=25.0)
        assert stats.max_abs_gate > 4.0
        assert stats.max_abs_up > 25.0

        for symbol, tactic in sorted(by_shape[shape]):
            tune_max_num_tokens = _force_profile(case, tactic)
            result = _run_private_api(
                case,
                "unpacked",
                linear_beta=25.0,
                do_finalize=True,
                tune_max_num_tokens=tune_max_num_tokens,
            )
            assert _PRIVATE_LOG_KEY in AutoTuner.get()._logged_file_hits, (
                f"forced private tactic {tactic} ({symbol}) did not hit the "
                "SiTU-specific autotuner cache identity"
            )
            _assert_quantized_reference_close(
                result[0], reference, f"FC1={symbol}, tactic={tactic}, shape={shape}"
            )
            launched.add(symbol)

        del case, reference
        gc.collect()
        torch.cuda.empty_cache()

    assert launched == metadata.expected_fc1_symbols
    assert len(launched) == _EXPECTED_FC1_CUBINS
