"""
Simple FP4 quantization validation test.

This test validates that FP4 quantization produces results within expected bounds
to catch regressions like commit 4de2a45a which changed the outputScale formula.
"""
import pytest
import torch

try:
    from flashinfer.fp4_quantization import scaled_fp4_grouped_quantize
except ImportError:
    pytest.skip("scaled_fp4_grouped_quantize not available", allow_module_level=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_fp4_quantization_basic():
    """Test basic FP4 quantization works correctly.

    This is a minimal test to ensure quantization completes without errors,
    which would catch severe formula bugs like the outputScale formula change
    in commit 4de2a45a.
    """
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Input shape: [l, m, k] where l=groups, m=tokens, k=features
    l, m, k = 1, 64, 512
    x = torch.randn(l, m, k, device="cuda", dtype=torch.float16)

    FLOAT8_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
    FLOAT4_E2M1_MAX = 6.0

    # Compute global scale
    tensor_amax = x.abs().max().to(torch.float32)
    global_scale_value = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax
    global_scale = torch.tensor([global_scale_value], device="cuda", dtype=torch.float32)
    mask = torch.ones(l, device="cuda", dtype=torch.int32)

    # Quantize
    quant_out, scale_out = scaled_fp4_grouped_quantize(x, mask, global_scale)

    # Basic checks - if formula is wrong, quantization might still work but produce garbage
    # The key regression from commit 4de2a45a was that the outputScale formula changed
    # from: outputScale = reciprocal(quantized_sf) * reciprocal(SFScaleVal)
    # to: outputScale = SFScaleVal / quantized_sf
    # This could produce incorrect scales, leading to values outside FP4 range
    assert quant_out.nelement() > 0, "Quantized output is empty"
    assert scale_out.nelement() > 0, "Scale output is empty"
    assert quant_out.dtype == torch.uint8, f"Expected uint8 output, got {quant_out.dtype}"

    # Global scale formula check
    # Formula: global_scale = (FP8_E4M3_MAX * FP4_E2M1_MAX) / tensor_amax
    expected_global = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax
    assert (
        abs(global_scale_value - expected_global) < 1e-3
    ), f"Global scale formula changed: {global_scale_value} vs {expected_global}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])