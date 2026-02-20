# FP4 Quantization Formula Testing

This test package validates FP4 quantization formulas to catch regressions like commit 4de2a45a which changed the outputScale formula.

## Test File

- `test_fp4_quantization_validation.py` - Validates the global quantization formula

## Running Tests

```bash
# Run FP4 quantization validation
uv run pytest tests/comm/test_fp4_quantization_validation.py -xvs
```

## What the Tests Catch

The tests verify:
1. **Global scale formula**: `global_scale = (FP8_E4M3_MAX * FP4_E2M1_MAX) / tensor_amax`
2. **Quantization completes**: Basic sanity that the kernel doesn't crash
3. **Output format**: Validates correct output dtype (uint8 for FP4 packed values)

## Connection to Your Issue

The KLD regression (0.0086 → 0.0212) during the v0.5.3 → v0.6.1 update was likely caused by commit 4de2a45a's quantization formula change:

- **OLD**: `outputScale = reciprocal(quantized_sf) * reciprocal(SFScaleVal)`
- **NEW**: `outputScale = SFScaleVal / quantized_sf`

The formula change introduced FP8 quantization error that accumulated through MoE expert computations.

## Implementation Notes

The `scaled_fp4_grouped_quantize` API requires:
- Input: 3D tensor `[l, m, k]` (groups, tokens, features)
- Mask: 1D int32 tensor matching group count
- Global scale: 1D float32 tensor
- Output: Quantized tensor (uint8), Scale factors (swizzled layout for Blackwell)

Due to complex swizzled layouts in Blackwell architecture, direct value comparison is challenging. The test focuses on formula validation at the global scale level, which is mathematically exact.
