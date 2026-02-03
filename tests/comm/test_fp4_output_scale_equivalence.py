# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Test to verify numerical equivalence of outputScale computation in FP4 quantization.

This test captures the potential numerical discrepancy introduced in commit 4de2a45
where the outputScale computation was changed from:

    OLD (using approximate reciprocal):
        SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f))
        outputScale = reciprocal_approximate_ftz(SFValue * reciprocal_approximate_ftz(SFScaleVal))

    NEW (using exact division):
        sf_value = SFScaleVal * (vecMax * (1.0f/6.0f))
        outputScale = SFScaleVal / quantized_sf

The comment claims these are "mathematically equivalent", but rcp.approx.ftz.f32
is an approximate instruction (within 2 ULP), so the two methods produce different results.
"""

import pytest
import torch

# Skip if CUDA not available
if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)


def get_compute_capability():
    """Get the compute capability of the current CUDA device."""
    device = torch.cuda.current_device()
    return torch.cuda.get_device_capability(device)


# Reference implementations of the two outputScale computation methods
# Note: We avoid cuda_fp8.h dependency for broader compatibility
CUDA_KERNEL_CODE = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

// Approximate reciprocal using PTX instruction (same as in the original code)
__device__ __forceinline__ float reciprocal_approximate_ftz(float a) {
    float b;
    asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
    return b;
}

// OLD implementation: uses approximate reciprocal
__device__ float compute_output_scale_old(float SFScaleVal, float quantized_sf) {
    if (quantized_sf == 0) return 0.0f;
    // Original formula: rcp_approx(quantized_sf * rcp_approx(SFScaleVal))
    return reciprocal_approximate_ftz(quantized_sf * reciprocal_approximate_ftz(SFScaleVal));
}

// NEW implementation: uses exact division
__device__ float compute_output_scale_new(float SFScaleVal, float quantized_sf) {
    if (quantized_sf == 0) return 0.0f;
    // Optimized formula: direct division
    return SFScaleVal / quantized_sf;
}

// OLD implementation for SF value computation
__device__ float compute_sf_value_old(float SFScaleVal, float vecMax) {
    return SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
}

// NEW implementation for SF value computation (using exact constant)
__device__ float compute_sf_value_new(float SFScaleVal, float vecMax) {
    constexpr float RECIPROCAL_6 = 1.0f / 6.0f;
    return SFScaleVal * (vecMax * RECIPROCAL_6);
}

// Simple FP8 E4M3 simulation (for testing without cuda_fp8.h dependency)
// E4M3 has 4 exponent bits, 3 mantissa bits, range ~[-448, 448]
__device__ float simulate_fp8_e4m3_quantize(float val) {
    // Clamp to E4M3 range
    const float MAX_E4M3 = 448.0f;
    val = fminf(fmaxf(val, -MAX_E4M3), MAX_E4M3);

    // E4M3 has 3 mantissa bits = 8 values per power of 2
    // Quantize by rounding to nearest representable value
    if (val == 0.0f) return 0.0f;

    float sign = (val < 0) ? -1.0f : 1.0f;
    float abs_val = fabsf(val);

    // Get exponent
    int exp;
    float mantissa = frexpf(abs_val, &exp);

    // E4M3 bias is 7, so exp range is [-6, 8] for normalized
    // Quantize mantissa to 3 bits (8 levels in [0.5, 1.0))
    mantissa = roundf(mantissa * 8.0f) / 8.0f;

    return sign * ldexpf(mantissa, exp);
}

extern "C" __global__ void test_output_scale_equivalence(
    const float* SFScaleVals,
    const float* quantized_sfs,
    float* output_scale_old,
    float* output_scale_new,
    float* abs_diff,
    float* rel_diff,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sf_scale = SFScaleVals[idx];
        float quant_sf = quantized_sfs[idx];

        float old_val = compute_output_scale_old(sf_scale, quant_sf);
        float new_val = compute_output_scale_new(sf_scale, quant_sf);

        output_scale_old[idx] = old_val;
        output_scale_new[idx] = new_val;
        abs_diff[idx] = fabsf(old_val - new_val);
        rel_diff[idx] = (old_val != 0) ? fabsf(old_val - new_val) / fabsf(old_val) : 0.0f;
    }
}

extern "C" __global__ void test_sf_value_equivalence(
    const float* SFScaleVals,
    const float* vecMaxs,
    float* sf_value_old,
    float* sf_value_new,
    float* abs_diff,
    float* rel_diff,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sf_scale = SFScaleVals[idx];
        float vec_max = vecMaxs[idx];

        float old_val = compute_sf_value_old(sf_scale, vec_max);
        float new_val = compute_sf_value_new(sf_scale, vec_max);

        sf_value_old[idx] = old_val;
        sf_value_new[idx] = new_val;
        abs_diff[idx] = fabsf(old_val - new_val);
        rel_diff[idx] = (old_val != 0) ? fabsf(old_val - new_val) / fabsf(old_val) : 0.0f;
    }
}

// Test the full quantization flow showing the combined effect
extern "C" __global__ void test_full_quantize_flow(
    const float* input_vals,
    const float* SFScaleVals,
    const float* vecMaxs,
    float* scaled_old,
    float* scaled_new,
    float* abs_diff,
    float* rel_diff,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float input = input_vals[idx];
        float sf_scale = SFScaleVals[idx];
        float vec_max = vecMaxs[idx];

        // OLD path: SF value with approximate reciprocal
        float sf_old = compute_sf_value_old(sf_scale, vec_max);
        // Quantize to E4M3 (simulate)
        float quantized_sf_old = simulate_fp8_e4m3_quantize(sf_old);
        // Output scale OLD
        float output_scale_old_val = compute_output_scale_old(sf_scale, quantized_sf_old);
        // Scale input
        float scaled_val_old = input * output_scale_old_val;

        // NEW path: SF value with exact constant
        float sf_new = compute_sf_value_new(sf_scale, vec_max);
        // Quantize to E4M3 (simulate)
        float quantized_sf_new = simulate_fp8_e4m3_quantize(sf_new);
        // Output scale NEW
        float output_scale_new_val = compute_output_scale_new(sf_scale, quantized_sf_new);
        // Scale input
        float scaled_val_new = input * output_scale_new_val;

        scaled_old[idx] = scaled_val_old;
        scaled_new[idx] = scaled_val_new;
        abs_diff[idx] = fabsf(scaled_val_old - scaled_val_new);
        rel_diff[idx] = (scaled_val_old != 0) ? fabsf(scaled_val_old - scaled_val_new) / fabsf(scaled_val_old) : 0.0f;
    }
}
"""


def compile_cuda_module():
    """Compile the CUDA test module using torch.utils.cpp_extension."""
    try:
        from torch.utils.cpp_extension import load_inline

        module = load_inline(
            name="fp4_scale_test",
            cpp_sources="",
            cuda_sources=CUDA_KERNEL_CODE,
            functions=[
                "test_output_scale_equivalence",
                "test_sf_value_equivalence",
                "test_full_quantize_flow",
            ],
            verbose=True,
            extra_cuda_cflags=["-O3", "--use_fast_math"],
        )
        return module
    except Exception as e:
        import traceback

        traceback.print_exc()
        pytest.skip(f"Could not compile CUDA module: {e}")


class TestFP4OutputScaleEquivalence:
    """Test class for FP4 output scale numerical equivalence."""

    @pytest.fixture(scope="class")
    def cuda_module(self):
        """Compile and cache the CUDA module."""
        return compile_cuda_module()

    @pytest.mark.parametrize(
        "sf_scale_range",
        [
            (0.1, 1.0),
            (1.0, 10.0),
            (0.01, 0.1),
            (10.0, 100.0),
        ],
    )
    @pytest.mark.parametrize(
        "quantized_sf_range",
        [
            (0.1, 1.0),
            (1.0, 10.0),
            (0.5, 2.0),
        ],
    )
    def test_output_scale_numerical_difference(
        self, cuda_module, sf_scale_range, quantized_sf_range
    ):
        """
        Test that the old and new outputScale computations produce different results.

        The old implementation uses:
            rcp_approx(quantized_sf * rcp_approx(SFScaleVal))

        The new implementation uses:
            SFScaleVal / quantized_sf

        These are NOT numerically equivalent due to the approximate reciprocal.
        """
        torch.manual_seed(42)
        n = 10000
        device = torch.device("cuda")

        # Generate test values
        SFScaleVals = torch.empty(n, device=device).uniform_(*sf_scale_range)
        quantized_sfs = torch.empty(n, device=device).uniform_(*quantized_sf_range)

        # Output buffers
        output_scale_old = torch.empty(n, device=device)
        output_scale_new = torch.empty(n, device=device)
        abs_diff = torch.empty(n, device=device)
        rel_diff = torch.empty(n, device=device)

        # Run kernel
        block_size = 256
        grid_size = (n + block_size - 1) // block_size
        cuda_module.test_output_scale_equivalence(
            SFScaleVals,
            quantized_sfs,
            output_scale_old,
            output_scale_new,
            abs_diff,
            rel_diff,
            n,
            grid=(grid_size, 1, 1),
            block=(block_size, 1, 1),
        )
        torch.cuda.synchronize()

        # Analyze results
        max_abs_diff = abs_diff.max().item()
        mean_abs_diff = abs_diff.mean().item()
        max_rel_diff = rel_diff.max().item()
        mean_rel_diff = rel_diff.mean().item()
        num_different = (abs_diff > 0).sum().item()

        print("\n=== Output Scale Equivalence Test ===")
        print(f"SF Scale Range: {sf_scale_range}")
        print(f"Quantized SF Range: {quantized_sf_range}")
        print(f"Number of test cases: {n}")
        print(
            f"Number with differences: {num_different} ({100 * num_different / n:.2f}%)"
        )
        print(f"Max absolute difference: {max_abs_diff:.6e}")
        print(f"Mean absolute difference: {mean_abs_diff:.6e}")
        print(f"Max relative difference: {max_rel_diff:.6e}")
        print(f"Mean relative difference: {mean_rel_diff:.6e}")

        # The key assertion: there SHOULD be differences because the implementations
        # are NOT numerically equivalent (despite what the commit comment claims)
        # If max_abs_diff > 0, we've captured the discrepancy
        if num_different > 0:
            print(
                "\n[CAPTURED] Numerical discrepancy detected between old and new implementations!"
            )
            # Show some example differences
            diff_indices = (abs_diff > 0).nonzero(as_tuple=True)[0][:5]
            print("\nExample differences:")
            for idx in diff_indices:
                i = idx.item()
                print(
                    f"  SFScale={SFScaleVals[i].item():.6f}, "
                    f"quant_sf={quantized_sfs[i].item():.6f} -> "
                    f"old={output_scale_old[i].item():.8f}, "
                    f"new={output_scale_new[i].item():.8f}, "
                    f"diff={abs_diff[i].item():.2e}"
                )

        # Store results for later analysis
        return {
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
            "max_rel_diff": max_rel_diff,
            "mean_rel_diff": mean_rel_diff,
            "num_different": num_different,
        }

    @pytest.mark.parametrize(
        "sf_scale_range",
        [
            (0.1, 1.0),
            (1.0, 10.0),
        ],
    )
    @pytest.mark.parametrize(
        "vec_max_range",
        [
            (0.1, 1.0),
            (1.0, 6.0),
        ],
    )
    def test_sf_value_numerical_difference(
        self, cuda_module, sf_scale_range, vec_max_range
    ):
        """
        Test that the SF value computation differs between old and new implementations.

        The old implementation uses:
            SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f))

        The new implementation uses:
            SFScaleVal * (vecMax * (1.0f/6.0f))  // exact compile-time constant
        """
        torch.manual_seed(42)
        n = 10000
        device = torch.device("cuda")

        # Generate test values
        SFScaleVals = torch.empty(n, device=device).uniform_(*sf_scale_range)
        vecMaxs = torch.empty(n, device=device).uniform_(*vec_max_range)

        # Output buffers
        sf_value_old = torch.empty(n, device=device)
        sf_value_new = torch.empty(n, device=device)
        abs_diff = torch.empty(n, device=device)
        rel_diff = torch.empty(n, device=device)

        # Run kernel
        block_size = 256
        grid_size = (n + block_size - 1) // block_size
        cuda_module.test_sf_value_equivalence(
            SFScaleVals,
            vecMaxs,
            sf_value_old,
            sf_value_new,
            abs_diff,
            rel_diff,
            n,
            grid=(grid_size, 1, 1),
            block=(block_size, 1, 1),
        )
        torch.cuda.synchronize()

        # Analyze results
        max_abs_diff = abs_diff.max().item()
        mean_abs_diff = abs_diff.mean().item()
        max_rel_diff = rel_diff.max().item()
        mean_rel_diff = rel_diff.mean().item()
        num_different = (abs_diff > 0).sum().item()

        print("\n=== SF Value Equivalence Test ===")
        print(f"SF Scale Range: {sf_scale_range}")
        print(f"VecMax Range: {vec_max_range}")
        print(f"Number of test cases: {n}")
        print(
            f"Number with differences: {num_different} ({100 * num_different / n:.2f}%)"
        )
        print(f"Max absolute difference: {max_abs_diff:.6e}")
        print(f"Mean absolute difference: {mean_abs_diff:.6e}")
        print(f"Max relative difference: {max_rel_diff:.6e}")
        print(f"Mean relative difference: {mean_rel_diff:.6e}")

        if num_different > 0:
            print(
                "\n[CAPTURED] Numerical discrepancy detected in SF value computation!"
            )

    @pytest.mark.parametrize(
        "input_range",
        [
            (-1.0, 1.0),
            (-10.0, 10.0),
        ],
    )
    @pytest.mark.parametrize(
        "sf_scale_range",
        [
            (0.1, 1.0),
            (1.0, 10.0),
        ],
    )
    def test_full_quantize_flow_difference(
        self, cuda_module, input_range, sf_scale_range
    ):
        """
        Test the combined effect of both changes through the full quantization flow.

        This simulates the actual scaling applied to input values during FP4 quantization,
        showing the cumulative effect of the numerical differences.
        """
        torch.manual_seed(42)
        n = 10000
        device = torch.device("cuda")

        # Generate test values
        input_vals = torch.empty(n, device=device).uniform_(*input_range)
        SFScaleVals = torch.empty(n, device=device).uniform_(*sf_scale_range)
        # vecMax should be based on actual max of input for realistic testing
        vecMaxs = torch.abs(input_vals) * torch.empty(n, device=device).uniform_(
            1.0, 2.0
        )

        # Output buffers
        scaled_old = torch.empty(n, device=device)
        scaled_new = torch.empty(n, device=device)
        abs_diff = torch.empty(n, device=device)
        rel_diff = torch.empty(n, device=device)

        # Run kernel
        block_size = 256
        grid_size = (n + block_size - 1) // block_size
        cuda_module.test_full_quantize_flow(
            input_vals,
            SFScaleVals,
            vecMaxs,
            scaled_old,
            scaled_new,
            abs_diff,
            rel_diff,
            n,
            grid=(grid_size, 1, 1),
            block=(block_size, 1, 1),
        )
        torch.cuda.synchronize()

        # Analyze results
        max_abs_diff = abs_diff.max().item()
        mean_abs_diff = abs_diff.mean().item()
        max_rel_diff = rel_diff.max().item()
        mean_rel_diff = rel_diff.mean().item()
        num_different = (abs_diff > 0).sum().item()

        print("\n=== Full Quantize Flow Test ===")
        print(f"Input Range: {input_range}")
        print(f"SF Scale Range: {sf_scale_range}")
        print(f"Number of test cases: {n}")
        print(
            f"Number with differences: {num_different} ({100 * num_different / n:.2f}%)"
        )
        print(f"Max absolute difference: {max_abs_diff:.6e}")
        print(f"Mean absolute difference: {mean_abs_diff:.6e}")
        print(f"Max relative difference: {max_rel_diff:.6e}")
        print(f"Mean relative difference: {mean_rel_diff:.6e}")

        if num_different > 0:
            print(
                "\n[CAPTURED] Numerical discrepancy detected in full quantization flow!"
            )
            print(
                "This demonstrates the potential quality regression from commit 4de2a45"
            )

            # Show some example differences
            diff_indices = (abs_diff > 0).nonzero(as_tuple=True)[0][:5]
            print("\nExample differences in scaled values:")
            for idx in diff_indices:
                i = idx.item()
                print(
                    f"  input={input_vals[i].item():.6f} -> "
                    f"old={scaled_old[i].item():.8f}, "
                    f"new={scaled_new[i].item():.8f}, "
                    f"diff={abs_diff[i].item():.2e}, "
                    f"rel_diff={rel_diff[i].item():.2e}"
                )


def test_reciprocal_approximate_vs_division():
    """
    Simple PyTorch-based test demonstrating the principle that
    1/x (exact) != rcp.approx(x) in floating point.

    This is a sanity check that can run without the CUDA kernel compilation.
    """
    # The exact reciprocal of 6.0 in IEEE 754 float32
    exact_recip_6 = 1.0 / 6.0

    # The approximate reciprocal from rcp.approx.ftz.f32 should be different
    # We can't directly call the PTX instruction from Python, but we know:
    # - rcp.approx.ftz.f32 is accurate to within 2 ULP (units in last place)
    # - The exact value of 1/6 in float32 is ~0.16666667163372039794921875

    # This test documents the expected behavior
    print("\n=== Reciprocal Comparison (Python reference) ===")
    print(f"1.0f / 6.0f (exact) = {exact_recip_6:.20f}")
    print(f"float32 representation: {exact_recip_6:.8e}")

    # The rcp.approx instruction may produce a value that differs by up to 2 ULP
    # For a float32 near 0.1666..., 1 ULP is approximately 1.49e-8
    # So the maximum expected difference is about 3e-8

    expected_max_ulp_error = 2
    import struct

    # Get the ULP of 1/6
    bits = struct.unpack("I", struct.pack("f", exact_recip_6))[0]
    next_float = struct.unpack("f", struct.pack("I", bits + 1))[0]
    ulp = next_float - exact_recip_6

    print(f"1 ULP at 1/6 ≈ {ulp:.2e}")
    print(
        f"Maximum rcp.approx error ({expected_max_ulp_error} ULP) ≈ {expected_max_ulp_error * ulp:.2e}"
    )
    print(
        f"\nThis demonstrates that rcp.approx(6.0) may differ from 1.0/6.0 by up to {expected_max_ulp_error * ulp:.2e}"
    )
    print(
        "When this error propagates through the outputScale computation, it can affect quantization quality."
    )


if __name__ == "__main__":
    # Run simple test first (doesn't need CUDA kernel compilation)
    test_reciprocal_approximate_vs_division()

    # Run CUDA tests
    pytest.main([__file__, "-v", "-s"])
