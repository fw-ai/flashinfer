/*
 * Copyright (c) 2024 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <flashinfer/air_top_p.cuh>
#include <flashinfer/sampling.cuh>

#include "sampling_utils.h"
#include "tvm_ffi_utils.h"

using namespace flashinfer;

using tvm::ffi::Optional;

void top_p_renorm_probs(TensorView probs, TensorView renorm_probs,
                        Optional<TensorView> maybe_top_p_arr, double top_p_val,
                        bool is_deterministic, TensorView workspace) {
  CHECK_INPUT(probs);
  CHECK_DIM(2, probs);  // probs: (batch_size, vocab_size)
  unsigned int batch_size = probs.size(0);
  unsigned int vocab_size = probs.size(1);
  check_tensor_param(maybe_top_p_arr, probs);
  bool has_top_p_arr = maybe_top_p_arr.has_value();

  ffi::CUDADeviceGuard device_guard(probs.device().device_id);
  auto stream = get_stream(probs.device());

  float* top_p_arr_ptr =
      has_top_p_arr ? static_cast<float*>(maybe_top_p_arr.value().data_ptr()) : nullptr;

  cudaError_t status;
  // Fallback to ternary search for small vocab where radix precision is insufficient
  if (vocab_size < sampling::air_top_p::NUM_BUCKETS) {
    status = sampling::TopPRenormProb<float>(
        static_cast<float*>(probs.data_ptr()), static_cast<float*>(renorm_probs.data_ptr()),
        top_p_arr_ptr, batch_size, top_p_val, vocab_size, stream);
  } else if (is_deterministic) {
    status = sampling::air_top_p::AirTopPRenormProb<true, float>(
        static_cast<float*>(probs.data_ptr()), static_cast<float*>(renorm_probs.data_ptr()),
        top_p_arr_ptr, batch_size, top_p_val, vocab_size, workspace.data_ptr(), stream);
  } else {
    status = sampling::air_top_p::AirTopPRenormProb<false, float>(
        static_cast<float*>(probs.data_ptr()), static_cast<float*>(renorm_probs.data_ptr()),
        top_p_arr_ptr, batch_size, top_p_val, vocab_size, workspace.data_ptr(), stream);
  }
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "TopPRenormProb failed with error code " << cudaGetErrorString(status);
}

// The Python API handles small vocabularies with the existing renorm fallback.
void top_p_mask(TensorView probs, TensorView output, Optional<TensorView> maybe_top_p_arr,
                double top_p_val, bool is_deterministic, TensorView workspace) {
  CHECK_INPUT(probs);
  CHECK_INPUT(output);
  CHECK_INPUT(workspace);
  CHECK_DIM(2, probs);
  CHECK_DIM(2, output);
  CHECK_DIM(1, workspace);
  CHECK_DEVICE(probs, output);
  CHECK_DEVICE(probs, workspace);
  TVM_FFI_ICHECK(probs.dtype() == dl_float32);
  TVM_FFI_ICHECK(output.dtype().code == kDLBool && output.dtype().bits == 8);
  TVM_FFI_ICHECK(workspace.dtype().code == kDLUInt && workspace.dtype().bits == 8);
  TVM_FFI_ICHECK(output.size(0) == probs.size(0) && output.size(1) == probs.size(1));
  TVM_FFI_ICHECK(probs.size(0) > 0 && probs.size(0) <= 64);
  TVM_FFI_ICHECK(probs.size(1) >= sampling::air_top_p::NUM_BUCKETS);
  check_tensor_param(maybe_top_p_arr, probs);
  float* top_p_arr = nullptr;
  if (maybe_top_p_arr.has_value()) {
    auto param = maybe_top_p_arr.value();
    CHECK_INPUT(param);
    CHECK_DEVICE(probs, param);
    TVM_FFI_ICHECK(param.dtype() == dl_float32);
    top_p_arr = static_cast<float*>(param.data_ptr());
  }
  auto align256 = [](size_t n) { return (n + 255) / 256 * 256; };
  const auto batch = probs.size(0), vocab = probs.size(1);
  const size_t required =
      align256(sizeof(sampling::air_top_p::Counter<float>) * batch) +
      align256((is_deterministic ? sizeof(uint64_t) : sizeof(float)) * 2048 * batch) +
      align256(sizeof(sampling::air_top_p::IdxT) * 2048 * batch) +
      2 * align256(sizeof(float) * sampling::air_top_p::calcBufLen<float>(vocab) * batch);
  TVM_FFI_ICHECK(workspace.size(0) >= required);
  ffi::CUDADeviceGuard guard(probs.device().device_id);
  auto stream = get_stream(probs.device());
  cudaError_t status;
  if (is_deterministic) {
    status = sampling::air_top_p::AirTopPRenormProb<true, float, true>(
        static_cast<float*>(probs.data_ptr()), static_cast<bool*>(output.data_ptr()), top_p_arr,
        batch, top_p_val, vocab, workspace.data_ptr(), stream);
  } else {
    status = sampling::air_top_p::AirTopPRenormProb<false, float, true>(
        static_cast<float*>(probs.data_ptr()), static_cast<bool*>(output.data_ptr()), top_p_arr,
        batch, top_p_val, vocab, workspace.data_ptr(), stream);
  }
  TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
  status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
}

void top_k_renorm_probs(TensorView probs, TensorView renorm_probs,
                        Optional<TensorView> maybe_top_k_arr, int64_t top_k_val,
                        TensorView row_states_buffer) {
  CHECK_INPUT(probs);
  CHECK_INPUT(row_states_buffer);
  CHECK_DIM(2, probs);  // probs: (batch_size, vocab_size)
  unsigned int batch_size = probs.size(0);
  unsigned int vocab_size = probs.size(1);
  check_tensor_param(maybe_top_k_arr, probs);
  bool has_top_k_arr = maybe_top_k_arr.has_value();

  ffi::CUDADeviceGuard device_guard(probs.device().device_id);
  auto stream = get_stream(probs.device());

  cudaError_t status;
  auto dtype = probs.dtype();

  // Use radix-based top-k with dtype dispatch for FP32/FP16/BF16
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP32_FP16(dtype, c_type, [&] {
    status = sampling::RadixTopKRenormProbMultiCTA<c_type, int>(
        static_cast<c_type*>(probs.data_ptr()), static_cast<c_type*>(renorm_probs.data_ptr()),
        has_top_k_arr ? static_cast<int*>(maybe_top_k_arr.value().data_ptr()) : nullptr, batch_size,
        top_k_val, vocab_size, static_cast<sampling::RadixRowState*>(row_states_buffer.data_ptr()),
        stream);
    return true;
  });

  TVM_FFI_ICHECK(status == cudaSuccess)
      << "TopKRenormProb failed with error code " << cudaGetErrorString(status);
}

void top_k_mask_logits(TensorView logits, TensorView mask_logits,
                       Optional<TensorView> maybe_top_k_arr, int64_t top_k_val,
                       TensorView row_states_buffer) {
  CHECK_INPUT(logits);
  CHECK_INPUT(row_states_buffer);
  CHECK_DIM(2, logits);  // logits: (batch_size, vocab_size)
  unsigned int batch_size = logits.size(0);
  unsigned int vocab_size = logits.size(1);
  check_tensor_param(maybe_top_k_arr, logits);
  bool has_top_k_arr = maybe_top_k_arr.has_value();

  ffi::CUDADeviceGuard device_guard(logits.device().device_id);
  auto stream = get_stream(logits.device());

  cudaError_t status;
  auto dtype = logits.dtype();

  // Use radix-based top-k with auto-selection (single-CTA for small vocab, multi-CTA for large
  // vocab)
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP32_FP16(dtype, c_type, [&] {
    status = sampling::RadixTopKMaskLogitsMultiCTA<c_type, int>(
        static_cast<c_type*>(logits.data_ptr()), static_cast<c_type*>(mask_logits.data_ptr()),
        has_top_k_arr ? static_cast<int*>(maybe_top_k_arr.value().data_ptr()) : nullptr, batch_size,
        top_k_val, vocab_size, static_cast<sampling::RadixRowState*>(row_states_buffer.data_ptr()),
        stream);
    return true;
  });

  TVM_FFI_ICHECK(status == cudaSuccess)
      << "TopKMaskLogits failed with error code " << cudaGetErrorString(status);
}
