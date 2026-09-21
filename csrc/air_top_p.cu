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
// Keep AIR kernels in their own compilation unit. Compiling them alongside
// sampling.cuh changes static shared-memory layout and adds CUDA Graph latency.
#include <flashinfer/air_top_p.cuh>

#include "sampling_utils.h"
#include "tvm_ffi_utils.h"

using namespace flashinfer;
using tvm::ffi::Optional;

cudaError_t air_top_p_renorm(float* probs, float* output, float* top_p_arr, uint32_t batch_size,
                             float top_p_val, uint32_t vocab_size, void* workspace,
                             bool is_deterministic, cudaStream_t stream) {
  if (is_deterministic) {
    return sampling::air_top_p::AirTopPRenormProb<true, float>(
        probs, output, top_p_arr, batch_size, top_p_val, vocab_size, workspace, stream);
  }
  return sampling::air_top_p::AirTopPRenormProb<false, float>(
      probs, output, top_p_arr, batch_size, top_p_val, vocab_size, workspace, stream);
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
