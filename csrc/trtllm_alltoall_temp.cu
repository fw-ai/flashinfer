// Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES
// Licensed under the Apache License, Version 2.0

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "fireworks/csrc/checks.h"

#include <tensorrt_llm/common/workspace.h>
#include <tensorrt_llm/kernels/fusedMoeCommKernels.h>
#include <tensorrt_llm/kernels/moePrepareKernels.h>
#include <tensorrt_llm/runtime/torchUtils.h>
#include <tensorrt_llm/thop/thUtils.h>
#include <torch/extension.h>

using namespace flashinfer::trtllm_alltoall_temp;

namespace flashinfer {
namespace trtllm_alltoall_temp {

static inline void setMoeCommFieldInfo(
    tensorrt_llm::kernels::MoeCommFieldInfo& fieldInfo, const at::Tensor& tensor) {
  TORCH_CHECK(tensor.dim() == 2, "tensor must be a 2D tensor");
  int eltSize = tensor.dtype().itemsize();
  fieldInfo.fillFieldInfo(
      static_cast<uint8_t*>(tensor.data_ptr()),
      eltSize,
      tensor.size(1),
      tensor.stride(0),
      torch_ext::convert_torch_dtype(tensor.scalar_type()));
}

static c10::List<at::Tensor> moe_comm(
    c10::List<at::Tensor> inputs,
    at::Tensor sendRankCumSum,
    at::Tensor sendIndiceTensor,
    at::Tensor recvRankCumSum,
    at::Tensor recvIndiceTensor,
    at::Tensor allWorkspaces,
    int64_t outputAllocationCount,
    int64_t epRank,
    int64_t epSize,
    c10::optional<c10::List<bool>> needZeroOutput = c10::nullopt,
    c10::optional<bool> useLowPrecision = c10::nullopt) {
  CHECK_INPUT(sendRankCumSum, at::kInt);
  CHECK_INPUT(sendIndiceTensor, at::kInt);
  CHECK_INPUT(recvRankCumSum, at::kInt);
  CHECK_INPUT(recvIndiceTensor, at::kInt);

  TORCH_CHECK(sendRankCumSum.dim() == 1);
  TORCH_CHECK(sendIndiceTensor.dim() == 1);
  TORCH_CHECK(recvRankCumSum.dim() == 1);
  TORCH_CHECK(recvIndiceTensor.dim() == 1);
  TORCH_CHECK(allWorkspaces.dim() == 2);
  TORCH_CHECK(sendRankCumSum.size(0) == epSize);
  TORCH_CHECK(recvRankCumSum.size(0) == epSize);
  TORCH_CHECK(allWorkspaces.size(0) == epSize);
  TORCH_CHECK(epRank >= 0 && epRank < epSize);
  TORCH_CHECK(!needZeroOutput.has_value() || needZeroOutput->size() == inputs.size());

  c10::List<at::Tensor> outputs;

  MoeEpWorldInfo epWorldInfo{static_cast<int>(epSize), static_cast<int>(epRank)};
  tensorrt_llm::kernels::FusedMoeWorldInfo worldInfo{epWorldInfo};

  SendRecvIndices sendIndices, recvIndices;
  sendIndices.rankCountCumSum = sendRankCumSum.data_ptr<int>();
  sendIndices.rankLocalIndices = sendIndiceTensor.data_ptr<int>();
  recvIndices.rankCountCumSum = recvRankCumSum.data_ptr<int>();
  recvIndices.rankLocalIndices = recvIndiceTensor.data_ptr<int>();

  int fieldCount = inputs.size();
  TORCH_CHECK(fieldCount <= tensorrt_llm::kernels::MOE_COMM_FIELD_MAX_COUNT);
  FusedMoeFieldInfo sendFieldInfo, recvFieldInfo;
  sendFieldInfo.isBasicInterleaved = false;
  recvFieldInfo.isBasicInterleaved = false;
  sendFieldInfo.fieldCount = fieldCount;
  recvFieldInfo.fieldCount = fieldCount;
  sendFieldInfo.expertScales = nullptr;
  recvFieldInfo.expertScales = nullptr;
  sendFieldInfo.tokenSelectedSlots = nullptr;
  recvFieldInfo.tokenSelectedSlots = nullptr;

  for (int i = 0; i < fieldCount; i++) {
    const at::Tensor& t = inputs[i];
    setMoeCommFieldInfo(sendFieldInfo.fieldsInfo[i], t);
    if (needZeroOutput.has_value() && (*needZeroOutput)[i]) {
      outputs.push_back(at::zeros({outputAllocationCount, t.size(1)}, t.options()));
    } else {
      outputs.push_back(at::empty({outputAllocationCount, t.size(1)}, t.options()));
    }
    setMoeCommFieldInfo(recvFieldInfo.fieldsInfo[i], outputs[i]);
  }
  sendFieldInfo.fillFieldPlacementInfo(0, false);
  recvFieldInfo.fillFieldPlacementInfo(0, false);

  FusedMoeCommKernelParam params{};
  params.worldInfo = worldInfo;
  params.sendIndices = sendIndices;
  params.recvIndices = recvIndices;
  params.sendFieldInfo = sendFieldInfo;
  params.recvFieldInfo = recvFieldInfo;

  bool useLowPrecisionVal = useLowPrecision.value_or(false);
  params.isLowPrecision = useLowPrecisionVal;
  params.sendFieldInfo.fillMetaInfo(&(params.sendCommMeta), params.expertParallelInfo.topK, false, false, useLowPrecisionVal);
  params.recvFieldInfo.fillMetaInfo(&(params.recvCommMeta), params.expertParallelInfo.topK, false, false, useLowPrecisionVal);

  FusedMoeWorkspace fusedMoeWorkspace;
  tensorrt_llm::kernels::constructWorkspace(
      &fusedMoeWorkspace, allWorkspaces.data_ptr<uint64_t>(), allWorkspaces.stride(0), epSize);

  auto stream = at::cuda::getCurrentCUDAStream();
  tensorrt_llm::kernels::moeAllToAll(params, fusedMoeWorkspace, stream);
  return outputs;
}

static int64_t get_moe_commworkspace_size_per_rank(int64_t epSize) {
  int epSize32 = static_cast<int>(epSize);
  return tensorrt_llm::kernels::getFusedMoeCommWorkspaceSize(epSize32);
}

static void set_moe_max_usable_sm_count(int64_t maxSmCount) {
  tensorrt_llm::kernels::setMaxUsableSmCount(maxSmCount);
}

static int64_t get_moe_prepare_workspace_size_per_rank(int64_t epSize) {
  int epSize32 = static_cast<int>(epSize);
  return tensorrt_llm::kernels::moe_prepare::getMoePrepareWorkspaceSize(epSize32);
}

static void moe_initialize_workspace(at::Tensor allWorkspaces, int64_t epRank, int64_t epSize) {
  TORCH_CHECK(allWorkspaces.dim() == 2);
  TORCH_CHECK(epRank >= 0 && epRank < epSize);
  MoeEpWorldInfo epWorldInfo{static_cast<int>(epSize), static_cast<int>(epRank)};
  tensorrt_llm::kernels::FusedMoeWorldInfo worldInfo{epWorldInfo};
  FusedMoeWorkspace fusedMoeWorkspace;
  tensorrt_llm::kernels::constructWorkspace(
      &fusedMoeWorkspace, allWorkspaces.data_ptr<uint64_t>(), allWorkspaces.stride(0), epSize);
  tensorrt_llm::kernels::initializeFusedMoeLocalWorkspace(&fusedMoeWorkspace, worldInfo);
}

static std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, c10::optional<at::Tensor>>
mnnvl_moe_alltoallv_prepare_without_allgather(
    at::Tensor expertsIds,
    c10::optional<at::Tensor> expertsStatics,
    at::Tensor allWorkspaces,
    int64_t maxTokenCountPerRank,
    int64_t epRank,
    int64_t epSize,
    int64_t expertCount,
    int64_t slotCount,
    int64_t topK) {
  CHECK_INPUT(expertsIds, at::kInt);
  TORCH_CHECK(expertCount % 4 == 0);
  TORCH_CHECK(slotCount % 4 == 0);
  TORCH_CHECK(expertCount + 1 <= 512);

  int64_t maxSendRanksPerToken = std::max<int64_t>(epSize, topK);
  int64_t tokenCount = expertsIds.size(0);

  at::Tensor preparedLocalExpertIds = at::empty({maxTokenCountPerRank * epSize, topK}, expertsIds.options().dtype(at::kInt));

  at::Tensor sendRankCountCumSum = at::empty({epSize}, expertsIds.options().dtype(at::kInt));
  at::Tensor recvRankCountCumSum = at::empty({epSize}, expertsIds.options().dtype(at::kInt));

  at::Tensor gatherRecvRankIndices = at::empty({maxTokenCountPerRank * epSize}, expertsIds.options().dtype(at::kInt));
  at::Tensor recvRankIndices = at::empty({maxTokenCountPerRank * epSize}, expertsIds.options().dtype(at::kInt));

  at::Tensor gatherBackwardRecvRankIndices = at::empty({maxTokenCountPerRank * maxSendRanksPerToken}, expertsIds.options().dtype(at::kInt));
  at::Tensor backwardRecvRankIndices = at::empty({maxTokenCountPerRank * maxSendRanksPerToken}, expertsIds.options().dtype(at::kInt));

  at::Tensor gatherSendRankIndices = at::empty({maxTokenCountPerRank * maxSendRanksPerToken}, expertsIds.options().dtype(at::kInt));
  at::Tensor sendRankIndices = at::empty({maxTokenCountPerRank * maxSendRanksPerToken}, expertsIds.options().dtype(at::kInt));

  int* localExpertStaticsPtr = nullptr;
  int* gatheredExpertStaticsPtr = nullptr;
  c10::optional<at::Tensor> gatheredExpertStatics;
  if (expertsStatics.has_value()) {
    localExpertStaticsPtr = expertsStatics->data_ptr<int>();
    gatheredExpertStatics = at::empty({epSize, expertCount}, expertsIds.options().dtype(at::kInt));
    gatheredExpertStaticsPtr = gatheredExpertStatics->data_ptr<int>();
  }

  tensorrt_llm::kernels::moe_prepare::MoeCommWorkspace workspace;
  workspace.workspacePtr = allWorkspaces.data_ptr<uint64_t>();
  workspace.rankStrideInU64 = allWorkspaces.stride(0);

  auto stream = at::cuda::getCurrentCUDAStream();
  tensorrt_llm::kernels::moe_prepare::computeCountAndIndice(
      expertsIds.data_ptr<int>(),
      sendRankCountCumSum.data_ptr<int>(),
      recvRankCountCumSum.data_ptr<int>(),
      sendRankIndices.data_ptr<int>(),
      backwardRecvRankIndices.data_ptr<int>(),
      recvRankIndices.data_ptr<int>(),
      localExpertStaticsPtr,
      gatheredExpertStaticsPtr,
      workspace,
      static_cast<int>(tokenCount),
      static_cast<int>(maxTokenCountPerRank),
      static_cast<int>(topK),
      static_cast<int>(slotCount),
      static_cast<int>(expertCount),
      static_cast<int>(epRank),
      static_cast<int>(epSize),
      stream);

  tensorrt_llm::kernels::moe_prepare::computeCumsum(
      sendRankCountCumSum.data_ptr<int>(),
      recvRankCountCumSum.data_ptr<int>(),
      static_cast<int>(epRank),
      static_cast<int>(epSize),
      stream);

  tensorrt_llm::kernels::moe_prepare::moveIndice(
      sendRankCountCumSum.data_ptr<int>(),
      recvRankCountCumSum.data_ptr<int>(),
      sendRankIndices.data_ptr<int>(),
      gatherSendRankIndices.data_ptr<int>(),
      backwardRecvRankIndices.data_ptr<int>(),
      gatherBackwardRecvRankIndices.data_ptr<int>(),
      recvRankIndices.data_ptr<int>(),
      gatherRecvRankIndices.data_ptr<int>(),
      static_cast<int>(epRank),
      static_cast<int>(epSize),
      static_cast<int>(maxTokenCountPerRank),
      stream);

  return std::make_tuple(
      sendRankCountCumSum,
      gatherSendRankIndices,
      recvRankCountCumSum,
      gatherRecvRankIndices,
      gatherBackwardRecvRankIndices,
      gatheredExpertStatics);
}

static void memset_expert_ids(
    at::Tensor expertsIds,
    at::Tensor recvRankCountCumSum,
    int64_t maxTokenCountPerRank,
    int64_t topK,
    int64_t slotCount,
    int64_t epSize) {
  CHECK_INPUT(expertsIds, at::kInt);
  TORCH_CHECK(expertsIds.dim() == 2);
  TORCH_CHECK(expertsIds.size(0) == maxTokenCountPerRank * epSize);
  TORCH_CHECK(expertsIds.size(1) == topK);

  CHECK_INPUT(recvRankCountCumSum, at::kInt);
  TORCH_CHECK(recvRankCountCumSum.dim() == 1);
  TORCH_CHECK(recvRankCountCumSum.size(0) == epSize);

  auto stream = at::cuda::getCurrentCUDAStream();
  tensorrt_llm::kernels::moe_prepare::memsetExpertIds(
      expertsIds.data_ptr<int>(),
      recvRankCountCumSum.data_ptr<int>(),
      static_cast<int>(maxTokenCountPerRank),
      static_cast<int>(topK),
      static_cast<int>(slotCount),
      static_cast<int>(epSize),
      stream);
}

} // namespace
} // namespace fireworks
