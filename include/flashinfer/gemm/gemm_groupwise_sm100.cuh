/*
 * Copyright (c) 2025 by FlashInfer team.
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
#ifndef FLASHINFER_GEMM_GROUPWISE_SM100_CUH_
#define FLASHINFER_GEMM_GROUPWISE_SM100_CUH_

#include <type_traits>
#include <typeinfo>
#include <cstdio>

#include "../allocator.h"
#include "../cutlass_utils.cuh"
#include "../utils.cuh"

namespace flashinfer {

namespace gemm {

using namespace cute;

template <int ScaleGranularityM, int ScaleGranularityN, int ScaleGranularityK, bool ScaleMajorK,
          int MmaSM, typename DTypeIn, typename DTypeOut>
cudaError_t CutlassGroupwiseScaledGEMMSM100(void* float_buffer, size_t float_buffer_size_in_bytes,
                                            DTypeIn* A_ptr, DTypeIn* B_ptr, float* SFA_ptr,
                                            float* SFB_ptr, DTypeOut* C_ptr, int m, int n, int k,
                                            int l, cudaStream_t stream) {
    // Note for our caes MmaSM = 1
//   printf("DTypeIn: %s\n", typeid(DTypeIn).name());
//   printf("DTypeOut: %s\n", typeid(DTypeOut).name());
//   printf("m: %d, n: %d, k: %d, l: %d\n", m, n, k, l);

  // x=(batch, In) * W^T=(In, Out)
  // x=(m,     k)  * W^T=(k,  n)
  // W=(n, k) row-major or W=(k, n) column-major
  using ElementA = DTypeIn;                   // Element type for A matrix operand
  using LayoutA = cutlass::layout::RowMajor;  // Layout type for A matrix operand
  constexpr int AlignmentA =
      128 / cutlass::sizeof_bits<ElementA>::value;  // Memory access granularity/alignment of A
  //                                                   // matrix in units of elements (up to 16 bytes)

  // // B matrix configuration
  using ElementB = DTypeIn;                      // Element type for B matrix operand
  using LayoutB = cutlass::layout::ColumnMajor;  // Layout type for B matrix operand
  constexpr int AlignmentB =
      128 / cutlass::sizeof_bits<ElementB>::value;  // Memory access granularity/alignment of A
                                                    // matrix in units of elements (up to 16 bytes)

  // // C/D matrix configuration
  using ElementC = DTypeOut;                  // Element type for C and D matrix operands
  using LayoutC = cutlass::layout::RowMajor;  // Layout type for C and D matrix operands
  constexpr int AlignmentC =
      128 / cutlass::sizeof_bits<ElementC>::value;  // Memory access granularity/alignment of A
                                                    // matrix in units of elements (up to 16 bytes)

  using ElementD = ElementC;
  using LayoutD = LayoutC;
  constexpr int AlignmentD = AlignmentC;

  // MMA type
  using ElementAccumulator = float;  // Element Accumulator will also be our scale factor type
  using ElementCompute = float;

  using MmaTileShape_MNK = Shape<cute::Int<128>, _16, _128>;
  using ClusterShape_MNK = Shape<int, int, _1>;

  // NOTE(Zihao):: UMMA::Major::MN, UMMA::Major::MN is the fastest configuration.

  // using ScaleConfig = std::conditional_t<
  //     ScaleMajorK,
  //     cutlass::detail::Sm100BlockwiseScaleConfig<ScaleGranularityM, ScaleGranularityN,
  //                                                ScaleGranularityK, UMMA::Major::K, UMMA::Major::K>,
  //     cutlass::detail::Sm100BlockwiseScaleConfig<ScaleGranularityM, ScaleGranularityN,
  //                                                ScaleGranularityK, UMMA::Major::MN,
  //                                                UMMA::Major::MN>>;
  using ScaleConfig = cutlass::detail::Sm1xxBlockwiseScaleConfig<128, 1, 128>;


//   printf("ScaleGranularityM: %d, ScaleGranularityN: %d, ScaleGranularityK: %d\n", ScaleGranularityM, ScaleGranularityN, ScaleGranularityK);
//   printf("UMMA::Major::K: %d, UMMA::Major::MN: %d\n", static_cast<int>(UMMA::Major::K), static_cast<int>(UMMA::Major::MN));
//   printf("ScaleMajorK: %d\n", ScaleMajorK);
  using LayoutSFA =
      decltype(ScaleConfig::deduce_layoutSFA());  // Layout type for SFA matrix operand
  using LayoutSFB =
      decltype(ScaleConfig::deduce_layoutSFB());  // Layout type for SFB matrix operand
//   using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
//       cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
//       /*TileShapeMNK=*/MmaTileShape_MNK, /*ClusterShape=*/ClusterShape_MNK,
//       /*EpilogueTileType=*/cutlass::epilogue::collective::EpilogueTileAuto,
//       ElementAccumulator, ElementCompute,
//       ElementC,/*GemmLayoutTagC=*/LayoutC, AlignmentC,
//       ElementD, /*GemmLayoutTagD=*/LayoutC, AlignmentD,
//       /*EpilogueScheduleType=*/cutlass::epilogue::collective::EpilogueScheduleAuto
//     >::CollectiveOp;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    cute::Shape<cute::_64, cute::_16, cute::_128>,
    cute::Shape<int, int, cute::_1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    void, cutlass::layout::ColumnMajor, 0,
    ElementD, cutlass::layout::ColumnMajor, 8,
    cutlass::epilogue::TmaWarpSpecialized1Sm,
    cutlass::epilogue::fusion::LinearCombination<
        ElementD,
        float,
        void,
        float
    >
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      /*ArchTag=*/cutlass::arch::Sm100, /*OpClass=*/cutlass::arch::OpClassTensorOp,
      ElementA,/*GemmLayoutA=*/cute::tuple<LayoutA, LayoutSFA>, 16,
      ElementB, /*GemmLayoutB=*/cute::tuple<LayoutB, LayoutSFB>, 16,
      ElementAccumulator,
      /*TileShapeMNK=*/MmaTileShape_MNK, ClusterShape_MNK,
      /*StageCountType=*/cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      /*KernelScheduleType=*/cutlass::gemm::KernelTmaWarpSpecializedBlockwise1SmSm100
    >::CollectiveOp;
  // using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
  //       cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
  //       ElementA, cutlass::layout::RowMajor, 16,
  //       ElementB, cutlass::layout::ColumnMajor, 16,
  //       float,
  //       cute::Shape<cute::_64, cute::_8, cute::_128>,
  //       cute::Shape<int, int, cute::_1>,
  //       cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
  //       cutlass::gemm::KernelTmaWarpSpecialized1SmSm100
  //   >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      /*ProblemShapeOrThreadblockMma_=*/Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      void>;
      // /*TileScheduler_=*/cutlass::gemm::StreamKScheduler>;  // Default to ClusterLaunchControl (CLC) based tile scheduler

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(n, k, l));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(m, k, l));
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(n, m, l));
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(n, m, l));

//   printf("stride_A type: %s\n", typeid(decltype(stride_A)).name());
//   printf("stride_B type: %s\n", typeid(decltype(stride_B)).name());
//   printf("stride_C type: %s\n", typeid(decltype(stride_C)).name());
//   printf("stride_D type: %s\n", typeid(decltype(stride_D)).name());

  auto layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(make_shape(n, m, k, l));
  auto layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(make_shape(n, m, k, l));

  // using TileSchedulerArguments = typename Gemm::GemmKernel::TileSchedulerArguments;
  // TileSchedulerArguments scheduler_args{};
  // scheduler_args.max_swizzle_size = 1;
  // scheduler_args.raster_order =
  //     cutlass::gemm::kernel::detail::RasterOrderOptions::Heuristic;

  typename Gemm::Arguments arguments{cutlass::gemm::GemmUniversalMode::kGemm,
                                     {n, m, k, l},
                                     {
                                       B_ptr,
                                       stride_B,
                                         A_ptr,
                                         stride_A,
                                         SFA_ptr,
                                         layout_SFA,
                                         SFB_ptr,
                                         layout_SFB,
                                     },
                                     {
                                         {},  // epilogue.thread
                                         nullptr,
                                         stride_C,
                                         C_ptr,
                                         stride_C,
                                     },
                                     // KernelHardwareInfo
                                     []() {
                                       // For some reason can_implement fails if this is not defined
                                       auto hw_info = cutlass::KernelHardwareInfo::make_kernel_hardware_info<GemmKernel>();
                                       hw_info.cluster_shape = {1, 1, 1};
                                       hw_info.cluster_shape_fallback = {1, 1, 1};
                                       return hw_info;
                                     }(),
                                    //  scheduler_args
                                    };
  // using DecompositionMode = cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams::DecompositionMode;
  // using ReductionMode = cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams::ReductionMode;
  // arguments.scheduler.splits = 2;
  // arguments.scheduler.decomposition_mode = DecompositionMode::StreamK;
  // arguments.scheduler.reduction_mode = ReductionMode::Deterministic;
  auto& fusion_args = arguments.epilogue.thread;
  fusion_args.alpha = 1.0f;
  fusion_args.beta = 0.0f;

  Gemm gemm;

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  AlignedAllocator float_allocator(float_buffer, float_buffer_size_in_bytes);
  auto workspace_ptr = float_allocator.aligned_alloc<void>(workspace_size, 16,
                                                           "sm100_groupwise_gemm_float_workspace");

  // printf("testing...\n");
  CUTLASS_CHECK(gemm.can_implement(arguments));
  CUTLASS_CHECK(gemm.initialize(arguments, workspace_ptr));
  CUTLASS_CHECK(gemm.run(stream));
  return cudaSuccess;
}

}  // namespace gemm

}  // namespace flashinfer

#endif  // FLASHINFER_GEMM_GROUPWISE_SM100_CUH_
