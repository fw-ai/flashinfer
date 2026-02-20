"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
Benchmark for trtllm_moe_finalize_allreduce_fusion operator.

This benchmark measures the performance of the MoE finalize allreduce fusion
kernel which combines:
- MoE finalize (gather + scale from permuted indices)
- Shared expert output addition
- Cross-rank allreduce
- Residual addition
- RMSNorm

Usage:
    # Run with 2 GPUs
    torchrun --nproc_per_node=2 benchmarks/bench_trtllm_moe_finalize_allreduce_fusion.py

    # Run with 4 GPUs and custom parameters
    torchrun --nproc_per_node=4 benchmarks/bench_trtllm_moe_finalize_allreduce_fusion.py \
        --seq-lens 16 32 64 128 --hidden-size 7168 --top-k 8

    # Run with bfloat16
    torchrun --nproc_per_node=2 benchmarks/bench_trtllm_moe_finalize_allreduce_fusion.py \
        --dtype bfloat16
"""

import argparse
import os

import numpy as np
import torch
import torch.distributed as dist

import flashinfer.comm as comm

# Constants matching DeepSeek MoE configuration
DEFAULT_HIDDEN_SIZE = 7168
DEFAULT_TOP_K = 8
MAX_TOKEN_NUM = 2048


def get_bandwidth_gb_s(
    seq_len: int,
    hidden_size: int,
    top_k: int,
    world_size: int,
    latency_ms: float,
    dtype: torch.dtype,
) -> float:
    """Calculate effective bandwidth in GB/s."""
    element_size = torch.tensor([], dtype=dtype).element_size()

    # Input data:
    # - fc2_output (allreduce_in): [seq_len * top_k, hidden_size]
    # - shared_expert_output: [seq_len, hidden_size]
    # - scale: [seq_len, top_k]
    # - expanded_idx_to_permuted_idx: [seq_len, top_k] (int32)
    # - residual: [seq_len, hidden_size]
    # - norm_weight: [hidden_size]
    input_bytes = (
        seq_len * top_k * hidden_size * element_size  # fc2_output
        + seq_len * hidden_size * element_size  # shared_expert_output
        + seq_len * top_k * element_size  # scale
        + seq_len * top_k * 4  # expanded_idx_to_permuted_idx (int32)
        + seq_len * hidden_size * element_size  # residual
        + hidden_size * element_size  # norm_weight
    )

    # Output data:
    # - residual_out: [seq_len, hidden_size]
    # - norm_out: [seq_len, hidden_size]
    output_bytes = 2 * seq_len * hidden_size * element_size

    # Allreduce communication: each rank sends and receives
    # [seq_len, hidden_size] data to/from all other ranks
    allreduce_bytes = seq_len * hidden_size * element_size * world_size * 2

    total_bytes = input_bytes + output_bytes + allreduce_bytes
    bandwidth_gb_s = total_bytes / (latency_ms * 1e-3) / 1e9

    return bandwidth_gb_s


def benchmark_worker(
    rank: int,
    world_size: int,
    args: argparse.Namespace,
) -> None:
    """Worker function to run the benchmark on a single GPU."""
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # Initialize process group
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    group = dist.group.WORLD

    dtype = getattr(torch, args.dtype)
    hidden_size = args.hidden_size
    top_k = args.top_k
    eps = 1e-5

    try:
        # Create workspace for allreduce fusion
        ipc_handles, workspace_tensor = (
            comm.trtllm_create_ipc_workspace_for_all_reduce_fusion(
                rank, world_size, MAX_TOKEN_NUM, hidden_size, group=group
            )
        )

        if rank == 0:
            print(f"\n{'='*80}")
            print(f"MoE Finalize Allreduce Fusion Benchmark")
            print(f"{'='*80}")
            print(f"World size: {world_size}")
            print(f"Hidden size: {hidden_size}")
            print(f"Top-k: {top_k}")
            print(f"Dtype: {args.dtype}")
            print(f"PDL: {args.pdl}")
            print(f"{'='*80}\n")
            print(
                f"{'seq_len':>10} {'latency_us':>12} {'bandwidth_gb_s':>16} {'tflops':>10}"
            )
            print(f"{'-'*50}")

        for seq_len in args.seq_lens:
            # Generate input tensors
            torch.manual_seed(42 + rank)

            fc2_output = torch.randn(
                (seq_len * top_k, hidden_size), dtype=dtype, device=device
            )
            shared_expert_output = torch.randn(
                (seq_len, hidden_size), dtype=dtype, device=device
            )
            scale = torch.randn((seq_len, top_k), dtype=dtype, device=device)
            expanded_idx_to_permuted_idx = torch.randint(
                0, seq_len * top_k, (seq_len, top_k), dtype=torch.int32, device=device
            )
            residual = torch.randn((seq_len, hidden_size), dtype=dtype, device=device)
            norm_weight = torch.randn((hidden_size,), dtype=dtype, device=device)

            # Output tensors
            norm_out = torch.empty_like(residual)
            residual_out = torch.empty_like(residual)

            dist.barrier(group=group)

            # Warmup
            warmup_iters = 10
            for _ in range(warmup_iters):
                comm.trtllm_moe_finalize_allreduce_fusion(
                    allreduce_in=fc2_output,
                    residual_in=residual,
                    norm_weight=norm_weight,
                    expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                    workspace_ptrs=workspace_tensor,
                    launch_with_pdl=args.pdl,
                    world_rank=rank,
                    world_size=world_size,
                    eps=eps,
                    shared_expert_output=shared_expert_output,
                    expert_scale_factor=scale,
                    norm_out=norm_out,
                    residual_out=residual_out,
                )
            torch.cuda.synchronize()

            dist.barrier(group=group)

            # Benchmark with CUDA events
            num_iters = args.num_iters
            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

            for i in range(num_iters):
                start_events[i].record()
                comm.trtllm_moe_finalize_allreduce_fusion(
                    allreduce_in=fc2_output,
                    residual_in=residual,
                    norm_weight=norm_weight,
                    expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                    workspace_ptrs=workspace_tensor,
                    launch_with_pdl=args.pdl,
                    world_rank=rank,
                    world_size=world_size,
                    eps=eps,
                    shared_expert_output=shared_expert_output,
                    expert_scale_factor=scale,
                    norm_out=norm_out,
                    residual_out=residual_out,
                )
                end_events[i].record()

            torch.cuda.synchronize()

            # Calculate latencies
            latencies_ms = [
                start_events[i].elapsed_time(end_events[i]) for i in range(num_iters)
            ]
            median_latency_ms = np.median(latencies_ms)
            latency_us = median_latency_ms * 1000

            # Calculate bandwidth
            bandwidth_gb_s = get_bandwidth_gb_s(
                seq_len, hidden_size, top_k, world_size, median_latency_ms, dtype
            )

            # Calculate TFLOPS (approximate)
            # - MoE finalize: seq_len * top_k * hidden_size * 2 (multiply + add)
            # - Shared expert add: seq_len * hidden_size
            # - Residual add: seq_len * hidden_size
            # - RMSNorm: seq_len * hidden_size * 5 (square, sum, rsqrt, multiply, multiply)
            flops = (
                seq_len * top_k * hidden_size * 2
                + seq_len * hidden_size
                + seq_len * hidden_size
                + seq_len * hidden_size * 5
            )
            tflops = flops / (median_latency_ms * 1e-3) / 1e12

            dist.barrier(group=group)

            if rank == 0:
                print(f"{seq_len:>10} {latency_us:>12.2f} {bandwidth_gb_s:>16.2f} {tflops:>10.3f}")

        if rank == 0:
            print(f"\n{'='*80}\n")

    finally:
        dist.barrier(group=group)
        comm.trtllm_destroy_ipc_workspace_for_all_reduce(ipc_handles, group=group)
        dist.destroy_process_group(group=group)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark trtllm_moe_finalize_allreduce_fusion"
    )
    parser.add_argument(
        "--seq-lens",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 32, 64, 128],
        help="Sequence lengths to benchmark",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=DEFAULT_HIDDEN_SIZE,
        help="Hidden dimension size",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of experts per token",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "bfloat16"],
        default="bfloat16",
        help="Data type",
    )
    parser.add_argument(
        "--pdl",
        action="store_true",
        default=True,
        help="Use programmatic dependent launch",
    )
    parser.add_argument(
        "--no-pdl",
        action="store_false",
        dest="pdl",
        help="Disable programmatic dependent launch",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=100,
        help="Number of iterations for benchmarking",
    )
    args = parser.parse_args()

    # Get rank and world_size from environment (set by torchrun)
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size < 2:
        print("Error: This benchmark requires at least 2 GPUs.")
        print("Usage: torchrun --nproc_per_node=2 benchmarks/bench_trtllm_moe_finalize_allreduce_fusion.py")
        return

    benchmark_worker(rank, world_size, args)


if __name__ == "__main__":
    main()
