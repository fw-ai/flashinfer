"""Unsupported Fireworks-private ABI surface for the isolated SiTU b552 runner.

These entry points are intentionally absent from the public ``flashinfer`` and
``flashinfer.fused_moe`` namespaces.  They are version-locked to the matched
Fireworks cubin and JIT-cache wheels and may change without compatibility.
"""

from .core import (
    trtllm_fp4_block_scale_situ_moe as trtllm_fp4_block_scale_situ_moe,
)
from .core import (
    trtllm_fp4_block_scale_situ_routed_moe as trtllm_fp4_block_scale_situ_routed_moe,
)
