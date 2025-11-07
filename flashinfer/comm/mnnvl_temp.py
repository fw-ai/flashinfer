# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Code imported from TensorRT-LLM
its a newer version of mnnvl.py that's faster and supports nvfp4
Avoids updating mnnvl.py to avoid merge conflicts
We can delete this file once the new changes from trtllm are upstreamed.
"""
from abc import ABC, abstractmethod
import ctypes
import os
import platform
import sys
from dataclasses import dataclass
from typing import List, Optional, Union

import pynvml
import torch
import torch.distributed as dist

try:
    from cuda.bindings import driver as cuda
except ImportError:
    from cuda import cuda

from enum import IntEnum
from typing import List

import torch

from ..jit import gen_comm_alltoall_module

class CpType(IntEnum):
    # CP type for ulysses parallelism
    ULYSSES = 0
    # CP type for star attention
    STAR = 1
    # CP type for ring attention
    RING = 2
    # CP type for helix parallelism
    HELIX = 3


class Mapping(object):
    '''
    A node with 8 GPUs, tp_size = 4, cp_size = 1, pp_size = 2

    2 tp groups:

    - [0, 1, 2, 3]
    - [4, 5, 6, 7]

    4 pp groups:

    - [0, 4]
    - [1, 5]
    - [2, 6]
    - [3, 7]

    A node with 8 GPUs, tp_size = 4, cp_size = 2, pp_size = 1

    2 tp groups:

    - [0, 1, 2, 3]
    - [4, 5, 6, 7]

    4 cp groups:

    - [0, 4]
    - [1, 5]
    - [2, 6]
    - [3, 7]

    A node with 8 GPUs, moe_tp_size = 2, moe_ep_size = 4

    4 moe_tp groups:

    - [0, 4]
    - [1, 5]
    - [2, 6]
    - [3, 7]

    2 moe_ep groups:

    - [0, 1, 2, 3]
    - [4, 5, 6, 7]

    2 nodes with 16 GPUs, moe_tp_size = 2, moe_ep_size = 4, pp_size = 2

    8 moe_tp groups:

    - [0 4]
    - [1 5]
    - [2 6]
    - [3 7]
    - [8 12]
    - [9 13]
    - [10 14]
    - [11 15]

    4 moe_ep groups:

    - [0, 1, 2, 3]
    - [4, 5, 6, 7]
    - [8, 9, 10, 11]
    - [12, 13, 14, 15]

    8 pp groups:

    - [0 8]
    - [1 9]
    - [2 10]
    - [3 11]
    - [4 12]
    - [5 13]
    - [6 14]
    - [7 15]

    2 nodes with 8 GPUs, tp_size 2, pp_size 2, cp_size 2

    4 tp groups:
    - [0, 1]
    - [2, 3]
    - [4, 5]
    - [6, 7]

    4 pp groups:
    - [0, 4]
    - [1, 5]
    - [2, 6]
    - [3, 7]

    4 cp groups:
    - [0, 2]
    - [1, 3]
    - [4, 6]
    - [5, 7]
    '''

    def __init__(
            self,
            world_size=1,
            rank=0,
            gpus_per_node=8,
            *,
            cp_size=1,
            cp_config=None,
            tp_size=1,
            pp_size=1,
            moe_cluster_size=-1,  # -1 means no moe
            moe_tp_size=-1,  # -1 means no moe
            moe_ep_size=-1,  # -1 means no moe
            attn_tp_size=-1,
            attn_cp_size=-1,
            auto_parallel=False,
            enable_attention_dp=False):
        # set default values for non-moe cases
        # or where only one MOE parallelism size is specified
        if moe_cluster_size == -1:
            moe_cluster_size = 1

        cp_type = CpType.ULYSSES if cp_config is None else cp_config.get(
            "cp_type", CpType.ULYSSES)
        moe_world_size = tp_size if cp_type == CpType.ULYSSES else tp_size * cp_size

        if moe_tp_size == -1 and moe_ep_size == -1:
            moe_tp_size = moe_world_size // moe_cluster_size
            moe_ep_size = 1

        elif moe_tp_size == -1:
            moe_tp_size = moe_world_size // (moe_ep_size * moe_cluster_size)

        elif moe_ep_size == -1:
            moe_ep_size = moe_world_size // (moe_tp_size * moe_cluster_size)

        if attn_tp_size == -1 and attn_cp_size == -1:
            if cp_type == CpType.ULYSSES:
                # fallback to ulysses
                attn_tp_size = tp_size * cp_size
                attn_cp_size = 1
            else:
                # fallback to helix
                attn_tp_size = tp_size
                attn_cp_size = cp_size

        elif attn_tp_size == -1:
            attn_tp_size = (tp_size * cp_size) // attn_cp_size

        elif attn_cp_size == -1:
            attn_cp_size = (tp_size * cp_size) // attn_tp_size

        if attn_cp_size != 1 and cp_type == CpType.ULYSSES:
            raise ValueError(
                f"attn_cp_size must be 1 for now for ulysses, but got {attn_tp_size}, {attn_cp_size}."
            )

        if auto_parallel:
            if tp_size != 1 or pp_size != 1 or cp_size != 1:
                raise ValueError(
                    "When auto parallel is enabled, tp_size, pp_size, cp_size must be 1, "
                    f"but got {tp_size}, {pp_size}, {cp_size}.")
        else:
            if tp_size * pp_size * cp_size != world_size:
                raise ValueError(
                    "world_size must equal to tp_size * pp_size * cp_size, "
                    f"but got {world_size} != {tp_size} * {pp_size} * {cp_size}."
                )

        moe_tp_ep_size = moe_tp_size * moe_ep_size
        moe_tp_cluster_ep_size = moe_tp_ep_size * moe_cluster_size
        if moe_tp_cluster_ep_size != moe_world_size:
            raise ValueError(
                "moe_tp_size * moe_ep_size * moe_cluster_size must equal to moe_world_size, "
                f"but got {moe_tp_cluster_ep_size} != {moe_world_size}")

        attn_tp_cp_size = attn_tp_size * attn_cp_size
        if attn_tp_cp_size != tp_size * cp_size:
            raise ValueError(
                "tp_size * cp_size must equal to attn_tp_size * attn_cp_size, "
                f"but got {tp_size} * {cp_size} != {attn_tp_size} * {attn_cp_size}"
            )

        if moe_ep_size != 1 and cp_size > 1 and cp_type != CpType.HELIX:
            raise NotImplementedError(
                f"CP {cp_type} doesn't support MoE tp/ep yet")

        self.tp_size = tp_size
        self.cp_size = cp_size
        self.cp_config = cp_config if cp_config is not None else {}
        self.pp_size = pp_size
        self.moe_tp_size = moe_tp_size
        self.moe_ep_size = moe_ep_size
        self.moe_cluster_size = moe_cluster_size
        self.attn_tp_size = attn_tp_size
        self.attn_cp_size = attn_cp_size
        self.auto_parallel = auto_parallel
        self.world_size = world_size
        self.enable_attention_dp = enable_attention_dp
        self.rank = rank
        self.gpus_per_node = gpus_per_node
        self.pp_groups = []
        self.cp_groups = []
        self.tp_groups = []
        self.moe_cluster_groups = []
        self.moe_tp_groups = []
        self.moe_ep_groups = []

        if moe_cluster_size > 1:
            assert moe_ep_size == 1

        # init pp group
        for i in range(tp_size * cp_size):
            ranks = range(i, world_size, tp_size * cp_size)
            self.pp_groups.append(list(ranks))

        # init cp group
        for i in range(pp_size):
            for j in range(tp_size):
                ranks = range(i * tp_size * cp_size + j,
                              (i + 1) * tp_size * cp_size + j, tp_size)
                self.cp_groups.append(list(ranks))

        # init tp group
        for i in range(pp_size):
            for j in range(cp_size):
                ranks = range(i * tp_size * cp_size + j * tp_size,
                              i * tp_size * cp_size + (j + 1) * tp_size)
                self.tp_groups.append(list(ranks))

        # init moe tp group
        for i in range(pp_size):
            for j in range(moe_cluster_size * moe_ep_size):
                ranks = range(i * moe_tp_cluster_ep_size + j,
                              (i + 1) * moe_tp_cluster_ep_size,
                              moe_cluster_size * moe_ep_size)
                self.moe_tp_groups.append(list(ranks))

        # init moe cluster group
        for i in range(pp_size):
            for j in range(moe_tp_size):
                ranks = range(
                    i * moe_tp_cluster_ep_size +
                    j * moe_cluster_size * moe_ep_size,
                    i * moe_tp_cluster_ep_size +
                    (j + 1) * moe_cluster_size * moe_ep_size)
                self.moe_cluster_groups.append(list(ranks))

        # init moe ep group
        for i in range(pp_size):
            for j in range(moe_tp_size):
                for k in range(moe_cluster_size):
                    ranks = range(
                        i * moe_tp_cluster_ep_size +
                        j * moe_cluster_size * moe_ep_size + k * moe_ep_size,
                        i * moe_tp_cluster_ep_size +
                        j * moe_cluster_size * moe_ep_size +
                        (k + 1) * moe_ep_size)
                    self.moe_ep_groups.append(list(ranks))

    def __eq__(self, other):
        if not isinstance(other, Mapping):
            return NotImplemented

        return (self.world_size == other.world_size and self.rank == other.rank
                and self.gpus_per_node == other.gpus_per_node
                and self.cp_size == other.cp_size
                and self.tp_size == other.tp_size
                and self.moe_cluster_size == other.moe_cluster_size
                and self.pp_size == other.pp_size
                and self.moe_tp_size == other.moe_tp_size
                and self.moe_ep_size == other.moe_ep_size
                and self.attn_tp_size == other.attn_tp_size
                and self.attn_cp_size == other.attn_cp_size
                and self.cp_config == other.cp_config
                and self.auto_parallel == other.auto_parallel)

    def __hash__(self):
        return hash((
            self.world_size,
            self.rank,
            self.gpus_per_node,
            self.cp_size,
            self.tp_size,
            self.pp_size,
            self.moe_tp_size,
            self.moe_cluster_size,
            self.moe_ep_size,
            self.attn_tp_size,
            self.attn_cp_size,
            # note: we do not allow updating cp_config after initialization
            tuple(sorted(self.cp_config.items())),
            self.auto_parallel,
        ))

    @property
    def rank(self):
        return self._rank

    @rank.setter
    def rank(self, rank: int):
        # TODO(qijun): skip check for enable_attention_dp temporarily, will support attention_dp_size
        if not self.enable_attention_dp:
            if not isinstance(rank, int) or rank < 0 or rank >= self.world_size:
                raise ValueError(
                    f"Rank should be an integer between 0 and {self.world_size-1}, but got {rank}."
                )
        self._rank = rank

    @property
    def tp_rank(self):
        return 0 if self.auto_parallel else self.rank % self.tp_size

    @property
    def pp_rank(self):
        return 0 if self.auto_parallel else self.rank // (self.tp_size *
                                                          self.cp_size)

    @property
    def cp_rank(self):
        return 0 if self.auto_parallel else self.rank % (
            self.tp_size * self.cp_size) // self.tp_size

    @property
    def moe_tp_rank(self):
        return self.tp_rank // (self.moe_ep_size * self.moe_cluster_size)

    @property
    def moe_cluster_rank(self):
        return self.tp_rank % self.moe_cluster_size

    @property
    def moe_ep_rank(self):
        return self.tp_rank % self.moe_ep_size

    @property
    def tp_group(self):
        return self.tp_groups[self.pp_rank * self.cp_size + self.cp_rank]

    @property
    def pp_group(self):
        return self.pp_groups[self.cp_rank * self.tp_size + self.tp_rank]

    @property
    def cp_group(self):
        return self.cp_groups[self.pp_rank * self.tp_size + self.tp_rank]

    @property
    def moe_tp_group(self):
        return self.moe_tp_groups[self.pp_rank * self.moe_cluster_size *
                                  self.moe_ep_size +
                                  self.moe_cluster_rank * self.moe_ep_size +
                                  self.moe_ep_rank]

    @property
    def moe_cluster_group(self):
        return self.moe_cluster_groups[self.pp_rank * self.moe_tp_size +
                                       self.moe_tp_rank]

    @property
    def moe_ep_group(self):
        return self.moe_ep_groups[self.pp_rank * self.moe_tp_size *
                                  self.moe_cluster_size +
                                  self.moe_tp_rank * self.moe_cluster_size +
                                  self.moe_cluster_rank]

    @property
    def node_rank(self):
        return self.rank // self.gpus_per_node

    @property
    def local_rank(self):
        return self.rank % self.gpus_per_node

    @property
    def dp_size(self):
        return self.tp_size if self.enable_attention_dp else 1

    def has_cp_ulysses(self):
        return self.cp_size > 1 and self.cp_config.get(
            "cp_type") == CpType.ULYSSES

    def has_cp_helix(self):
        return self.cp_size > 1 and self.cp_config.get(
            "cp_type") == CpType.HELIX

    def get_node_rank(self, rank: int):
        return rank // self.gpus_per_node

    def get_local_rank(self, rank: int):
        return rank % self.gpus_per_node

    def is_multi_node(self):
        return self.world_size > self.gpus_per_node

    def has_tp(self):
        return self.tp_size > 1

    def is_last_pp_rank(self):
        return self.pp_rank == self.pp_size - 1

    def is_second_last_pp_rank(self):
        return self.pp_rank == self.pp_size - 2

    def is_first_pp_rank(self):
        return self.pp_rank == 0

    def has_pp(self):
        return self.pp_size > 1

    def prev_pp_rank(self):
        p = self.rank - self.tp_size * self.cp_size
        if p < 0:
            p = p + self.world_size
        return p

    def next_pp_rank(self):
        p = self.rank + self.tp_size * self.cp_size
        if p >= self.world_size:
            p = p - self.world_size
        return p

    def is_last_cp_rank(self):
        return self.cp_rank == self.cp_size - 1

    def is_first_cp_rank(self):
        return self.cp_rank == 0

    def has_cp(self):
        return self.cp_size > 1

    def prev_cp_rank(self):
        p = self.rank - self.tp_size
        if p // (self.tp_size * self.cp_size) < self.rank // (self.tp_size *
                                                              self.cp_size):
            return p + self.tp_size * self.cp_size
        return p

    def next_cp_rank(self):
        p = self.rank + self.tp_size
        if p // (self.tp_size * self.cp_size) > self.rank // (self.tp_size *
                                                              self.cp_size):
            return p - self.tp_size * self.cp_size
        return p

    def has_moe_cluster(self):
        return self.moe_cluster_size > 1

    def has_moe_tp(self):
        return self.moe_tp_size > 1

    def has_moe_ep(self):
        return self.moe_ep_size > 1

    def pp_layers(self, num_layers: int) -> List[int]:
        # If num_layers % pp_size = n != 0, first n ranks get one extra layer
        return torch.tensor_split(torch.arange(num_layers),
                                  self.pp_size)[self.pp_rank].tolist()

    def ep_experts(self, num_experts: int) -> List[int]:
        assert self.cp_size == 1
        experts_per_rank = num_experts // self.moe_ep_size
        experts_range = range(self.moe_ep_rank * experts_per_rank,
                              (self.moe_ep_rank + 1) * experts_per_rank)
        return list(experts_range)

    @classmethod
    def from_dict(cls, mapping: dict):
        return cls(**mapping)

    def to_dict(self):
        return {
            'world_size': self.world_size,
            'rank': self.rank,
            'gpus_per_node': self.gpus_per_node,
            'cp_size': self.cp_size,
            'tp_size': self.tp_size,
            'pp_size': self.pp_size,
            'moe_tp_size': self.moe_tp_size,
            'moe_cluster_size': self.moe_cluster_size,
            'moe_ep_size': self.moe_ep_size,
            'attn_tp_size': self.attn_tp_size,
            'attn_cp_size': self.attn_cp_size,
            'cp_config': self.cp_config,
            'auto_parallel': self.auto_parallel,
        }


import ctypes
from ctypes import (
    CFUNCTYPE,
    POINTER,
    c_int,
    c_int64,
    c_size_t,
    c_uint8,
    c_uint16,
    c_void_p,
    pointer,
)

import torch

# Define data structures required for DLPack
class DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", c_uint8),  # Data type code, e.g., 2 for float
        ("bits", c_uint8),  # Number of bits per element, e.g., 32
        ("lanes", c_uint16),  # Number of lanes, usually 1
    ]


class DLDevice(ctypes.Structure):
    _fields_ = [
        ("device_type", c_int),  # Device type, typically 2 for GPU
        ("device_id", c_int),  # Device ID, usually 0 for default GPU
    ]


class DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", c_void_p),  # Data pointer
        ("device", DLDevice),  # Device information
        ("ndim", c_int),  # Number of dimensions
        ("dtype", DLDataType),  # Data type
        ("shape", POINTER(c_int64)),  # Pointer to array of dimension sizes
        (
            "strides",
            POINTER(c_int64),
        ),  # Pointer to strides array (can be NULL for default contiguous layout)
        ("byte_offset", c_size_t),  # Byte offset (usually 0)
    ]


# Deleter type for DLManagedTensor
DLManagedTensorDeleter = CFUNCTYPE(None, POINTER(ctypes.c_void_p))  # Not used directly here


# Define DLManagedTensor structure, with deleter prototype void(*deleter)(DLManagedTensor*)
class DLManagedTensor(ctypes.Structure):
    pass


DLManagedTensor._fields_ = [
    ("dl_tensor", DLTensor),
    ("manager_ctx", c_void_p),
    ("deleter", CFUNCTYPE(None, POINTER(DLManagedTensor))),
]


# A no-op deleter that doesn't perform any operation
@CFUNCTYPE(None, POINTER(DLManagedTensor))
def no_op_deleter(dmt_ptr):
    # You can also call cudaFree here if you want to free memory when the tensor's lifecycle ends
    pass


# Wrapper class to prevent Python garbage collection of DLPack-related objects
class CapsuleWrapper:
    """
    A wrapper class that holds references to the PyCapsule and its associated data.

    This class prevents Python's garbage collector from collecting the shape_array and
    managed_tensor objects while the capsule is still in use. It serves as a container
    to maintain the lifecycle of all DLPack-related objects.
    """

    def __init__(self, capsule, shape_array, managed_tensor):
        """
        Initialize the CapsuleWrapper with the necessary objects.

        Parameters:
            capsule: The PyCapsule object that follows the DLPack protocol
            shape_array: The array containing tensor shape information
            managed_tensor: The DLManagedTensor instance that the capsule points to
        """
        self.capsule = capsule  # The main PyCapsule object that can be passed to other libraries
        self._shape_array = shape_array  # Keep reference to prevent garbage collection
        self._managed_tensor = managed_tensor  # Keep reference to prevent garbage collection


def create_dlpack_capsule(ptr, segment_size, segment_stride, num_segments, torch_dtype, dev_id):
    """
    Parameters:
      ptr: GPU memory address obtained from cudaMalloc (Python int)
      segment_size: Memory size of each segments in bytes
      segment_stride: Memory stride size between segments in bytes
      num_segments: Number of segments
      torch_dtype: torch dtype
      dev_id: device id.
    Returns:
      A PyCapsule object compliant with DLPack specification, which can be directly converted to a
      tensor using torch.utils.dlpack.from_dlpack
    """
    bits_per_elements = 0
    dldata_type_code = 0
    # refer to https://github.com/dmlc/dlpack/blob/main/include/dlpack/dlpack.h#L160
    if torch_dtype in [
        torch.float8_e5m2,
        torch.float8_e4m3fn,
        torch.bfloat16,
        torch.float16,
        torch.float32,
        torch.float64,
    ]:
        bits_per_elements = torch.finfo(torch_dtype).bits
        dldata_type_code = 2
    elif torch_dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
        bits_per_elements = torch.iinfo(torch_dtype).bits
        dldata_type_code = 0
    elif torch_dtype in [torch.uint8, torch.uint16, torch.uint32, torch.uint64]:
        bits_per_elements = torch.iinfo(torch_dtype).bits
        dldata_type_code = 1
    else:
        raise NotImplementedError(torch_dtype)
    bytes_per_element = bits_per_elements // 8
    # Allocate space for shape (constructing a one-dimensional tensor here)
    ShapeArrayType = c_int64 * 2  # 1 dimension
    shape_array = ShapeArrayType(num_segments, segment_size // bytes_per_element)
    stride_array = ShapeArrayType(segment_stride // bytes_per_element, 1)
    # Set device information: GPU (device_type=2) and device_id=dev_id (modify as needed)
    device = DLDevice(device_type=2, device_id=dev_id)
    # Set data type
    dtype = DLDataType(code=dldata_type_code, bits=bits_per_elements, lanes=1)
    # Construct DLTensor
    dltensor = DLTensor()
    dltensor.data = c_void_p(ptr)
    dltensor.device = device
    dltensor.ndim = 2
    dltensor.dtype = dtype
    dltensor.shape = ctypes.cast(shape_array, POINTER(c_int64))
    dltensor.strides = ctypes.cast(stride_array, POINTER(c_int64))
    dltensor.byte_offset = 0
    # Construct DLManagedTensor and set deleter to no-op (you can also call cudaFree here)
    managed_tensor = DLManagedTensor()
    managed_tensor.dl_tensor = dltensor
    managed_tensor.manager_ctx = None
    managed_tensor.deleter = no_op_deleter
    # Note: Must ensure that shape_array and managed_tensor are not garbage collected by Python,
    # A simple way is to attach them to the capsule object.
    # Call PyCapsule_New to create capsule
    PyCapsule_New = ctypes.pythonapi.PyCapsule_New
    PyCapsule_New.restype = c_void_p
    PyCapsule_New.argtypes = [c_void_p, ctypes.c_char_p, c_void_p]
    # Allocate managed_tensor on the heap (note that pointer returns a pointer)
    managed_tensor_ptr = pointer(managed_tensor)
    # The capsule name must be "dltensor", as required by the DLPack specification
    capsule_ptr = PyCapsule_New(managed_tensor_ptr, b"dltensor", None)
    # Convert capsule_ptr to Python object
    capsule = ctypes.cast(capsule_ptr, ctypes.py_object).value
    # To prevent shape_array and managed_tensor from being collected, we attach them as attributes to the capsule
    capsule_wrapper = CapsuleWrapper(capsule, shape_array, managed_tensor)
    return capsule_wrapper


def pack_strided_memory(
    ptr: int, segment_size: int, segment_stride: int, num_segments: int, dtype: torch.dtype, dev_id
):
    """
    Pack GPU memory into a PyTorch tensor with specified stride.

    Parameters:
        ptr: GPU memory address obtained from cudaMalloc
        segment_size: Memory size of each segment in bytes
        segment_stride: Memory stride size between segments in bytes
        num_segments: Number of segments
        dtype: PyTorch data type for the resulting tensor
        dev_id: CUDA device ID

    Returns:
        PyTorch tensor that references the provided memory

    Note:
        This function creates a new DLPack capsule each time it's called,
        even with the same pointer. Each capsule is consumed only once.
    """
    # Create a new capsule each time
    capsule_wrapper = create_dlpack_capsule(
        ptr, segment_size, segment_stride, num_segments, dtype, dev_id
    )
    torch_tensor = torch.utils.dlpack.from_dlpack(capsule_wrapper.capsule)
    torch_tensor._capsule_wrapper = capsule_wrapper
    return torch_tensor


def _check_cu_result(cu_func_ret):
    if isinstance(cu_func_ret, tuple):
        cu_result, *others = cu_func_ret
        if cu_result != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(cu_result)
        if len(others) == 1:
            return others[0]
        elif len(others) > 1:
            return tuple(others)
        else:
            return None
    else:
        if cu_func_ret != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(cu_func_ret)
        return None

class CommBackend(ABC):
    """Abstract communication backend interface"""

    @abstractmethod
    def Get_rank(self) -> int: ...

    @abstractmethod
    def Get_size(self) -> int: ...

    @abstractmethod
    def allgather(self, data: int) -> List[int]: ...

    @abstractmethod
    def Split(self, color: int, key: int) -> "CommBackend": ...

class MnnvlMemory:
    initialized: bool = False

    current_mem_offset: int = 0
    current_rank_stride: int = 0  # stride for ranks and also address space size.
    current_start_address: int = 0

    # allocation granularity
    allocation_granularity: int = 0

    # fabric address page size (512 MB)
    fabric_page_size: int = 1 << 29

    # MPI communicator
    comm = None

    dev_id: int = None

    allocated_map = {}
    address_refcnt = {}

    def __init__(self, mapping: Mapping, size: int):
        self.mapping = mapping
        self.segment_size = size
        self.ptr, self.rank_stride = MnnvlMemory.open_mnnvl_memory(self.mapping, size)

    def __del__(self):
        if not sys.is_finalizing():
            if hasattr(self, "ptr"):
                MnnvlMemory.close_mnnvl_memory(self.ptr)

    def as_torch_strided_tensor(self, dtype):
        num_segments = MnnvlMemory.comm.Get_size()
        return pack_strided_memory(
            self.ptr, self.segment_size, self.rank_stride, num_segments, dtype, MnnvlMemory.dev_id
        )

    @staticmethod
    def initialize(comm_backend: CommBackend):
        if not MnnvlMemory.initialized:
            # use a dummy torch CUDA tensor to trigger CUDA context initialization
            _ = torch.empty(1, device="cuda")
            # ensure nvml is initialized.
            try:
                pynvml.nvmlDeviceGetCount()
            except pynvml.NVMLError_Uninitialized:
                pynvml.nvmlInit()
            MnnvlMemory.initialized = True
            MnnvlMemory.comm_backend = comm_backend

    @staticmethod
    def get_comm(mapping: Mapping):
        if MnnvlMemory.comm is not None:
            return MnnvlMemory.comm
        comm = MnnvlMemory.comm_backend.Split(
            mapping.pp_rank * mapping.cp_size + mapping.cp_rank, mapping.tp_rank
        )
        MnnvlMemory.comm = comm
        return comm

    @staticmethod
    def get_allocation_prop(dev_id: int):
        location = cuda.CUmemLocation()
        location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        location.id = dev_id
        allocation_prop = cuda.CUmemAllocationProp()
        allocation_prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED

        # TODO: We differentiate FABRIC for GB200 (aarch64) and POSIX_FILE_DESCRIPTOR for BB200 (x86_64).
        # May need to find a better way to handle this.
        arch = platform.machine().lower()
        is_on_aarch64 = "aarch64" in arch
        if is_on_aarch64:
            allocation_prop.requestedHandleTypes = (
                cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
            )
        else:
            allocation_prop.requestedHandleTypes = (
                cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
            )
        allocation_prop.location = location
        return allocation_prop

    @staticmethod
    def get_allocation_granularity(dev_id: int):
        if MnnvlMemory.allocation_granularity != 0:
            return MnnvlMemory.allocation_granularity
        allocation_prop = MnnvlMemory.get_allocation_prop(dev_id)
        option = cuda.CUmemAllocationGranularity_flags(
            cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
        )
        granularity = _check_cu_result(
            cuda.cuMemGetAllocationGranularity(prop=allocation_prop, option=option)
        )
        MnnvlMemory.allocation_granularity = granularity
        return MnnvlMemory.allocation_granularity

    @staticmethod
    def new_mnnvl_memory_address(mapping: Mapping, size: int):
        page_count = (size + MnnvlMemory.fabric_page_size - 1) // MnnvlMemory.fabric_page_size
        current_rank_stride = page_count * MnnvlMemory.fabric_page_size
        comm = MnnvlMemory.get_comm(mapping)
        comm_size = comm.Get_size()
        address_size = current_rank_stride * comm_size
        ptr = _check_cu_result(
            cuda.cuMemAddressReserve(address_size, MnnvlMemory.fabric_page_size, 0, 0)
        )
        MnnvlMemory.current_start_address = int(ptr)
        MnnvlMemory.current_rank_stride = current_rank_stride
        MnnvlMemory.current_mem_offset = 0

    @staticmethod
    def open_mnnvl_memory(mapping: Mapping, size: int):
        dev = _check_cu_result(cuda.cuCtxGetDevice())
        dev_id = int(dev)
        if MnnvlMemory.dev_id is None:
            MnnvlMemory.dev_id = dev_id
        assert dev_id == MnnvlMemory.dev_id, (
            f"Different dev_id found dev_id={dev_id} but MnnvlMemory.dev_id={MnnvlMemory.dev_id}"
        )
        comm = MnnvlMemory.get_comm(mapping)
        comm_rank = comm.Get_rank()
        comm_size = comm.Get_size()
        all_rank_allocate_sizes = comm.allgather(size)
        assert len(all_rank_allocate_sizes) == comm_size
        assert all(x == size for x in all_rank_allocate_sizes), "Not all rank allocating same size."
        granularity = MnnvlMemory.get_allocation_granularity(dev_id)
        aligned_size = (size + granularity - 1) // granularity * granularity

        if MnnvlMemory.current_mem_offset + aligned_size > MnnvlMemory.current_rank_stride:
            MnnvlMemory.new_mnnvl_memory_address(mapping, aligned_size)

        assert MnnvlMemory.current_mem_offset + aligned_size <= MnnvlMemory.current_rank_stride

        allocation_prop = MnnvlMemory.get_allocation_prop(dev_id)
        allocated_mem_handle = _check_cu_result(
            cuda.cuMemCreate(aligned_size, allocation_prop, flags=0)
        )
        exported_fabric_handle = _check_cu_result(
            cuda.cuMemExportToShareableHandle(
                allocated_mem_handle, allocation_prop.requestedHandleTypes, 0
            )
        )
        if (
            allocation_prop.requestedHandleTypes
            == cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
        ):
            all_handles_data = comm.allgather(exported_fabric_handle.data)
        else:
            all_handles_data = comm.allgather(exported_fabric_handle)
            all_pids = comm.allgather(os.getpid())
            libc = ctypes.CDLL(None, use_errno=True)
            syscall = libc.syscall
            SYS_pidfd_open = 434
            SYS_pidfd_getfd = 438
            pidfds = []
            for i, pid in enumerate(all_pids):
                pidfd = syscall(SYS_pidfd_open, pid, 0)
                if pidfd < 0:
                    err = ctypes.get_errno()
                    raise RuntimeError(
                        f"pidfd_open({pid}) failed with errno {err}: {os.strerror(err)}"
                    )
                pidfds.append(pidfd)

            remote_fds = []
            for i, (pidfd, fd) in enumerate(zip(pidfds, all_handles_data)):
                remote_fd = syscall(SYS_pidfd_getfd, pidfd, fd, 0)
                if remote_fd < 0:
                    err = ctypes.get_errno()
                    error_msg = f"pidfd_getfd(pidfd={pidfd}, fd={fd}) failed with errno {err}: {os.strerror(err)}."
                    if err == 1:  # EPERM
                        error_msg += (
                            " Permission denied. If running in a container, try adding --cap-add=SYS_PTRACE "
                            "to your docker run command."
                        )
                    else:
                        error_msg += " This may be due to kernel version (requires Linux 5.6+)."
                    raise RuntimeError(error_msg)
                remote_fds.append(remote_fd)

            all_handles_data = remote_fds
        # all_handles_data like b'\x00\x00\x00 \x00\x00\x00\x00\x8f\xec\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\t\x00\x00\x00\x00\x00\x1d\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'  # noqa: E501
        # can use buf = memoryview(data) to import if using plain buffer for data.

        madesc = cuda.CUmemAccessDesc()
        madesc.location = allocation_prop.location
        madesc.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE

        mem_handles = [None] * comm_size

        for i, remote_handle_data in enumerate(all_handles_data):
            rank_ptr = (
                MnnvlMemory.current_start_address
                + MnnvlMemory.current_rank_stride * i
                + MnnvlMemory.current_mem_offset
            )
            if i == comm_rank:
                # Local memory mapping
                mem_handles[i] = allocated_mem_handle
                _check_cu_result(cuda.cuMemMap(rank_ptr, aligned_size, 0, allocated_mem_handle, 0))
            else:
                # Fabric memory mapping
                imported_mem_handle = _check_cu_result(
                    cuda.cuMemImportFromShareableHandle(
                        remote_handle_data, allocation_prop.requestedHandleTypes
                    )
                )
                mem_handles[i] = imported_mem_handle
                _check_cu_result(cuda.cuMemMap(rank_ptr, aligned_size, 0, imported_mem_handle, 0))

            _check_cu_result(cuda.cuMemSetAccess(rank_ptr, aligned_size, [madesc], 1))

        ptr = MnnvlMemory.current_start_address + MnnvlMemory.current_mem_offset
        stride = MnnvlMemory.current_rank_stride
        MnnvlMemory.allocated_map[ptr] = (
            mapping,
            aligned_size,
            mem_handles,
            MnnvlMemory.current_start_address,
            MnnvlMemory.current_rank_stride,
            MnnvlMemory.current_mem_offset,
        )
        MnnvlMemory.address_refcnt[MnnvlMemory.current_start_address] = (
            MnnvlMemory.address_refcnt.get(MnnvlMemory.current_start_address, 0) + 1
        )

        MnnvlMemory.current_mem_offset += aligned_size
        return ptr, stride

    @staticmethod
    def close_mnnvl_memory(ptr: int):
        mapping, aligned_size, mem_handles, start_address, rank_stride, address_offset = (
            MnnvlMemory.allocated_map.pop(ptr)
        )
        comm = MnnvlMemory.get_comm(mapping)
        comm_size = comm.Get_size()
        for i in range(comm_size):
            rank_ptr = start_address + i * rank_stride + address_offset
            _check_cu_result(cuda.cuMemUnmap(rank_ptr, aligned_size))
            _check_cu_result(cuda.cuMemRelease(mem_handles[i]))
        MnnvlMemory.address_refcnt[start_address] -= 1

        if MnnvlMemory.address_refcnt[start_address] == 0:
            MnnvlMemory.address_refcnt.pop(start_address)
            device_ptr = cuda.CUdeviceptr(start_address)
            _check_cu_result(cuda.cuMemAddressFree(device_ptr, comm_size * rank_stride))
            if start_address == MnnvlMemory.current_start_address:
                MnnvlMemory.current_start_address = 0
                MnnvlMemory.current_rank_stride = 0
                MnnvlMemory.current_mem_offset = 0

    @staticmethod
    def support_nvlink(need_all_up: bool = True):
        dev_id = torch.cuda.current_device()
        handle = pynvml.nvmlDeviceGetHandleByIndex(dev_id)
        link_count = pynvml.NVML_NVLINK_MAX_LINKS
        active_links = 0
        available_links = 0
        for link_idx in range(link_count):
            try:
                if pynvml.nvmlDeviceGetNvLinkCapability(
                    handle, link_idx, pynvml.NVML_NVLINK_CAP_P2P_SUPPORTED
                ):
                    available_links += 1
                    is_active = pynvml.nvmlDeviceGetNvLinkState(handle, link_idx)
                    if is_active:
                        active_links += 1
            except pynvml.NVMLError_NotSupported:
                continue
        return (
            active_links == available_links and available_links > 0
            if need_all_up
            else available_links > 0
        )

    @staticmethod
    def supports_mnnvl() -> bool:
        # TODO:
        # We check if it has all NVLink up now.
        # But it is not equivalent to MNNVL support.
        # May need better support check.
        support_nvlink_and_all_up = MnnvlMemory.support_nvlink(True)
        return support_nvlink_and_all_up


@dataclass
class MoEAlltoallInfo:
    local_gather_indices: torch.Tensor
    send_rank_count_cumsum: torch.Tensor
    send_rank_local_indices: torch.Tensor
    recv_rank_count_cumsum: torch.Tensor
    recv_rank_local_indices: torch.Tensor
    backward_recv_rank_local_indices: torch.Tensor
    local_token_allocation_count: int


class MnnvlMoe:
    moe_workspace: MnnvlMemory = None
    moe_prepare_workspace: MnnvlMemory = None
    moe_workspace_tensor: torch.Tensor = None
    moe_prepare_workspace_tensor: torch.Tensor = None
    moe_mapping: Mapping = None

    @staticmethod
    def get_moe_workspaces(mapping: Mapping):
        if MnnvlMoe.moe_workspace is not None:
            assert mapping == MnnvlMoe.moe_mapping, "only one moe mapping supported now"
            return MnnvlMoe.moe_workspace_tensor

        MnnvlMoe.moe_mapping = mapping
        workspace_size_per_rank = torch.ops.fireworks.get_moe_commworkspace_size_per_rank(
            mapping.tp_size
        )
        MnnvlMoe.moe_workspace = MnnvlMemory(mapping, workspace_size_per_rank)
        MnnvlMoe.moe_workspace_tensor = MnnvlMoe.moe_workspace.as_torch_strided_tensor(torch.uint64)
        torch.ops.fireworks.moe_initialize_workspace(
            MnnvlMoe.moe_workspace_tensor, mapping.tp_rank, mapping.tp_size
        )
        torch.cuda.synchronize()
        MnnvlMoe.moe_workspace.comm.barrier()
        return MnnvlMoe.moe_workspace_tensor

    @staticmethod
    def get_moe_prepare_workspace(mapping: Mapping):
        if MnnvlMoe.moe_prepare_workspace_tensor is not None:
            assert mapping == MnnvlMoe.moe_mapping, "only one moe mapping supported now"
            return MnnvlMoe.moe_prepare_workspace_tensor
        workspace_size_per_rank = torch.ops.fireworks.get_moe_prepare_workspace_size_per_rank(
            mapping.tp_size
        )
        MnnvlMoe.moe_prepare_workspace = MnnvlMemory(mapping, workspace_size_per_rank)
        MnnvlMoe.moe_prepare_workspace_tensor = (
            MnnvlMoe.moe_prepare_workspace.as_torch_strided_tensor(torch.uint64)
        )
        return MnnvlMoe.moe_prepare_workspace_tensor

    @staticmethod
    def compute_target_rank_id(
        token_selected_experts: torch.Tensor, expert_count: int, ep_size: int
    ):
        assert expert_count % ep_size == 0, "expert_count should be divisible by ep_size"
        expert_per_rank = expert_count // ep_size
        token_target_rank_ids = token_selected_experts // expert_per_rank
        return token_target_rank_ids

    @staticmethod
    def mnnvl_moe_alltoallv_prepare_without_allgather(
        expert_ids: torch.Tensor,
        expert_statics: Optional[torch.Tensor],
        workspace: torch.Tensor,
        max_token_count_per_rank: int,
        ep_rank: int,
        ep_size: int,
        expert_count: int,
        slot_count: int,
        top_k: int,
    ):
        (
            local_send_rank_count_cumsum,
            local_send_rank_indices,
            local_recv_rank_count_cumsum,
            local_recv_rank_indices,
            backward_local_recv_rank_indices,
            gathered_expert_statics,
        ) = torch.ops.fireworks.mnnvl_moe_alltoallv_prepare_without_allgather(
            expert_ids,
            expert_statics,
            workspace,
            max_token_count_per_rank,
            ep_rank,
            ep_size,
            expert_count,
            slot_count,
            top_k,
        )

        local_token_allocation_count = max_token_count_per_rank * ep_size
        # Looks like we don't need this.
        local_gather_indices = None

        alltoall_info = MoEAlltoallInfo(
            local_gather_indices,
            local_send_rank_count_cumsum,
            local_send_rank_indices,
            local_recv_rank_count_cumsum,
            local_recv_rank_indices,
            backward_local_recv_rank_indices,
            local_token_allocation_count,
        )

        return alltoall_info, gathered_expert_statics

    @staticmethod
    def mnnvl_moe_expert_static_allgather(
        expert_ids: torch.Tensor,
        workspace: torch.Tensor,
        ep_rank: int,
        ep_size: int,
        expert_count: int,
    ):
        # NOTE: Prefer Fireworks-native path when available; fall back to identity for single rank.
        if ep_size == 1:
            return expert_ids
        # If a native op exists later, swap to torch.ops.fireworks.mnnvl_moe_expert_static_allgather
        gathered_expert_ids = torch.ops.trtllm.mnnvl_moe_expert_static_allgather(
            expert_ids, workspace, ep_rank, ep_size, expert_count
        )
        return gathered_expert_ids

    @staticmethod
    def mnnvl_moe_alltoallv_prepare(
        gathered_target_rank_ids: torch.Tensor,
        real_rank_token_count_cumsum: Optional[torch.Tensor],
        gathered_expert_ids: torch.Tensor,
        gathered_scales: Optional[torch.Tensor],
        max_token_count_per_rank: int,
        expert_count: int,
        top_k: int,
        ep_rank: int,
        ep_size: int,
    ):
        (
            local_gather_indices,
            send_rank_count_cumsum,
            send_rank_local_indices,
            recv_rank_count_cumsum,
            recv_rank_local_indices,
            backward_recv_rank_local_indices,
        ) = torch.ops.fireworks.moe_comm_prepare_indices(
            gathered_target_rank_ids,
            real_rank_token_count_cumsum,
            max_token_count_per_rank,
            expert_count,
            top_k,
            ep_rank,
            ep_size,
        )

        local_token_allocation_count = max_token_count_per_rank * ep_size

        local_expert_ids = torch.empty(
            local_token_allocation_count, top_k, dtype=torch.int32, device=torch.device("cuda")
        )
        if gathered_scales is None:
            local_scales = None
        else:
            local_scales = torch.empty(
                local_token_allocation_count,
                top_k,
                dtype=torch.float32,
                device=torch.device("cuda"),
            )

        torch.ops.fireworks.moe_local_gather(
            recv_rank_count_cumsum,
            local_gather_indices,
            gathered_expert_ids,
            gathered_scales,
            local_expert_ids,
            local_scales,
            max_token_count_per_rank,
            expert_count,
            top_k,
            ep_rank,
            ep_size,
        )

        alltoall_info = MoEAlltoallInfo(
            local_gather_indices,
            send_rank_count_cumsum,
            send_rank_local_indices,
            recv_rank_count_cumsum,
            recv_rank_local_indices,
            backward_recv_rank_local_indices,
            local_token_allocation_count,
        )
        return alltoall_info, local_expert_ids, local_scales

    @staticmethod
    def mnnvl_moe_alltoallv(
        x: Union[torch.Tensor, List[Optional[torch.Tensor]]],
        alltoall_info: MoEAlltoallInfo,
        workspace: torch.Tensor,
        ep_rank: int,
        ep_size: int,
    ) -> Union[torch.Tensor, List[Optional[torch.Tensor]]]:
        # Convert single tensor to list for unified handling
        is_single_tensor = not isinstance(x, list)
        if is_single_tensor:
            assert x.dim() == 2, "only 2D tensor supported, please reshape."
            x = [x]

        assert len(x) > 0, "Empty tensor list not supported"

        # Filter out None values
        valid_list = [tensor is not None for tensor in x]
        valid_tensors = [tensor for tensor in x if tensor is not None]

        if len(valid_tensors) == 0:
            # All tensors are None, return list of None
            result = [None] * len(x)
        else:
            first_dim = None
            for tensor in valid_tensors:
                # Validate dimensions of valid tensors
                assert tensor.dim() == 2, "only 2D tensor supported, please reshape."
                if first_dim is None:
                    first_dim = tensor.shape[0]
                else:
                    assert tensor.shape[0] == first_dim, (
                        f"All tensors must have the same first dimension, got {tensor.shape[0]} vs {first_dim}"
                    )

            # Process only valid tensors
            output_tensors = torch.ops.trtllm.moe_comm(
                valid_tensors,
                alltoall_info.send_rank_count_cumsum,
                alltoall_info.send_rank_local_indices,
                alltoall_info.recv_rank_count_cumsum,
                alltoall_info.recv_rank_local_indices,
                workspace,
                alltoall_info.local_token_allocation_count,
                ep_rank,
                ep_size,
            )

            # Restore None positions in output
            idx = 0
            result = []
            for is_valid in valid_list:
                if is_valid:
                    result.append(output_tensors[idx])
                    idx += 1
                else:
                    result.append(None)

        # If input was a single tensor, return a single tensor
        if is_single_tensor:
            result = result[0]

        return result

    @staticmethod
    def mnnvl_moe_alltoallv_combine(
        x: torch.Tensor,
        alltoall_info: MoEAlltoallInfo,
        workspace: torch.Tensor,
        ep_rank: int,
        ep_size: int,
        top_k: int,
        token_count: int,
        use_low_precision_combine: bool = False,
        do_reduce: bool = True,
    ):
        assert x.dim() == 2, "2D tensor supported, please reshape."
        output_tensors = torch.ops.fireworks.moe_comm(
            [x],
            alltoall_info.recv_rank_count_cumsum,
            alltoall_info.recv_rank_local_indices,
            alltoall_info.send_rank_count_cumsum,
            alltoall_info.backward_recv_rank_local_indices,
            workspace,
            token_count * top_k,
            ep_rank,
            ep_size,
            [True],
            use_low_precision_combine,
        )
        output_tensor = output_tensors[0].reshape(token_count, top_k, x.shape[1])
        if do_reduce:
            return torch.sum(output_tensor, dim=1, keepdim=False)
        else:
            return output_tensor
