"""Private AOT-only TensorRT-LLM b552 runner for fused SiTU MoE."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from . import env as jit_env
from .core import JitSpec, current_compilation_context, gen_jit_spec
from .cpp_ext import get_cuda_path

MODULE_BASENAME = "fused_moe_trtllm_sm100_situ_b552_64ed071e"
# These two values are derived from the canonical native source and ABI
# contracts below.  The AOT build recomputes both and fails closed if either
# value is stale.  Keep the native digest in the module name so an older
# JIT-cache wheel cannot satisfy this spec merely because it used the same PR
# #2917 source snapshot.
NATIVE_SOURCE_MANIFEST_SHA256 = (
    "2f8d9bdf2439dc04bc39c5a51c6a375398ee8e4e5d3c53e44e08ce7058c041ac"
)
NATIVE_MODULE_ABI_SHA256 = (
    "4136c08c87d437e4c94367088ebd2c4ea485190eae0bfd854a0142893a2ff4a2"
)
MODULE_NAME = f"{MODULE_BASENAME}_{NATIVE_MODULE_ABI_SHA256[:12]}"
ARTIFACT_RELATIVE_ROOT = Path("fireworks/situ_b552/batched_gemm-b5521162-64ed071e")
SOURCE_COMMIT = "64ed071e23bf8d5d2d5af5c91577e5b8e036a1cf"
CUTLASS_COMMIT = "da5e086dab31d63815acafdac9a9c5893b1c69e2"
ROUTING_SOURCE_COMMIT = "57ba7eeb7ea3003a2d6ad5d9a057c4f952709bac"
SOURCE_MANIFEST_SHA256 = (
    "ed9de00c740fb1eb1caa8aa65ef88ecfcb08a3e19bd6983424979b1411d4aba1"
)
ROUTING_SOURCE_MANIFEST_SHA256 = (
    "8a6aadfccc8cd04a563cdb266c40844e0997f9fdc6de14587a22247b6298790d"
)
BASE_ARTIFACT_ROOT = (
    "b55211623be7f5697c5262ffd8361fc06c147bc9/batched_gemm-b3c1646-c111d7c/"
)
BASE_MANIFEST_SHA256 = (
    "0af823880730c4f0b3832d2208fab035946694b83444410b9309db5613d60195"
)
SEALED_MANIFEST_SHA256 = (
    "504ca05b32d75242df92cd2beffb559837d994f63825903bfea4472751c80350"
)
CUDA_TOOLCHAIN_IDENTITY = {
    "build": "cuda_13.2.r13.2/compiler.37668154_0",
    "release": "13.2",
    "version": "13.2.78",
}
CUBIN_COMPILE_CONTRACT = {
    # One SM100-family cubin bundle serves both B200 and the target B300 shape.
    # The native host runner still carries architecture-specific sm_100a and
    # sm_103a images for code that is not family-compatible.
    "architecture": "sm_100f",
    "cxx_standard": "c++17",
    "fast_math": True,
    "ndebug": True,
    "optimization": "O3",
    "output": "cubin",
    "random_seed": "sha256(source_basename)[:16]",
    "work_root": "/tmp/flashinfer-situ-b552-64ed071e-sm100f",
}
KERNEL_PARAMS_ABI = {
    "alignof": 128,
    "offsets": {
        "k": 856,
        "ptrClampLimit": 832,
        "ptrDynamicTileCounter": 17400,
        "ptrGatedActAlpha": 840,
        "ptrGatedActBeta": 848,
        "ptrPartialRowMax": 17384,
    },
    "sizeof": 17408,
}
ACTIVATION_ABI = {
    "runner_enum_name": "Swiglu",
    "runner_enum_value": 3,
    "generated_enum_name": "SwiGlu",
    "generated_enum_value": 0,
    "private_semantics": "situ",
}
NATIVE_RUNNER_ABI = "b552_situ_precomputed_ids_v1"
NATIVE_FFI_ABI = {
    "logits_moe": "trtllm_fp4_block_scale_situ_logits_moe",
    "pre_routed_moe": "trtllm_fp4_block_scale_situ_routed_moe",
    "valid_configs": "trtllm_get_valid_situ_moe_configs",
    "manifest_digest": "trtllm_situ_b552_manifest_digest",
    "native_abi_digest": "trtllm_situ_b552_native_abi_digest",
}
_SOURCE_BUILD_TARGETS = frozenset({(10, "0a"), (10, "3a")})
_SOURCE_BUILD_GENCODE_FLAGS = frozenset(
    {
        "-gencode=arch=compute_100a,code=sm_100a",
        "-gencode=arch=compute_103a,code=sm_103a",
    }
)
_SOURCE_BUILD_ENV_GUARDS = (
    "FLASHINFER_JIT_DEBUG",
    "FLASHINFER_JIT_VERBOSE",
    "FLASHINFER_JIT_LINEINFO",
)
_SOURCE_BUILD_EXTRA_FLAG_ENV = (
    "FLASHINFER_EXTRA_CFLAGS",
    "FLASHINFER_EXTRA_CUDAFLAGS",
)
_SOURCE_BUILD_CXX_COMMAND = "c++"
NATIVE_COMPILE_CONTRACT = {
    # Deliberately omit value-bearing manifest/native-digest defines: the
    # native ABI and cubin manifest are checked independently at runtime, and
    # including either value here would make the digests self-referential.
    "cuda_defines": [
        "CUTLASS_ENABLE_GDC_FOR_SM100=1",
        "ENABLE_BF16",
        "ENABLE_FP4",
        "ENABLE_FP8",
        "FLASHINFER_ENABLE_BF16",
        "FLASHINFER_ENABLE_F16",
        "FLASHINFER_ENABLE_FP4_E2M1",
        "FLASHINFER_ENABLE_FP8_E4M3",
        "FLASHINFER_ENABLE_FP8_E5M2",
        "FLASHINFER_ENABLE_FP8_E8M0",
        "TLLM_ENABLE_CUDA",
        "TLLM_GEN_EXPORT_FLASHINFER",
        "TLLM_GEN_EXPORT_INTERFACE",
    ],
    "cubin_artifact_root": ARTIFACT_RELATIVE_ROOT.as_posix(),
    "cxx_standard": "c++17",
    "cuda_targets": ["sm_100a", "sm_103a"],
    "cuda_toolchain": CUDA_TOOLCHAIN_IDENTITY,
    "debug": False,
    "fatbin_compression": "all",
    "host_visibility": "hidden",
    "lineinfo": False,
    "ndebug": True,
    "optimization": "O3",
    "runtime_mode": "aot_only",
    "use_fast_math": True,
}
B552_HEADER_SHA256 = {
    "include/trtllmGen_bmm_export/BatchedGemmInterface.h": (
        "868d59c26ca4b340c1ac7b9625856f44f12a19aaa9531695b19d192bd52301eb"
    ),
    "include/trtllmGen_bmm_export/GemmGatedActOptions.h": (
        "95e346f6ca7479e86c31d2d29008f75f9798f3194d0b5addb0c54aee023069ac"
    ),
    "include/trtllmGen_bmm_export/KernelParams.h": (
        "211801f8cede395409091d4050bf0fa26afbb0e503986fce1efff5ce27b0e7e3"
    ),
    "include/trtllmGen_bmm_export/KernelParamsDecl.h": (
        "413caec4ffaa3d3162049ee822e00cefd7eee385e1f8fc879d3ff683dcb11675"
    ),
}
PROVENANCE_SHA256 = {
    "provenance/source-files.sha256": SOURCE_MANIFEST_SHA256,
    "provenance/flashinfer-LICENSE": (
        "cb67c224f503e0a063908950b12f89a7280c6e527dcffac972aa114e4bf3c5de"
    ),
    "provenance/flashinfer-NOTICE": (
        "90bb9e1dec06f26a34f8f8ce98c53ded10b4fd8e1a9e91e2360e6be354e81db3"
    ),
}
_MANIFEST_DIGEST_CHUNK_WIDTHS = (13, 13, 13, 13, 12)
_FC1_CUBIN = re.compile(
    r"^Bmm_MxE4m3_MxE2m1MxE4m3_.*_bA32_bB32_bC32_.*"
    r"_situ_.*_sm100f\.cubin$"
)
_FC2_CUBIN = re.compile(r"^Bmm_Bfloat16_MxE2m1MxE4m3_Fp32_bA32_bB32_.*_sm100f\.cubin$")

ROUTING_SOURCE_SHA256 = {
    "csrc/trtllm_fused_moe_runner.cu": (
        "ac6056f88636bba0a4e575d7a6366261e3d5246fa23dd5ed4824e4fc49c7c7b2"
    ),
    "csrc/fused_moe/trtllm_backend/trtllm_fused_moe_routing_custom.cu": (
        "b2f31d8e91b398460170897df321e02b2e4a26b1d1c159d1fe864f54907a0e15"
    ),
    "csrc/fused_moe/trtllm_backend/trtllm_fused_moe_routing_common.cu": (
        "df7605a8e4274cf4e1786d931a48beab4fc928ade029222f56311dabe2e905c2"
    ),
    "csrc/fused_moe/trtllm_backend/trtllm_fused_moe_routing_deepseek.cu": (
        "cc25032b0df2c12319dfe5854a06d31532639515f625cbc37d727539e9e9da45"
    ),
    "csrc/fused_moe/trtllm_backend/trtllm_fused_moe_routing_llama4.cu": (
        "ff7a7963152a503315d66faccb98f5b70fee69e70659747419f5d6b3492dc42e"
    ),
    "include/flashinfer/trtllm/fused_moe/RoutingKernel.h": (
        "446c4cf377a1b629ddd7fcbd75c751a67dc35ca6babd4861b6f161b0a92f6622"
    ),
    "include/flashinfer/trtllm/fused_moe/RoutingKernel.cuh": (
        "b313d3f5e624a71336b3a25967208759026512341df2ff22d01ed3b87c2bcadb"
    ),
    "include/flashinfer/trtllm/fused_moe/RoutingKernelTopK.cuh": (
        "ea67761588f9cfd007f36267a73dbd5d1e731442cfd120543d9d1617b5bb0ab5"
    ),
    "include/flashinfer/trtllm/fused_moe/RoutingCustomPolicy.cuh": (
        "7826985aee55366ff441f9c5aac84c5299791235b3b962909ca32fc1037d01cf"
    ),
    "include/flashinfer/trtllm/fused_moe/RoutingDevKernel.h": (
        "7eb65d059dd9b0f27547f2b67e1f1587e6c80c3d1267571281244e357337a5ad"
    ),
}

_ROUTING_RUNTIME_SOURCES = tuple(
    relative
    for relative in ROUTING_SOURCE_SHA256
    if relative.startswith("csrc/fused_moe/")
)
_ROUTING_HEADERS = tuple(
    relative for relative in ROUTING_SOURCE_SHA256 if relative.startswith("include/")
)

_NATIVE_CRITICAL_PRIVATE_HEADERS = (
    "include/flashinfer/cubin_loader.h",
    "include/flashinfer/trtllm/common/cudaUtils.h",
    "include/flashinfer/trtllm/fused_moe/RoutingDevKernel.h",
    "include/flashinfer/trtllm/fused_moe/runner.h",
)

_B552_RUNTIME_SOURCES = (
    "csrc/nv_internal/cpp/common/envUtils.cpp",
    "csrc/nv_internal/cpp/common/logger.cpp",
    "csrc/nv_internal/cpp/common/stringUtils.cpp",
    "csrc/nv_internal/cpp/common/tllmException.cpp",
    "csrc/nv_internal/cpp/common/memoryUtils.cu",
    "csrc/trtllm_fused_moe_kernel_launcher.cu",
    "csrc/trtllm_fused_moe_runner.cu",
    "csrc/fused_moe/trtllm_backend/trtllm_fused_moe_dev_kernel.cu",
    "csrc/trtllm_batched_gemm_runner.cu",
)

_EXPORT_BLOCK = re.compile(
    r"TVM_FFI_DLL_EXPORT_TYPED_FUNC\(trtllm_bf16_moe, trtllm_bf16_moe\);\n"
    r"TVM_FFI_DLL_EXPORT_TYPED_FUNC\(trtllm_fp8_per_tensor_scale_moe, "
    r"trtllm_fp8_per_tensor_scale_moe\);\n"
    r"TVM_FFI_DLL_EXPORT_TYPED_FUNC\(trtllm_fp8_block_scale_moe, "
    r"trtllm_fp8_block_scale_moe\);\n"
    r"TVM_FFI_DLL_EXPORT_TYPED_FUNC\(trtllm_fp4_block_scale_moe, "
    r"trtllm_fp4_block_scale_moe\);\n"
    r"TVM_FFI_DLL_EXPORT_TYPED_FUNC\(trtllm_mxint4_block_scale_moe, "
    r"trtllm_mxint4_block_scale_moe\);\n"
    r"TVM_FFI_DLL_EXPORT_TYPED_FUNC\(trtllm_get_valid_moe_configs, "
    r"trtllm_get_valid_moe_configs\);"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _native_module_abi_contract() -> dict[str, object]:
    """Return the source-level ABI identity shared by all platform wheels."""
    return {
        "activation_abi": ACTIVATION_ABI,
        "compile_contract": NATIVE_COMPILE_CONTRACT,
        "cutlass_commit": CUTLASS_COMMIT,
        "kernel_params_abi": KERNEL_PARAMS_ABI,
        "module_basename": MODULE_BASENAME,
        "native_ffi_abi": NATIVE_FFI_ABI,
        "native_runner_abi": NATIVE_RUNNER_ABI,
        "native_source_manifest_sha256": NATIVE_SOURCE_MANIFEST_SHA256,
        "routing_source_manifest_sha256": ROUTING_SOURCE_MANIFEST_SHA256,
        "schema": 1,
        "source_commit": SOURCE_COMMIT,
    }


def _verify_native_module_abi_constant() -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", NATIVE_SOURCE_MANIFEST_SHA256):
        raise RuntimeError("Malformed SiTU b552 native source manifest digest")
    if not re.fullmatch(r"[0-9a-f]{64}", NATIVE_MODULE_ABI_SHA256):
        raise RuntimeError("Malformed SiTU b552 native module ABI digest")
    actual = _canonical_json_sha256(_native_module_abi_contract())
    if actual != NATIVE_MODULE_ABI_SHA256:
        raise RuntimeError(
            "SiTU b552 native module ABI constant is stale: "
            f"expected {NATIVE_MODULE_ABI_SHA256}, derived {actual}"
        )


def _native_source_manifest_digest(entries: dict[str, Path]) -> str:
    """Hash the exact native translation-unit and patched-header closure."""
    lines = []
    for logical_name, path in sorted(entries.items()):
        if not path.is_file():
            raise RuntimeError(
                f"SiTU b552 native source closure is missing {logical_name}: {path}"
            )
        lines.append(f"{_sha256(path)}  {logical_name}\n")
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def _routing_source_path(relative: str) -> Path:
    parts = Path(relative).parts
    if not parts:
        raise RuntimeError("Empty SiTU b552 routing source path")
    if parts[0] == "csrc":
        return jit_env.FLASHINFER_CSRC_DIR.joinpath(*parts[1:])
    if parts[0] == "include":
        return jit_env.FLASHINFER_INCLUDE_DIR.joinpath(*parts[1:])
    raise RuntimeError(f"Unexpected SiTU b552 routing source path: {relative}")


def _verify_current_routing_sources() -> dict[str, Path]:
    """Verify and resolve the exact current routing closure used by the AOT build."""
    resolved = {}
    manifest_lines = []
    for relative, expected in ROUTING_SOURCE_SHA256.items():
        path = _routing_source_path(relative)
        if not path.is_file():
            raise RuntimeError(f"SiTU b552 current routing source is missing: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(
                "SiTU b552 current routing source hash mismatch for "
                f"{relative}: expected {expected}, got {actual}"
            )
        resolved[relative] = path
        manifest_lines.append(f"{actual}  {relative}\n")

    actual_manifest = hashlib.sha256("".join(manifest_lines).encode()).hexdigest()
    if actual_manifest != ROUTING_SOURCE_MANIFEST_SHA256:
        raise RuntimeError(
            "SiTU b552 current routing manifest mismatch: "
            f"expected {ROUTING_SOURCE_MANIFEST_SHA256}, got {actual_manifest}"
        )
    return resolved


def _manifest_digest_chunks(digest: str) -> tuple[int, ...]:
    """Split a SHA-256 digest into signed-int64-safe FFI values."""
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise RuntimeError(f"Invalid SiTU b552 manifest digest: {digest!r}")
    chunks = []
    offset = 0
    for width in _MANIFEST_DIGEST_CHUNK_WIDTHS:
        chunks.append(int(digest[offset : offset + width], 16))
        offset += width
    if offset != len(digest) or any(chunk >= (1 << 63) for chunk in chunks):
        raise RuntimeError("SiTU b552 manifest digest cannot be represented safely")
    return tuple(chunks)


def _git_revision(root: Path, label: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"{label} is not an immutable Git checkout: {root}"
        ) from error


def _require_clean_git_revision(root: Path, expected: str, label: str) -> None:
    actual = _git_revision(root, label)
    if actual != expected:
        raise RuntimeError(
            f"{label} revision mismatch: expected {expected}, got {actual}"
        )
    try:
        status = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"{label} is not an immutable Git checkout: {root}"
        ) from error
    if status:
        raise RuntimeError(f"{label} checkout is not clean: {root}\n{status}")


def _cuda_tool_identity(tool: Path) -> dict[str, str]:
    try:
        output = subprocess.run(
            [str(tool), "--version"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"Cannot query CUDA tool identity: {tool}") from error
    version = re.search(
        r"Cuda compilation tools, release ([0-9.]+), V([0-9.]+)", output
    )
    build = re.search(r"^Build (cuda_[^\r\n]+)$", output, re.MULTILINE)
    if version is None or build is None:
        raise RuntimeError(f"Cannot parse CUDA tool identity from {tool}:\n{output}")
    return {
        "build": build.group(1),
        "release": version.group(1),
        "version": version.group(2),
    }


def _resolve_source_build_executable(command: str, label: str) -> Path:
    if not command or command != command.strip():
        raise RuntimeError(
            f"SiTU b552 AOT source build has an invalid {label} command: {command!r}"
        )
    if os.sep in command:
        resolved = Path(command)
    else:
        discovered = shutil.which(command)
        if discovered is None:
            raise RuntimeError(
                f"SiTU b552 AOT source build cannot find {label}: {command}"
            )
        resolved = Path(discovered)
    if not resolved.is_file():
        raise RuntimeError(
            f"SiTU b552 AOT source build cannot find {label}: {resolved}"
        )
    return resolved.resolve()


def _require_source_build_cuda_toolchain() -> dict[str, str]:
    cuda_home = Path(get_cuda_path())
    configured = os.environ.get("FLASHINFER_NVCC")
    nvcc = _resolve_source_build_executable(
        configured or str(cuda_home / "bin/nvcc"), "NVCC"
    )
    cuda_home_nvcc = (cuda_home / "bin/nvcc").resolve()
    if nvcc != cuda_home_nvcc:
        raise RuntimeError(
            "SiTU b552 AOT source build NVCC must resolve to the compiler under "
            f"CUDA_HOME: expected {cuda_home_nvcc}, got {nvcc}"
        )
    cxx = _resolve_source_build_executable(_SOURCE_BUILD_CXX_COMMAND, "CXX")
    actual = _cuda_tool_identity(nvcc)
    if actual != CUDA_TOOLCHAIN_IDENTITY:
        raise RuntimeError(
            "SiTU b552 AOT source build CUDA toolchain mismatch: "
            f"expected {CUDA_TOOLCHAIN_IDENTITY}, got {actual}"
        )
    # JitSpec binds both compilation and the non-device link rule to this
    # absolute CXX path. It also binds NVCC explicitly, so the identity checked
    # above is exactly the compiler ninja will execute.
    return {"cxx": str(cxx), "linker": str(cxx), "nvcc": str(nvcc)}


def _source_build_nvcc_flags() -> list[str]:
    enabled_guards = [
        name
        for name in _SOURCE_BUILD_ENV_GUARDS
        if os.environ.get(name) not in (None, "", "0")
    ]
    if enabled_guards:
        raise RuntimeError(
            "SiTU b552 AOT source builds require release-only JIT settings; "
            "disable: " + ", ".join(enabled_guards)
        )
    injected_flags = [
        name
        for name in _SOURCE_BUILD_EXTRA_FLAG_ENV
        if os.environ.get(name, "").strip()
    ]
    if injected_flags:
        raise RuntimeError(
            "SiTU b552 AOT source builds reject environment-injected compiler "
            "flags so the signed compile contract remains exact; unset: "
            + ", ".join(injected_flags)
        )

    filtered_targets = {
        target
        for target in current_compilation_context.TARGET_CUDA_ARCHS
        if target[0] == 10
    }
    if filtered_targets != _SOURCE_BUILD_TARGETS:
        raise RuntimeError(
            "SiTU b552 AOT source builds require exactly the filtered CUDA "
            "targets {(10, '0a'), (10, '3a')}; got "
            f"{sorted(filtered_targets)!r}"
        )
    flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[10]
    )
    actual_gencode = frozenset(flag for flag in flags if flag.startswith("-gencode="))
    if actual_gencode != _SOURCE_BUILD_GENCODE_FLAGS:
        raise RuntimeError(
            "SiTU b552 AOT source-build gencode mismatch: "
            f"expected {sorted(_SOURCE_BUILD_GENCODE_FLAGS)!r}, "
            f"got {sorted(actual_gencode)!r}"
        )
    return flags


def _verify_source_build_spec(spec: JitSpec, toolchain: dict[str, str]) -> None:
    host_flags = tuple(spec.extra_cflags or ())
    cuda_flags = tuple(spec.extra_cuda_cflags or ())
    required_host = {"-std=c++17", "-DNDEBUG", "-O3", "-fvisibility=hidden"}
    required_cuda = {
        "-std=c++17",
        "-DNDEBUG",
        "-O3",
        "-use_fast_math",
        "-Xfatbin=-compress-all",
        "-Xcompiler=-fvisibility=hidden",
    }
    missing_host = sorted(required_host.difference(host_flags))
    missing_cuda = sorted(required_cuda.difference(cuda_flags))
    if missing_host or missing_cuda:
        raise RuntimeError(
            "SiTU b552 AOT source-build release flag mismatch: "
            f"missing host={missing_host!r}, CUDA={missing_cuda!r}"
        )
    actual_gencode = frozenset(
        flag for flag in cuda_flags if flag.startswith("-gencode=")
    )
    if actual_gencode != _SOURCE_BUILD_GENCODE_FLAGS:
        raise RuntimeError(
            "SiTU b552 realized AOT target mismatch: "
            f"expected {sorted(_SOURCE_BUILD_GENCODE_FLAGS)!r}, "
            f"got {sorted(actual_gencode)!r}"
        )
    forbidden_exact = {"-g", "-G", "-O0", "--device-debug"}
    forbidden = [
        flag
        for flag in (*host_flags, *cuda_flags)
        if flag in forbidden_exact
        or "lineinfo" in flag.lower()
        or flag == "--ptxas-options=-v"
        or flag.startswith("-DCUTLASS_DEBUG_TRACE_LEVEL=")
    ]
    if forbidden:
        raise RuntimeError(
            "SiTU b552 AOT source builds forbid debug, verbose, and lineinfo "
            f"flags; got {forbidden!r}"
        )
    expected_commands = {
        "cxx": toolchain["cxx"],
        "nvcc": toolchain["nvcc"],
        "cxx_launcher": "",
        "nvcc_launcher": "",
        "use_environment_flags": False,
        "needs_device_linking": False,
        "extra_ldflags": None,
    }
    actual_commands = {name: getattr(spec, name, None) for name in expected_commands}
    if actual_commands != expected_commands or toolchain["linker"] != spec.cxx:
        raise RuntimeError(
            "SiTU b552 AOT source-build compiler/linker command mismatch: "
            f"expected {expected_commands!r} with linker={toolchain['linker']!r}, "
            f"got {actual_commands!r}"
        )


def get_situ_b552_artifact_root() -> Path:
    return jit_env.FLASHINFER_CUBIN_DIR / ARTIFACT_RELATIVE_ROOT


def verify_situ_b552_artifacts() -> str:
    """Verify the complete private bundle and return its manifest digest."""
    _verify_native_module_abi_constant()
    root = get_situ_b552_artifact_root()
    manifest_path = root / "checksums.txt"
    seal_path = root / "manifest.sha256"
    if not manifest_path.is_file() or not seal_path.is_file():
        raise RuntimeError(
            "The matched flashinfer-cubin wheel does not contain the private "
            f"SiTU b552 bundle at {root}"
        )
    fields = seal_path.read_text().strip().split()
    if (
        len(fields) != 2
        or not re.fullmatch(r"[0-9a-f]{64}", fields[0])
        or fields[1] != "checksums.txt"
    ):
        raise RuntimeError(f"Malformed SiTU b552 manifest seal: {seal_path}")
    expected_manifest = fields[0]
    if expected_manifest != SEALED_MANIFEST_SHA256:
        raise RuntimeError(
            "SiTU b552 sealed manifest digest mismatch: "
            f"expected {SEALED_MANIFEST_SHA256}, got {expected_manifest}"
        )
    actual_manifest = _sha256(manifest_path)
    if actual_manifest != expected_manifest:
        raise RuntimeError(
            f"SiTU b552 manifest mismatch: expected {expected_manifest}, got {actual_manifest}"
        )

    entries: dict[str, str] = {}
    for line_number, line in enumerate(manifest_path.read_text().splitlines(), 1):
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or not re.fullmatch(r"[0-9a-f]{64}", fields[0]):
            raise RuntimeError(
                f"Malformed SiTU b552 manifest entry on line {line_number}"
            )
        expected, relative = fields[0], fields[1].strip()
        relative_path = Path(relative)
        if (
            not relative
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or "\\" in relative
            or relative in entries
        ):
            raise RuntimeError(
                f"Unsafe or duplicate SiTU b552 manifest path: {relative!r}"
            )
        entries[relative] = expected
        artifact = root / relative_path
        if not artifact.is_file() or _sha256(artifact) != expected:
            raise RuntimeError(f"Missing or corrupt SiTU b552 artifact: {artifact}")

    cubins = [
        Path(relative).name for relative in entries if relative.endswith(".cubin")
    ]
    situ_count = sum(_FC1_CUBIN.fullmatch(name) is not None for name in cubins)
    fc2_count = sum(_FC2_CUBIN.fullmatch(name) is not None for name in cubins)
    if situ_count != 102 or fc2_count != 101 or len(cubins) != 203:
        raise RuntimeError(
            "SiTU b552 bundle must contain exactly 102 SiTU FC1 and 101 "
            "runner-selected BF16/MXFP4 FC2 cubins; "
            f"got {situ_count}, {fc2_count}, and {len(cubins)} total"
        )

    for relative, expected in PROVENANCE_SHA256.items():
        if entries.get(relative) != expected:
            raise RuntimeError(
                f"SiTU b552 provenance is missing or mismatched for {relative}"
            )

    contract_path = root / "contract.json"
    if "contract.json" not in entries:
        raise RuntimeError("SiTU b552 contract.json is not covered by checksums.txt")
    try:
        contract = json.loads(contract_path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Invalid SiTU b552 ABI contract: {contract_path}"
        ) from error
    if not isinstance(contract, dict):
        raise RuntimeError(f"Invalid SiTU b552 ABI contract: {contract_path}")

    expected_contract_fields = {
        "activation_abi": ACTIVATION_ABI,
        "artifact_root": ARTIFACT_RELATIVE_ROOT.as_posix(),
        "base_artifact_root": BASE_ARTIFACT_ROOT,
        "base_manifest_sha256": BASE_MANIFEST_SHA256,
        "cubin_compile_contract": CUBIN_COMPILE_CONTRACT,
        "cuda_arch": "sm_100f",
        "cuda_toolchain": CUDA_TOOLCHAIN_IDENTITY,
        "cutlass_commit": CUTLASS_COMMIT,
        "fc1_cubins": 102,
        "fc2_cubins": 101,
        "kernel_params_abi": KERNEL_PARAMS_ABI,
        "module_name": MODULE_NAME,
        "native_ffi_abi": NATIVE_FFI_ABI,
        "native_module_abi_sha256": NATIVE_MODULE_ABI_SHA256,
        "native_runner_abi": NATIVE_RUNNER_ABI,
        "native_source_manifest_sha256": NATIVE_SOURCE_MANIFEST_SHA256,
        "routing_source_commit": ROUTING_SOURCE_COMMIT,
        "routing_source_manifest_sha256": ROUTING_SOURCE_MANIFEST_SHA256,
        "source_commit": SOURCE_COMMIT,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
    }
    for field, expected_contract_value in expected_contract_fields.items():
        if contract.get(field) != expected_contract_value:
            raise RuntimeError(
                f"SiTU b552 ABI contract mismatch for {field}: "
                f"expected {expected_contract_value!r}, "
                f"got {contract.get(field)!r}"
            )

    header_sha256 = contract.get("header_sha256")
    if not isinstance(header_sha256, dict):
        raise RuntimeError("SiTU b552 ABI contract is missing header_sha256")
    expected_header_paths = set(B552_HEADER_SHA256) | {"include/flashinferMetaInfo.h"}
    if set(header_sha256) != expected_header_paths:
        raise RuntimeError(
            "SiTU b552 ABI contract has an unexpected generated-header set"
        )
    for relative, expected in header_sha256.items():
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise RuntimeError(f"Malformed SiTU b552 header digest for {relative}")
        if relative in B552_HEADER_SHA256 and expected != B552_HEADER_SHA256[relative]:
            raise RuntimeError(
                f"SiTU b552 generated-header ABI mismatch for {relative}"
            )
        if entries.get(relative) != expected:
            raise RuntimeError(
                f"SiTU b552 header digest is not manifest-bound for {relative}"
            )
    return actual_manifest


def _replace_exact(
    text: str, before: str, after: str, label: str, expected: int = 1
) -> str:
    count = text.count(before)
    if count != expected:
        raise RuntimeError(
            f"Expected {expected} exact PR #2917 {label} site(s), got {count}"
        )
    return text.replace(before, after)


def _write_text_if_different(destination: Path, text: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.read_text() == text:
        return
    destination.write_text(text)


def _sync_private_headers(source_root: Path, destination_root: Path) -> None:
    excluded = {
        Path("cubin_loader.h"),
        Path("trtllm/fused_moe/runner.h"),
    }
    generated_headers = Path("trtllm/batched_gemm/trtllmGen_bmm_export")
    for source in sorted(source_root.rglob("*")):
        relative = source.relative_to(source_root)
        if relative in excluded or generated_headers in (relative, *relative.parents):
            continue
        destination = destination_root / relative
        if source.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink():
            destination.unlink()
        if not destination.is_file() or source.read_bytes() != destination.read_bytes():
            shutil.copy2(source, destination)


def _sync_current_routing_headers(
    routing_sources: dict[str, Path], include_root: Path
) -> None:
    for relative in _ROUTING_HEADERS:
        source = routing_sources[relative]
        destination = include_root / Path(relative).relative_to("include")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink():
            destination.unlink()
        if not destination.is_file() or source.read_bytes() != destination.read_bytes():
            shutil.copy2(source, destination)


def _patch_cuda_utils_header(source: Path, destination: Path) -> None:
    """Add only the current-routing CUDA helpers missing from b552.

    The private overlay must retain b552's utility behavior for the old GEMM
    runner. Current routing additionally needs the CUDA FP8 declarations and
    ``getSMVersion``; inject those two compatibility pieces without replacing
    the pinned b552 header with its newer counterpart.
    """
    text = source.read_text()
    text = _replace_exact(
        text,
        "#include <driver_types.h>\n",
        "#include <driver_types.h>\n#ifdef ENABLE_FP8\n#include <cuda_fp8.h>\n#endif\n",
        "current-routing CUDA FP8 declarations",
    )
    get_sm_version = """inline int getSMVersion() {
  int device{-1};
  FLASHINFER_CHECK(cudaGetDevice(&device) == cudaSuccess, "cudaGetDevice failed");
  int sm_major = 0;
  int sm_minor = 0;
  FLASHINFER_CHECK(
      cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device) == cudaSuccess,
      "cudaDeviceGetAttribute(ComputeCapabilityMajor) failed");
  FLASHINFER_CHECK(
      cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, device) == cudaSuccess,
      "cudaDeviceGetAttribute(ComputeCapabilityMinor) failed");
  return sm_major * 10 + sm_minor;
}

"""
    text = _replace_exact(
        text,
        "inline int getMultiProcessorCount() {\n",
        get_sm_version + "inline int getMultiProcessorCount() {\n",
        "current-routing SM version helper",
    )
    _write_text_if_different(destination, text)


def _patch_routing_dev_kernel_header(source: Path, destination: Path) -> None:
    """Let current routing replace only b552's two routing launch macros."""
    text = source.read_text()
    for macro in (
        "LAUNCH_ROUTING_LLAMA4",
        "LAUNCH_ROUTING_WITH_NUM_EXPERTS_FORCE_FLOAT_INPUT",
    ):
        text = _replace_exact(
            text,
            f"#define {macro}",
            f"#undef {macro}\n#define {macro}",
            f"{macro} b552 compatibility undef",
        )
    _write_text_if_different(destination, text)


def _patch_runner_header(source: Path, destination: Path) -> None:
    text = source.read_text()
    text = _replace_exact(
        text,
        "  // TopK only (no softmax)\n"
        "  TopK = 5,\n"
        "  // Unspecified\n"
        "  Unspecified = 6,",
        "  // TopK only (no softmax)\n"
        "  TopK = 5,\n"
        "  // SigmoidRenorm: Sigmoid -> TopK -> Renormalize "
        "(divide by sum of top-K weights)\n"
        "  SigmoidRenorm = 6,\n"
        "  // MiniMax2: Sigmoid + Bias -> TopK -> ScaledSumNormalize "
        "(routeScale=1.0, epsilon=1e-20)\n"
        "  MiniMax2 = 7,\n"
        "  // Sigmoid: Sigmoid -> TopK (no renormalization)\n"
        "  Sigmoid = 8,\n"
        "  // Unspecified\n"
        "  Unspecified = 9,",
        "current routing method enum values",
    )
    text = _replace_exact(
        text,
        '    case RoutingMethodType::TopK:\n      return "TopK";\n    default:',
        "    case RoutingMethodType::TopK:\n"
        '      return "TopK";\n'
        "    case RoutingMethodType::SigmoidRenorm:\n"
        '      return "SigmoidRenorm";\n'
        "    case RoutingMethodType::MiniMax2:\n"
        '      return "MiniMax2";\n'
        "    case RoutingMethodType::Sigmoid:\n"
        '      return "Sigmoid";\n'
        "    default:",
        "current routing method serialization",
    )
    text = _replace_exact(
        text,
        "           bool useDeepSeekFp8, RoutingMethodType routingMethodType, "
        "cudaStream_t stream);",
        "           bool useDeepSeekFp8, RoutingMethodType routingMethodType, "
        "cudaStream_t stream,\n"
        "           int32_t* expertIds);",
        "routing runner declaration",
    )
    _write_text_if_different(destination, text)


def _patch_cubin_loader_exports(source: Path, destination: Path) -> None:
    text = source.read_text()
    for symbol in ("FlashInferSetCubinCallback", "FlashInferSetCurrentCubin"):
        text = _replace_exact(
            text,
            f'extern "C" void {symbol}',
            f'extern "C" __attribute__((visibility("default"))) void {symbol}',
            f"{symbol} default visibility",
        )
    _write_text_if_different(destination, text)


def _routing_namespace_bounds(text: str) -> tuple[int, int]:
    start = text.index("namespace Routing {")
    marker = "}  // namespace Routing"
    end = text.index(marker, start) + len(marker)
    return start, end


def _runner_run_signature(text: str, namespace_start: int) -> str:
    start = text.index("void Runner::run(", namespace_start)
    end = text.index(") {", start) + len(") {")
    return text[start:end]


def _patch_runner_source(source: Path, routing_source: Path, destination: Path) -> None:
    """Keep old b552 GEMM code while replacing only its Routing namespace."""
    text = source.read_text()
    routing_text = routing_source.read_text()
    old_start, old_end = _routing_namespace_bounds(text)
    current_start, current_end = _routing_namespace_bounds(routing_text)

    old_signature = _runner_run_signature(text, old_start)
    private_signature = _replace_exact(
        old_signature,
        "RoutingMethodType routingMethodType, cudaStream_t stream) {",
        "RoutingMethodType routingMethodType, cudaStream_t stream, "
        "int32_t* expertIds) {",
        "routing runner definition ABI",
    )
    current_signature = _runner_run_signature(routing_text, current_start)
    routing = routing_text[current_start:current_end]
    routing = _replace_exact(
        routing,
        current_signature,
        private_signature
        + "\n"
        + "  auto const dtypeLogits = dtypeScore;\n"
        + "  bool const normTopkProb = true;\n"
        + "  int16_t* const routing_replay_out = nullptr;",
        "current routing namespace ABI adaptation",
    )
    _write_text_if_different(destination, text[:old_start] + routing + text[old_end:])


def _patch_deepseek_logits_dtype(text: str) -> str:
    return _replace_exact(
        text,
        "    if (routing_logits.has_value()) {\n"
        "      if (static_cast<RoutingMethodType>(routing_method_type) == "
        "RoutingMethodType::DeepSeekV3) {\n"
        "        TVM_FFI_ICHECK_EQ(routing_logits.value().dtype(), dl_float32)\n"
        '            << "routing_logits must be float.";\n'
        "        mDtypeScore = btg::Dtype::Fp32;\n"
        "      } else if (routing_logits.value().dtype() == dl_float32) {",
        "    if (routing_logits.has_value()) {\n"
        "      if (routing_logits.value().dtype() == dl_float32) {",
        "obsolete DeepSeek float-only routing logits check",
    )


def _patch_non_fp4_routing_calls(text: str) -> str:
    start = text.index("class FP4BlockScaleLauncher")
    prefix = text[:start]
    suffix = text[start:]
    prefix = _replace_exact(
        prefix,
        "        static_cast<RoutingMethodType>(routing_method_type), routing_stream);",
        "        static_cast<RoutingMethodType>(routing_method_type), "
        "routing_stream, nullptr);",
        "non-FP4 routing runner call",
        expected=2,
    )
    return prefix + suffix


def _patch_kernel_params_abi_assertions(text: str) -> str:
    text = _replace_exact(
        text,
        "#include <cmath>\n",
        "#include <cmath>\n#include <cstddef>\n",
        "KernelParams offsetof include",
    )
    text = _replace_exact(
        text,
        '#include "flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/'
        'GemmGatedActOptions.h"\n',
        '#include "flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/'
        'GemmGatedActOptions.h"\n'
        '#include "flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/'
        'KernelParamsDecl.h"\n',
        "private KernelParams declaration include",
    )
    assertions = """static_assert(sizeof(::batchedGemm::KernelParams) == 17408,
              "SiTU b552 KernelParams sizeof mismatch");
static_assert(alignof(::batchedGemm::KernelParams) == 128,
              "SiTU b552 KernelParams alignment mismatch");
static_assert(offsetof(::batchedGemm::KernelParams, ptrClampLimit) == 832,
              "SiTU b552 ptrClampLimit offset mismatch");
static_assert(offsetof(::batchedGemm::KernelParams, ptrGatedActAlpha) == 840,
              "SiTU b552 ptrGatedActAlpha offset mismatch");
static_assert(offsetof(::batchedGemm::KernelParams, ptrGatedActBeta) == 848,
              "SiTU b552 ptrGatedActBeta offset mismatch");
static_assert(offsetof(::batchedGemm::KernelParams, k) == 856,
              "SiTU b552 k offset mismatch");
static_assert(offsetof(::batchedGemm::KernelParams, ptrPartialRowMax) == 17384,
              "SiTU b552 ptrPartialRowMax offset mismatch");
static_assert(offsetof(::batchedGemm::KernelParams, ptrDynamicTileCounter) == 17400,
              "SiTU b552 ptrDynamicTileCounter offset mismatch");
static_assert(
    static_cast<int64_t>(
        ::tensorrt_llm::kernels::trtllmgen_moe::MoE::ActivationType::Swiglu) == 3,
    "SiTU b552 runner activation enum mismatch");
static_assert(static_cast<int64_t>(::batchedGemm::gemmGatedAct::ActType::SwiGlu) == 0,
              "SiTU b552 generated activation enum mismatch");

"""
    return _replace_exact(
        text,
        "namespace flashinfer {\n",
        assertions + "namespace flashinfer {\n",
        "private KernelParams ABI assertions",
    )


def _patch_fp4_launcher(text: str) -> str:
    start = text.index("class FP4BlockScaleLauncher")
    end = text.index("Array<Tensor> trtllm_bf16_moe", start)
    fp4 = text[start:end]
    fp4 = _replace_exact(
        fp4,
        "      TensorView const& expert_weights)",
        "      Tensor const& expert_weights)",
        "FP4 owning expert weights argument",
    )
    fp4 = _replace_exact(
        fp4,
        "        expert_weights(expert_weights) {}",
        "        routing_expert_weights(expert_weights) {}",
        "FP4 owning expert weights initialization",
    )
    fp4 = _replace_exact(
        fp4,
        "    auto routing_bias_dtype = routing_bias.has_value() ? "
        "routing_bias.value().dtype() : dl_bfloat16;\n"
        "    mRoutingBiasDtype = routing_bias_dtype == dl_bfloat16 ? "
        "btg::Dtype::Bfloat16 : btg::Dtype::Fp32;",
        "    auto routing_bias_dtype = routing_bias.has_value() ? "
        "routing_bias.value().dtype() : dl_bfloat16;\n"
        "    mRoutingBiasDtype = routing_bias_dtype == dl_bfloat16 ? "
        "btg::Dtype::Bfloat16 : btg::Dtype::Fp32;\n"
        "    if (routing_logits.has_value()) {\n"
        "      mDtypeScore = routing_logits.value().dtype() == dl_float32\n"
        "                        ? btg::Dtype::Fp32\n"
        "                        : btg::Dtype::Bfloat16;\n"
        "    }",
        "FP4 routing logits dtype propagation",
    )
    fp4 = _replace_exact(
        fp4,
        "    workspace.routing_expert_indexes =\n"
        "        static_cast<int*>(const_cast<void*>(expert_indices.data_ptr()));\n"
        "    workspace.expert_weights = const_cast<void*>(expert_weights.data_ptr());",
        "    if (args->routing_logits == nullptr) {\n"
        "      // Pre-routed IDs are read-only input. Keep a separate packed output\n"
        "      // buffer so the caller's IDs are never overwritten in place.\n"
        "      FusedMoeLauncher::expert_indexes = alloc_tensor(\n"
        "          {args->num_tokens, args->top_k}, dl_int32, hidden_states.device());\n"
        "      workspace.routing_expert_indexes =\n"
        "          static_cast<int*>(FusedMoeLauncher::expert_indexes.data_ptr());\n"
        "    } else {\n"
        "      workspace.routing_expert_indexes =\n"
        "          static_cast<int*>(const_cast<void*>(expert_indices.data_ptr()));\n"
        "    }\n"
        "    workspace.expert_weights = routing_expert_weights.data_ptr();",
        "FP4 private routing workspace",
    )
    fp4 = _replace_exact(
        fp4,
        "    cudaStream_t routing_stream = get_stream(hidden_states.device());\n\n"
        "    routing_runner.run(",
        "    cudaStream_t routing_stream = get_stream(hidden_states.device());\n"
        "    int32_t* expert_ids =\n"
        "        args->routing_logits == nullptr\n"
        "            ? static_cast<int32_t*>(const_cast<void*>(expert_indices.data_ptr()))\n"
        "            : nullptr;\n\n"
        "    routing_runner.run(",
        "FP4 precomputed input selection",
    )
    fp4 = _replace_exact(
        fp4,
        "        args->routed_scaling_factor, static_cast<int*>(expert_indices.data_ptr()),",
        "        args->routed_scaling_factor, workspace.routing_expert_indexes,",
        "FP4 packed routing output",
    )
    fp4 = _replace_exact(
        fp4,
        "        static_cast<RoutingMethodType>(routing_method_type), routing_stream);",
        "        static_cast<RoutingMethodType>(routing_method_type), "
        "routing_stream, expert_ids);",
        "FP4 unpacked routing input",
    )
    fp4 = _replace_exact(
        fp4,
        "  TensorView expert_weights;",
        "  // Keep an owning reference so the raw-FC2 return never exposes the\n"
        "  // default-constructed base tensor from immutable b552.\n"
        "  Tensor routing_expert_weights;",
        "FP4 owning expert weights member",
    )
    fp4 = _replace_exact(
        fp4,
        "        static_cast<int*>(permuted_idx_to_token_idx.data_ptr()), "
        "expert_weights.data_ptr(),",
        "        static_cast<int*>(permuted_idx_to_token_idx.data_ptr()), "
        "routing_expert_weights.data_ptr(),",
        "FP4 owning expert weights routing input",
    )
    fp4 = _replace_exact(
        fp4,
        "    return {gemm2_output, FusedMoeLauncher::expert_weights, "
        "expanded_idx_to_permuted_idx};",
        "    return {gemm2_output, routing_expert_weights, expanded_idx_to_permuted_idx};",
        "FP4 raw-FC2 owning expert weights return",
    )
    text = text[:start] + fp4 + text[end:]
    return _replace_exact(
        text,
        "Array<Tensor> trtllm_fp4_block_scale_moe(\n"
        "    Optional<TensorView> routing_logits, TensorView expert_indices, "
        "TensorView expert_weights,",
        "Array<Tensor> trtllm_fp4_block_scale_moe(\n"
        "    Optional<TensorView> routing_logits, TensorView expert_indices, "
        "Tensor expert_weights,",
        "FP4 exported owning expert weights ABI",
    )


def _prepare_include_overlay(
    generated_dir: Path,
    artifact_root: Path,
    source_root: Path,
    routing_sources: dict[str, Path],
) -> Path:
    include_root = generated_dir / "private_include"
    # This directory is an input-precedence overlay.  Recreate it rather than
    # incrementally syncing so a removed or manually injected header can never
    # shadow the immutable source roots in a later AOT build.
    if include_root.is_symlink() or include_root.is_file():
        include_root.unlink()
    elif include_root.exists():
        shutil.rmtree(include_root)
    flashinfer_source = source_root / "include/flashinfer"
    flashinfer_overlay = include_root / "flashinfer"
    trtllm_source = flashinfer_source / "trtllm"
    trtllm_overlay = flashinfer_overlay / "trtllm"
    fused_moe_source = trtllm_source / "fused_moe"
    fused_moe_overlay = trtllm_overlay / "fused_moe"
    # runner.h uses quoted sibling includes, and DevKernel.h reaches ../../ to
    # shared FlashInfer headers. Copy the immutable header subtree so all relative
    # includes resolve inside the private namespace before replacing only the
    # runner ABI and generated-header directory.  Then replace only the pinned
    # current-routing closure; DevKernel.h and every GEMM header stay at b552.
    _sync_private_headers(flashinfer_source, flashinfer_overlay)
    _sync_current_routing_headers(routing_sources, include_root)
    _patch_cuda_utils_header(
        trtllm_source / "common/cudaUtils.h",
        trtllm_overlay / "common/cudaUtils.h",
    )
    _patch_routing_dev_kernel_header(
        routing_sources["include/flashinfer/trtllm/fused_moe/RoutingDevKernel.h"],
        fused_moe_overlay / "RoutingDevKernel.h",
    )
    _patch_cubin_loader_exports(
        flashinfer_source / "cubin_loader.h",
        flashinfer_overlay / "cubin_loader.h",
    )
    link = include_root / "flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export"
    target = artifact_root / "include/trtllmGen_bmm_export"
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() != target.resolve():
            link.unlink()
    elif link.is_file():
        link.unlink()
    elif link.exists():
        shutil.rmtree(link)
    if not link.is_symlink():
        link.symlink_to(target, target_is_directory=True)
    _patch_runner_header(
        fused_moe_source / "runner.h",
        fused_moe_overlay / "runner.h",
    )
    return include_root


def _patch_launcher(source: Path, destination: Path) -> None:
    text = _patch_kernel_params_abi_assertions(source.read_text())
    text = _patch_deepseek_logits_dtype(text)
    text = _patch_non_fp4_routing_calls(text)
    text = _patch_fp4_launcher(text)
    replacement = """Array<int64_t> trtllm_situ_b552_manifest_digest() {
  Array<int64_t> digest;
  digest.push_back(FLASHINFER_SITU_B552_MANIFEST_CHUNK_0);
  digest.push_back(FLASHINFER_SITU_B552_MANIFEST_CHUNK_1);
  digest.push_back(FLASHINFER_SITU_B552_MANIFEST_CHUNK_2);
  digest.push_back(FLASHINFER_SITU_B552_MANIFEST_CHUNK_3);
  digest.push_back(FLASHINFER_SITU_B552_MANIFEST_CHUNK_4);
  return digest;
}

Array<int64_t> trtllm_situ_b552_native_abi_digest() {
  Array<int64_t> digest;
  digest.push_back(FLASHINFER_SITU_B552_NATIVE_ABI_CHUNK_0);
  digest.push_back(FLASHINFER_SITU_B552_NATIVE_ABI_CHUNK_1);
  digest.push_back(FLASHINFER_SITU_B552_NATIVE_ABI_CHUNK_2);
  digest.push_back(FLASHINFER_SITU_B552_NATIVE_ABI_CHUNK_3);
  digest.push_back(FLASHINFER_SITU_B552_NATIVE_ABI_CHUNK_4);
  return digest;
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_fp4_block_scale_situ_logits_moe,
                              trtllm_fp4_block_scale_moe);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_fp4_block_scale_situ_routed_moe,
                              trtllm_fp4_block_scale_moe);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_get_valid_situ_moe_configs,
                              trtllm_get_valid_moe_configs);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_situ_b552_manifest_digest,
                              trtllm_situ_b552_manifest_digest);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_situ_b552_native_abi_digest,
                              trtllm_situ_b552_native_abi_digest);"""
    text, replacements = _EXPORT_BLOCK.subn(replacement, text)
    if replacements != 1:
        raise RuntimeError(
            f"Expected one matching PR #2917 export block in {source}, got {replacements}"
        )
    _write_text_if_different(destination, text)


def gen_trtllm_gen_fused_moe_situ_b552_module() -> JitSpec:
    manifest = verify_situ_b552_artifacts()
    artifact_root = get_situ_b552_artifact_root()
    source_root_value = os.environ.get("FLASHINFER_SITU_B552_SOURCE_ROOT")
    generated_dir = jit_env.FLASHINFER_GEN_SRC_DIR / MODULE_NAME

    toolchain = None
    if source_root_value:
        toolchain = _require_source_build_cuda_toolchain()
        nvcc_flags = _source_build_nvcc_flags()
        source_root = Path(source_root_value)
        routing_sources = _verify_current_routing_sources()
        _require_clean_git_revision(
            source_root, SOURCE_COMMIT, "FlashInfer PR #2917 source root"
        )
        cutlass_root_value = os.environ.get("FLASHINFER_SITU_B552_CUTLASS_ROOT")
        if not cutlass_root_value:
            raise RuntimeError(
                "Building the SiTU b552 AOT module requires "
                "FLASHINFER_SITU_B552_CUTLASS_ROOT"
            )
        cutlass_root = Path(cutlass_root_value)
        _require_clean_git_revision(cutlass_root, CUTLASS_COMMIT, "CUTLASS root")
        missing = [
            path for path in _B552_RUNTIME_SOURCES if not (source_root / path).is_file()
        ]
        if missing:
            raise RuntimeError(
                "The PR #2917 runtime source snapshot is incomplete; missing: "
                + ", ".join(missing)
            )
        launcher = generated_dir / "trtllm_fused_moe_kernel_launcher_situ_b552.cu"
        runner = generated_dir / "trtllm_fused_moe_runner_situ_b552.cu"
        _patch_launcher(
            source_root / "csrc/trtllm_fused_moe_kernel_launcher.cu", launcher
        )
        _patch_runner_source(
            source_root / "csrc/trtllm_fused_moe_runner.cu",
            routing_sources["csrc/trtllm_fused_moe_runner.cu"],
            runner,
        )
        patched_sources = {
            "csrc/trtllm_fused_moe_kernel_launcher.cu": launcher,
            "csrc/trtllm_fused_moe_runner.cu": runner,
        }
        native_source_entries = {
            f"translation_units/{path}": patched_sources.get(path, source_root / path)
            for path in _B552_RUNTIME_SOURCES
        }
        native_source_entries.update(
            {
                f"translation_units/{path}": routing_sources[path]
                for path in _ROUTING_RUNTIME_SOURCES
            }
        )
        sources = list(native_source_entries.values())
        private_include = _prepare_include_overlay(
            generated_dir, artifact_root, source_root, routing_sources
        )
        native_source_entries.update(
            {
                f"private_headers/{relative}": private_include
                / Path(relative).relative_to("include")
                for relative in _NATIVE_CRITICAL_PRIVATE_HEADERS
            }
        )
        native_source_manifest = _native_source_manifest_digest(native_source_entries)
        if native_source_manifest != NATIVE_SOURCE_MANIFEST_SHA256:
            raise RuntimeError(
                "SiTU b552 native source manifest mismatch: "
                f"expected {NATIVE_SOURCE_MANIFEST_SHA256}, derived "
                f"{native_source_manifest}"
            )
        include_paths: list[str | Path] = [
            private_include,
            artifact_root / "include",
            source_root / "include",
            # The launcher is copied into FLASHINFER_GEN_SRC_DIR before its
            # exports are renamed, so quoted includes such as tvm_ffi_utils.h
            # no longer resolve relative to PR #2917's csrc directory.
            source_root / "csrc",
            source_root / "csrc/nv_internal",
            source_root / "csrc/nv_internal/include",
            cutlass_root / "include",
            cutlass_root / "tools/util/include",
        ]
    else:
        # Source paths are not consulted when the matched AOT module exists.
        # If it is absent, build_and_load fails closed below instead of falling
        # back to the stock d2c runner or attempting a network/JIT build.
        sources = [generated_dir / "aot_only_situ_b552.cu"]
        include_paths = [artifact_root / "include"]
        # A runtime host only needs to discover its matching prebuilt image;
        # do not require both build-time targets on a single-device B200 or B300.
        nvcc_flags = current_compilation_context.get_nvcc_flags_list(
            supported_major_versions=[10]
        )

    manifest_chunks = _manifest_digest_chunks(manifest)
    manifest_defines = [
        f"-DFLASHINFER_SITU_B552_MANIFEST_CHUNK_{index}=0x{chunk:x}LL"
        for index, chunk in enumerate(manifest_chunks)
    ]
    native_abi_chunks = _manifest_digest_chunks(NATIVE_MODULE_ABI_SHA256)
    native_abi_defines = [
        f"-DFLASHINFER_SITU_B552_NATIVE_ABI_CHUNK_{index}=0x{chunk:x}LL"
        for index, chunk in enumerate(native_abi_chunks)
    ]
    spec = gen_jit_spec(
        MODULE_NAME,
        sources,
        extra_cuda_cflags=[
            "-DTLLM_GEN_EXPORT_INTERFACE",
            "-DTLLM_GEN_EXPORT_FLASHINFER",
            "-DTLLM_ENABLE_CUDA",
            "-DENABLE_BF16",
            "-DENABLE_FP8",
            "-DENABLE_FP4",
            "-DCUTLASS_ENABLE_GDC_FOR_SM100=1",
            "-Xcompiler=-fvisibility=hidden",
            *manifest_defines,
            *native_abi_defines,
            f'-DTLLM_GEN_GEMM_CUBIN_PATH=\\"{ARTIFACT_RELATIVE_ROOT.as_posix()}/\\"',
            *nvcc_flags,
        ],
        extra_cflags=["-fvisibility=hidden"],
        extra_include_paths=include_paths,
        cxx=toolchain["cxx"] if toolchain is not None else None,
        nvcc=toolchain["nvcc"] if toolchain is not None else None,
        cxx_launcher="" if toolchain is not None else None,
        nvcc_launcher="" if toolchain is not None else None,
        use_environment_flags=toolchain is None,
    )
    if source_root_value:
        assert toolchain is not None
        _verify_source_build_spec(spec, toolchain)
    return spec


def load_trtllm_gen_fused_moe_situ_b552_module():
    spec = gen_trtllm_gen_fused_moe_situ_b552_module()
    if not spec.is_aot and not os.environ.get("FLASHINFER_SITU_B552_SOURCE_ROOT"):
        raise RuntimeError(
            f"Required AOT module {MODULE_NAME}.so is missing from the matched "
            "flashinfer-jit-cache wheel; runtime JIT fallback is disabled for SiTU b552"
        )
    module = spec.build_and_load()
    expected_native_abi = _manifest_digest_chunks(NATIVE_MODULE_ABI_SHA256)
    actual_native_abi = tuple(
        int(value) for value in module.trtllm_situ_b552_native_abi_digest()
    )
    if actual_native_abi != expected_native_abi:
        raise RuntimeError(
            "SiTU b552 JIT-cache/native module ABI mismatch: "
            f"{actual_native_abi!r} != {expected_native_abi!r}"
        )
    expected_digest = _manifest_digest_chunks(verify_situ_b552_artifacts())
    actual_digest = tuple(
        int(value) for value in module.trtllm_situ_b552_manifest_digest()
    )
    if actual_digest != expected_digest:
        raise RuntimeError(
            "SiTU b552 JIT-cache/cubin manifest mismatch: "
            f"{actual_digest!r} != {expected_digest!r}"
        )
    missing_exports = [
        export_name
        for export_name in NATIVE_FFI_ABI.values()
        if not hasattr(module, export_name)
    ]
    if missing_exports:
        raise RuntimeError(
            "SiTU b552 native module is missing contract-bound FFI exports: "
            + ", ".join(sorted(missing_exports))
        )
    return spec, module
