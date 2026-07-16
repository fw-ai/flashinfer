#!/usr/bin/env python3
"""Build the private MXFP8/MXFP4 TensorRT-LLM SiTU cubin bundle.

The generated TensorRT-LLM sources are intentionally not mixed with the stock
FlashInfer d2c bundle.  This tool consumes the vendored, hash-pinned source
pairs from FlashInfer PR #2917, rewrites only the SwiGLU epilogue, compiles the
102 affected FC1 kernels, and combines them with the matching b552 FC2 kernels
and vendored generated headers under a private artifact root.

The source checkout and NVCC are build-time inputs.  Neither is needed by the
runtime wheel.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, suppress
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
from urllib.request import urlopen

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
VENDORED_SOURCE_ROOT = REPOSITORY_ROOT / "3rdparty/situ_b552/pr2917"
VENDORED_B552_ROOT = REPOSITORY_ROOT / "3rdparty/situ_b552/b552"

SOURCE_COMMIT = "64ed071e23bf8d5d2d5af5c91577e5b8e036a1cf"
CUTLASS_COMMIT = "da5e086dab31d63815acafdac9a9c5893b1c69e2"
SOURCE_RELATIVE_DIR = Path("csrc/trtllm_kernels/batched_gemm/trtllmGen_bmm_export/src")
SOURCE_MANIFEST_SHA256 = (
    "ed9de00c740fb1eb1caa8aa65ef88ecfcb08a3e19bd6983424979b1411d4aba1"
)
SOURCE_LICENSE_SHA256 = (
    "cb67c224f503e0a063908950b12f89a7280c6e527dcffac972aa114e4bf3c5de"
)
SOURCE_NOTICE_SHA256 = (
    "90bb9e1dec06f26a34f8f8ce98c53ded10b4fd8e1a9e91e2360e6be354e81db3"
)
# The private AOT module deliberately combines the immutable b552 GEMM source
# with the current FlashInfer routing implementation that contains the proven
# K3 896-expert/top-16 policy.  Bind that exact source closure into the cubin
# contract so mismatched AOT and cubin wheels fail closed.
ROUTING_SOURCE_COMMIT = "57ba7eeb7ea3003a2d6ad5d9a057c4f952709bac"
ROUTING_SOURCE_MANIFEST_SHA256 = (
    "8a6aadfccc8cd04a563cdb266c40844e0997f9fdc6de14587a22247b6298790d"
)
EXPECTED_SOURCE_FILES = 204
EXPECTED_CUDA_SOURCES = 102
EXPECTED_FC2_CUBINS = 101
EXPECTED_EPILOGUES = 704
EXPECTED_B552_HEADERS = 18
B552_GENERATED_HEADERS_MANIFEST_SHA256 = (
    "82d97df674784b3632463fbfc5935a80b74bc9a50d2f8aaebb5b7c81ae2c99b6"
)
CUDA_TOOLCHAIN_IDENTITY = {
    # NGC 26.05 is the Fireworks dependency builder for both x86_64 and
    # aarch64.  Cubins are sealed byte-for-byte, so accept only its exact
    # device compiler instead of producing architecture-specific wheel drift.
    "build": "cuda_13.2.r13.2/compiler.37668154_0",
    "release": "13.2",
    "version": "13.2.78",
}
CUBIN_COMPILE_CONTRACT = {
    # The rollout shape is B300/SM103.  The immutable b552 metadata is
    # SM100-family metadata, so sm_100f is intentional: sm_100a would make the
    # same sealed cubins B200-only and unusable by the target deployment.
    "architecture": "sm_100f",
    "cxx_standard": "c++17",
    "fast_math": True,
    "ndebug": True,
    "optimization": "O3",
    "output": "cubin",
    "random_seed": "sha256(source_basename)[:16]",
    "work_root": "/tmp/flashinfer-situ-b552-64ed071e-sm100f",
}
DETERMINISTIC_WORK_ROOT = Path(CUBIN_COMPILE_CONTRACT["work_root"])
_WORK_ROOT_LOCK = ".build.lock"

BASE_ARTIFACT_ROOT = (
    "b55211623be7f5697c5262ffd8361fc06c147bc9/batched_gemm-b3c1646-c111d7c/"
)
BASE_ARTIFACT_URL = (
    "https://edge.urm.nvidia.com/artifactory/"
    "sw-kernelinferencelibrary-public-generic-local/" + BASE_ARTIFACT_ROOT
)
BASE_MANIFEST_SHA256 = (
    "0af823880730c4f0b3832d2208fab035946694b83444410b9309db5613d60195"
)
SEALED_MANIFEST_SHA256 = (
    "504ca05b32d75242df92cd2beffb559837d994f63825903bfea4472751c80350"
)
CUDA_PTX_URL = (
    "https://edge.urm.nvidia.com/artifactory/"
    "sw-kernelinferencelibrary-public-generic-local/"
    "5db0eee675e8d8d81aa56853ae0c235f6bca7f55/cuda_ptx-5ee61af/cuda_ptx.h"
)
CUDA_PTX_SHA256 = "0003bb62b07a87881844f40cf19eb8f98a99cb2bfd5a782b0deb7d3a0750df6f"

ARTIFACT_RELATIVE_ROOT = Path("fireworks/situ_b552/batched_gemm-b5521162-64ed071e")
MODULE_BASENAME = "fused_moe_trtllm_sm100_situ_b552_64ed071e"
NATIVE_SOURCE_MANIFEST_SHA256 = (
    "2f8d9bdf2439dc04bc39c5a51c6a375398ee8e4e5d3c53e44e08ce7058c041ac"
)
NATIVE_MODULE_ABI_SHA256 = (
    "4136c08c87d437e4c94367088ebd2c4ea485190eae0bfd854a0142893a2ff4a2"
)
MODULE_NAME = f"{MODULE_BASENAME}_{NATIVE_MODULE_ABI_SHA256[:12]}"
NATIVE_FFI_ABI = {
    "logits_moe": "trtllm_fp4_block_scale_situ_logits_moe",
    "pre_routed_moe": "trtllm_fp4_block_scale_situ_routed_moe",
    "valid_configs": "trtllm_get_valid_situ_moe_configs",
    "manifest_digest": "trtllm_situ_b552_manifest_digest",
    "native_abi_digest": "trtllm_situ_b552_native_abi_digest",
}

# Host-layout probe compiled against CUDA 13's CUtensorMap declaration and the
# immutable b552 KernelParamsDecl.h.  These values are part of the cubin ABI,
# not implementation details: the runtime module must reject a mismatched
# generated-header bundle before launching a kernel.
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
    # MoE::ActivationType, passed across the exported runner ABI.
    "runner_enum_name": "Swiglu",
    "runner_enum_value": 3,
    # gemmGatedAct::ActType, embedded in the generated b552 metadata.
    "generated_enum_name": "SwiGlu",
    "generated_enum_value": 0,
    "private_semantics": "situ",
}
NATIVE_RUNNER_ABI = "b552_situ_precomputed_ids_v1"
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

_SOURCE_NAME = re.compile(
    r"^Bmm_MxE4m3_MxE2m1MxE4m3_.*_bA32_bB32_bC32_.*" r"_swiGlu_.*_sm100f\.(?:cu|h)$"
)
_TARGET_CUBIN = re.compile(
    r"^Bmm_MxE4m3_MxE2m1MxE4m3_.*_bA32_bB32_bC32_.*" r"_swiGlu_.*_sm100f\.cubin$"
)
_FC2_CUBIN = re.compile(r"^Bmm_Bfloat16_MxE2m1MxE4m3_Fp32_bA32_bB32_.*_sm100f\.cubin$")

# Generated by the b552 code generator.  The two indentation variants are
# handled by the leading capture.  Every source occurrence must match exactly;
# otherwise the build fails rather than silently shipping a SwiGLU tactic.
_SWIGLU_EPILOGUE = re.compile(
    r"(?P<i>^[ ]+)cutlass::Array<float, 2> fusedScaleArray\{"
    r"\(mGatedActAlpha\) \* \(float\{1\.442695\}\),\n"
    r"(?P=i)[ ]+\(mGatedActAlpha\) \* \(float\{1\.442695\}\)\};\n"
    r"(?P=i)cutlass::Array<float, 2> betaScaleGateArray\{mGatedActBeta, mGatedActBeta\};\n"
    r"(?P=i)cutlass::Array<float, 2> scaleGateArray\{float\{1\}, float\{1\}\};\n"
    r"(?P=i)cutlass::Array<float, 2> x0ScaleGateArray;\n"
    r"(?P=i)x0ScaleGateArray = trtllm::dev::ffma2\(x0Array, scaleGateArray, betaScaleGateArray\);\n"
    r"(?P=i)cutlass::Array<float, 2> x1ScaledArray;\n"
    r"(?P=i)x1ScaledArray = trtllm::dev::fmul2\(x1Array, fusedScaleArray\);\n"
    r"(?P=i)cutlass::Array<float, 2> actArray;\n"
    r"(?P=i)actArray = trtllm::dev::sigmoid2_base2\(x1ScaledArray\);\n"
    r"(?P=i)cutlass::Array<float, 2> swishArray;\n"
    r"(?P=i)swishArray = trtllm::dev::fmul2\(x1Array, actArray\);\n"
    r"(?P=i)cutlass::Array<float, 2> gatedActArray;\n"
    r"(?P=i)gatedActArray = trtllm::dev::fmul2\(x0ScaleGateArray, swishArray\);",
    re.MULTILINE,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validate_private_work_root(work_dir: Path) -> tuple[int, int]:
    info = work_dir.lstat()
    mode = stat.S_IMODE(info.st_mode)
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()
        or mode & 0o077
        or mode & 0o700 != 0o700
    ):
        raise RuntimeError(
            "Deterministic SiTU work root must be a private directory owned "
            f"by the current user: {work_dir} (uid={info.st_uid}, mode={mode:o})"
        )
    return info.st_dev, info.st_ino


def _clear_locked_work_root(work_dir: Path) -> None:
    for child in work_dir.iterdir():
        if child.name == _WORK_ROOT_LOCK:
            continue
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)
        else:
            raise RuntimeError(f"Unsupported entry in SiTU work root: {child}")


@contextmanager
def _locked_deterministic_work_root():
    """Own one stable NVCC source path without allowing concurrent deletion."""
    with suppress(FileExistsError):
        DETERMINISTIC_WORK_ROOT.mkdir(parents=True, mode=0o700)
    initial_identity = _validate_private_work_root(DETERMINISTIC_WORK_ROOT)

    lock_path = DETERMINISTIC_WORK_ROOT / _WORK_ROOT_LOCK
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    with os.fdopen(descriptor, "a+") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if _validate_private_work_root(DETERMINISTIC_WORK_ROOT) != initial_identity:
            raise RuntimeError(
                "Deterministic SiTU work root changed while acquiring its lock"
            )
        descriptor_info = os.fstat(lock_file.fileno())
        path_info = lock_path.lstat()
        if (
            not stat.S_ISREG(path_info.st_mode)
            or path_info.st_uid != os.geteuid()
            or stat.S_IMODE(path_info.st_mode) & 0o077
            or (path_info.st_dev, path_info.st_ino)
            != (descriptor_info.st_dev, descriptor_info.st_ino)
        ):
            raise RuntimeError(f"Unsafe deterministic SiTU build lock: {lock_path}")
        _clear_locked_work_root(DETERMINISTIC_WORK_ROOT)
        try:
            yield DETERMINISTIC_WORK_ROOT
        finally:
            _clear_locked_work_root(DETERMINISTIC_WORK_ROOT)


def _read_url(url: str) -> bytes:
    with urlopen(url) as response:  # noqa: S310 - immutable, hash-verified inputs
        return response.read()


def _write_verified(path: Path, data: bytes, expected_sha256: str) -> None:
    actual = _sha256(data)
    if actual != expected_sha256:
        raise RuntimeError(
            f"SHA-256 mismatch for {path.name}: expected {expected_sha256}, got {actual}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _normalize_artifact_file_modes(root: Path) -> None:
    """Make every packaged regular file readable and non-executable."""
    for path in root.rglob("*"):
        if path.is_file() and not path.is_symlink():
            path.chmod(0o644)


def _source_files(source_root: Path) -> list[Path]:
    source_dir = source_root / SOURCE_RELATIVE_DIR
    files = sorted(p for p in source_dir.iterdir() if _SOURCE_NAME.fullmatch(p.name))
    if len(files) != EXPECTED_SOURCE_FILES:
        raise RuntimeError(
            f"Expected {EXPECTED_SOURCE_FILES} b552 source files, found {len(files)}"
        )

    actual = _source_manifest_sha256(files, source_root)
    if actual != SOURCE_MANIFEST_SHA256:
        raise RuntimeError(
            "The b552 source snapshot is not the immutable PR #2917 input: "
            f"expected aggregate {SOURCE_MANIFEST_SHA256}, got {actual}"
        )
    return files


def _source_manifest_sha256(files: list[Path], source_root: Path) -> str:
    return _sha256(_source_manifest_bytes(files, source_root))


def _source_manifest_bytes(files: list[Path], source_root: Path) -> bytes:
    entries = []
    for path in files:
        relative = path.relative_to(source_root).as_posix()
        entries.append(f"{_sha256(path.read_bytes())}  {relative}\n")
    return "".join(entries).encode()


def _vendored_b552_headers(manifest: dict[str, str]) -> list[Path]:
    expected = sorted(name for name in manifest if name.startswith("include/"))
    actual = sorted(
        path.relative_to(VENDORED_B552_ROOT).as_posix()
        for path in VENDORED_B552_ROOT.rglob("*")
        if path.is_file()
    )
    if len(expected) != EXPECTED_B552_HEADERS or actual != expected:
        raise RuntimeError(
            "Vendored b552 generated-header set does not match the pinned "
            f"base artifact: expected {len(expected)}, found {len(actual)}"
        )

    entries = []
    result = []
    for relative in actual:
        path = VENDORED_B552_ROOT / relative
        digest = _sha256(path.read_bytes())
        if digest != manifest[relative]:
            raise RuntimeError(f"Vendored b552 generated header is corrupt: {relative}")
        entries.append(f"{digest}  {relative}\n")
        result.append(path)
    aggregate = _sha256("".join(entries).encode())
    if aggregate != B552_GENERATED_HEADERS_MANIFEST_SHA256:
        raise RuntimeError(
            "Vendored b552 generated-header manifest changed: "
            f"expected {B552_GENERATED_HEADERS_MANIFEST_SHA256}, got {aggregate}"
        )
    return result


def _require_clean_git_revision(root: Path, expected: str, label: str) -> None:
    try:
        actual = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
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
    if actual != expected:
        raise RuntimeError(
            f"{label} revision mismatch: expected {expected}, got {actual}"
        )
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


def _require_cuda_toolchain_identity(*tools: Path) -> None:
    for tool in tools:
        actual = _cuda_tool_identity(tool)
        if actual != CUDA_TOOLCHAIN_IDENTITY:
            raise RuntimeError(
                f"CUDA toolchain mismatch for {tool}: "
                f"expected {CUDA_TOOLCHAIN_IDENTITY}, got {actual}"
            )


def _situ_epilogue(match: re.Match[str]) -> str:
    i = match.group("i")
    lines = [
        "cutlass::Array<float, 2> log2eArray{float{1.442695}, float{1.442695}};",
        "cutlass::Array<float, 2> twoArray{float{2}, float{2}};",
        "cutlass::Array<float, 2> minusOneArray{float{-1}, float{-1}};",
        "cutlass::Array<float, 2> gateBetaArray{mGatedActAlpha, mGatedActAlpha};",
        "cutlass::Array<float, 2> gateSigmoidInput;",
        "gateSigmoidInput = trtllm::dev::fmul2(x1Array, log2eArray);",
        "cutlass::Array<float, 2> gateSigmoidArray;",
        "gateSigmoidArray = trtllm::dev::sigmoid2_base2(gateSigmoidInput);",
        "cutlass::Array<float, 2> gateTanhScaleArray{",
        "  (float{2} * float{1.442695}) / mGatedActAlpha,",
        "  (float{2} * float{1.442695}) / mGatedActAlpha};",
        "cutlass::Array<float, 2> gateTanhInput;",
        "gateTanhInput = trtllm::dev::fmul2(x1Array, gateTanhScaleArray);",
        "cutlass::Array<float, 2> gateTanhArray;",
        "gateTanhArray = trtllm::dev::sigmoid2_base2(gateTanhInput);",
        "gateTanhArray = trtllm::dev::ffma2(gateTanhArray, twoArray, minusOneArray);",
        "gateTanhArray = trtllm::dev::fmul2(gateTanhArray, gateBetaArray);",
        "cutlass::Array<float, 2> upArray;",
        "upArray = x0Array;",
        "if (mGatedActBeta > float{0}) {",
        "  cutlass::Array<float, 2> upTanhScaleArray{",
        "    (float{2} * float{1.442695}) / mGatedActBeta,",
        "    (float{2} * float{1.442695}) / mGatedActBeta};",
        "  cutlass::Array<float, 2> upTanhInput;",
        "  upTanhInput = trtllm::dev::fmul2(x0Array, upTanhScaleArray);",
        "  upArray = trtllm::dev::sigmoid2_base2(upTanhInput);",
        "  upArray = trtllm::dev::ffma2(upArray, twoArray, minusOneArray);",
        "  cutlass::Array<float, 2> linearBetaArray{mGatedActBeta, mGatedActBeta};",
        "  upArray = trtllm::dev::fmul2(upArray, linearBetaArray);",
        "}",
        "cutlass::Array<float, 2> gatedActArray;",
        "gatedActArray = trtllm::dev::fmul2(gateTanhArray, gateSigmoidArray);",
        "gatedActArray = trtllm::dev::fmul2(upArray, gatedActArray);",
    ]
    return ("\n" + i).join(lines)


def transform_sources(source_root: Path, output_dir: Path) -> list[Path]:
    files = _source_files(source_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    cuda_outputs: list[Path] = []
    total_replacements = 0

    for source in files:
        text = source.read_text()
        if source.suffix == ".cu":
            text, replacements = _SWIGLU_EPILOGUE.subn(_situ_epilogue, text)
            if replacements == 0:
                raise RuntimeError(f"No exact SwiGLU epilogue found in {source.name}")
            total_replacements += replacements
        text = text.replace("_swiGlu_", "_situ_")
        destination = output_dir / source.name.replace("_swiGlu_", "_situ_")
        destination.write_text(text)
        if destination.suffix == ".cu":
            if "swiGlu" in text or "swishArray" in text:
                raise RuntimeError(
                    f"Residual SwiGLU implementation in {destination.name}"
                )
            cuda_outputs.append(destination)

    if len(cuda_outputs) != EXPECTED_CUDA_SOURCES:
        raise RuntimeError(
            f"Expected {EXPECTED_CUDA_SOURCES} CUDA sources, got {len(cuda_outputs)}"
        )
    if total_replacements != EXPECTED_EPILOGUES:
        raise RuntimeError(
            f"Expected {EXPECTED_EPILOGUES} epilogues, replaced {total_replacements}"
        )
    return cuda_outputs


def _parse_manifest(data: bytes) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in data.decode().splitlines():
        digest, name = line.split(maxsplit=1)
        result[name] = digest
    return result


def _download_base_inputs(work_dir: Path) -> tuple[dict[str, str], Path, list[Path]]:
    manifest_data = _read_url(BASE_ARTIFACT_URL + "checksums.txt")
    if _sha256(manifest_data) != BASE_MANIFEST_SHA256:
        raise RuntimeError("Public b552 checksums.txt does not match the pinned digest")
    manifest = _parse_manifest(manifest_data)

    selected = [name for name in manifest if _FC2_CUBIN.fullmatch(name)]
    if len(selected) != EXPECTED_FC2_CUBINS:
        raise RuntimeError(
            f"Expected {EXPECTED_FC2_CUBINS} runner-selected b552 FC2 cubins, "
            f"found {len(selected)}"
        )

    downloaded: list[Path] = []
    for name in selected:
        destination = work_dir / "base" / name
        _write_verified(
            destination, _read_url(BASE_ARTIFACT_URL + name), manifest[name]
        )
        downloaded.append(destination)

    for source in _vendored_b552_headers(manifest):
        relative = source.relative_to(VENDORED_B552_ROOT)
        destination = work_dir / "base" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        downloaded.append(destination)

    cuda_ptx = work_dir / "cuda_ptx" / "cuda_ptx" / "cuda_ptx.h"
    _write_verified(cuda_ptx, _read_url(CUDA_PTX_URL), CUDA_PTX_SHA256)
    return manifest, cuda_ptx.parent.parent, downloaded


def compile_cubins(
    sources: list[Path],
    output_dir: Path,
    source_root: Path,
    cuda_ptx_include: Path,
    b552_generated_include: Path,
    nvcc: Path,
    cutlass_root: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    export_root = source_root / SOURCE_RELATIVE_DIR.parent
    include_paths = [
        sources[0].parent,
        b552_generated_include,
        source_root / "csrc/trtllm_kernels",
        export_root,
        cuda_ptx_include,
        source_root / "csrc/nv_internal",
        source_root / "csrc/nv_internal/include",
        cutlass_root / "include",
        cutlass_root / "tools/util/include",
    ]
    flags = [
        "--expt-relaxed-constexpr",
        "--use_fast_math",
        "-std=c++17",
        "-arch=sm_100f",
        "-DTLLM_ENABLE_CUDA",
        "-DNDEBUG=1",
        "-DCUTLASS_ARCH_MMA_SM100A_ENABLED",
        "-DTLLM_PUBLIC_RELEASE=1",
        "-diag-suppress=177",
        "-diag-suppress=2361",
        "-diag-suppress=550",
        "-O3",
        "-cubin",
    ]
    outputs: list[Path] = []
    for source in sources:
        destination = output_dir / (source.stem + ".cubin")
        random_seed = hashlib.sha256(source.name.encode()).hexdigest()[:16]
        command = [
            str(nvcc),
            str(source),
            "-o",
            str(destination),
            *flags,
            f"--frandom-seed=0x{random_seed}",
        ]
        command.extend(f"-I{path}" for path in include_paths)
        subprocess.run(command, check=True)
        outputs.append(destination)
    return outputs


def _rewrite_metainfo(metainfo: str, cubins: list[Path]) -> str:
    for cubin in cubins:
        new_name = cubin.name.removesuffix(".cubin")
        old_name = new_name.replace("_situ_", "_swiGlu_")
        old_symbol = "bmm_" + old_name.removeprefix("Bmm_")
        new_symbol = old_symbol.replace("_swiGlu_", "_situ_")
        digest = _sha256(cubin.read_bytes())
        pattern = re.compile(
            r'(\{nullptr, 0, \d+, "'
            + re.escape(old_symbol)
            + r'", \d+, ")[0-9a-f]{64}("[,}])'
        )
        metainfo, replacements = pattern.subn(rf"\g<1>{digest}\g<2>", metainfo)
        if replacements != 1:
            raise RuntimeError(
                f"Expected one metadata entry for {old_symbol}, got {replacements}"
            )
        metainfo = metainfo.replace(old_symbol, new_symbol, 1)
    if "bmm_MxE4m3_MxE2m1MxE4m3_" in metainfo:
        residual = re.findall(r'"bmm_MxE4m3_MxE2m1MxE4m3_[^"]*_swiGlu_[^"]*"', metainfo)
        if residual:
            raise RuntimeError(
                f"Residual target SwiGLU metadata entries: {len(residual)}"
            )
    return metainfo


def _cubin_kernel_symbol(cubin: Path) -> str:
    if cubin.suffix != ".cubin" or not cubin.name.startswith("Bmm_"):
        raise RuntimeError(f"Unexpected SiTU b552 cubin name: {cubin.name}")
    return "bmm_" + cubin.stem.removeprefix("Bmm_")


def _run_cuobjdump(cuobjdump: Path, cubin: Path) -> str:
    try:
        return subprocess.run(
            [str(cuobjdump), "--dump-elf", str(cubin)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        stderr = getattr(error, "stderr", "")
        raise RuntimeError(
            f"cuobjdump failed while validating SiTU b552 cubins: {stderr}"
        ) from error


def _validate_cubin_binaries(
    cubins: list[Path], metainfo: str, cuobjdump: Path
) -> None:
    """Bind every selected metadata entry to one SM100 ELF kernel binary."""
    if not cubins:
        raise RuntimeError("No SiTU b552 cubins were provided for validation")
    if len(set(cubins)) != len(cubins):
        raise RuntimeError("Duplicate SiTU b552 cubin paths were provided")

    expected_symbols = {_cubin_kernel_symbol(cubin) for cubin in cubins}
    if len(expected_symbols) != len(cubins):
        raise RuntimeError("Duplicate SiTU b552 cubin kernel symbols were provided")

    for cubin in cubins:
        elf_dump = _run_cuobjdump(cuobjdump, cubin)
        architectures = re.findall(
            r"^64-bit ELF:.*\bsm=(\d+)\b", elf_dump, re.MULTILINE
        )
        if architectures != ["100"]:
            raise RuntimeError(
                f"SiTU b552 cubin must contain exactly one SM100 ELF image: {cubin}"
            )

        symbol = _cubin_kernel_symbol(cubin)
        expected_entries = (symbol, f"{symbol}GetSmemSize")
        missing_entries = []
        for entry in expected_entries:
            # ELF info=0x12 is STB_GLOBAL|STT_FUNC and other=0x10 is the CUDA
            # STO_ENTRY visibility used by cuModuleGetFunction.
            pattern = re.compile(
                r"^\s*\S+\s+\S+\s+\S+\s+0x12\s+0x10\s+\S+\s+"
                + re.escape(entry)
                + r"\s*$",
                re.MULTILINE,
            )
            if len(pattern.findall(elf_dump)) != 1:
                missing_entries.append(entry)
        if missing_entries:
            raise RuntimeError(
                f"SiTU b552 cubin ELF entry mismatch for {cubin}: {missing_entries}"
            )

    missing_metadata = sorted(
        symbol for symbol in expected_symbols if metainfo.count(f'"{symbol}"') != 1
    )
    if missing_metadata:
        raise RuntimeError(
            "SiTU b552 metadata must resolve every cubin symbol exactly once: "
            f"{missing_metadata}"
        )


def assemble_artifact(
    output_root: Path,
    base_files: list[Path],
    compiled_cubins: list[Path],
    work_dir: Path,
    source_root: Path,
    source_files: list[Path],
    cuobjdump: Path,
) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    for source in base_files:
        relative = source.relative_to(work_dir / "base")
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for source in compiled_cubins:
        shutil.copy2(source, output_root / source.name)

    metainfo_path = output_root / "include/flashinferMetaInfo.h"
    metainfo_path.write_text(
        _rewrite_metainfo(metainfo_path.read_text(), compiled_cubins)
    )
    all_cubins = sorted(output_root.glob("*.cubin"))
    if len(all_cubins) != EXPECTED_CUDA_SOURCES + EXPECTED_FC2_CUBINS:
        raise RuntimeError(
            f"Expected {EXPECTED_CUDA_SOURCES} SiTU FC1 plus "
            f"{EXPECTED_FC2_CUBINS} runner-selected FC2 cubins, "
            f"found {len(all_cubins)}"
        )
    _validate_cubin_binaries(all_cubins, metainfo_path.read_text(), cuobjdump)

    header_sha256 = dict(B552_HEADER_SHA256)
    for relative, expected in header_sha256.items():
        actual = _sha256((output_root / relative).read_bytes())
        if actual != expected:
            raise RuntimeError(
                f"b552 generated header mismatch for {relative}: "
                f"expected {expected}, got {actual}"
            )
    header_sha256["include/flashinferMetaInfo.h"] = _sha256(metainfo_path.read_bytes())

    # Preserve a signed, reviewable record of every mechanically transformed
    # source plus the immutable snapshot's license notices.  Shared headers are
    # additionally bound by the clean Git revision check in main().
    provenance = output_root / "provenance"
    provenance.mkdir(parents=True)
    source_manifest = _source_manifest_bytes(source_files, source_root)
    if _sha256(source_manifest) != SOURCE_MANIFEST_SHA256:
        raise RuntimeError("Source provenance manifest changed during the build")
    (provenance / "source-files.sha256").write_bytes(source_manifest)
    for source_name, destination_name, expected in (
        ("LICENSE", "flashinfer-LICENSE", SOURCE_LICENSE_SHA256),
        ("NOTICE", "flashinfer-NOTICE", SOURCE_NOTICE_SHA256),
    ):
        data = (source_root / source_name).read_bytes()
        _write_verified(provenance / destination_name, data, expected)

    # Write the ABI contract before checksums.txt so the manifest seal covers
    # it.  The manifest digest is deliberately not embedded here because that
    # would create a self-referential digest.
    (output_root / "contract.json").write_text(
        json.dumps(
            {
                "activation_abi": ACTIVATION_ABI,
                "artifact_root": ARTIFACT_RELATIVE_ROOT.as_posix(),
                "base_artifact_root": BASE_ARTIFACT_ROOT,
                "base_manifest_sha256": BASE_MANIFEST_SHA256,
                "cubin_compile_contract": CUBIN_COMPILE_CONTRACT,
                "cuda_arch": "sm_100f",
                "cuda_toolchain": CUDA_TOOLCHAIN_IDENTITY,
                "cutlass_commit": CUTLASS_COMMIT,
                "fc1_cubins": EXPECTED_CUDA_SOURCES,
                "fc2_cubins": EXPECTED_FC2_CUBINS,
                "header_sha256": header_sha256,
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
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    # copy2 deliberately preserves the immutable input timestamps, but its mode
    # preservation can make headers unreadable when the build inputs were
    # created by root with a restrictive umask.  Wheel contents are data, not
    # executables, so normalize every regular artifact before sealing it.
    _normalize_artifact_file_modes(output_root)

    entries: list[tuple[str, str]] = []
    for path in sorted(p for p in output_root.rglob("*") if p.is_file()):
        relative = path.relative_to(output_root).as_posix()
        entries.append((_sha256(path.read_bytes()), relative))
    manifest = "".join(f"{digest}  {name}\n" for digest, name in entries).encode()
    checksums_path = output_root / "checksums.txt"
    checksums_path.write_bytes(manifest)
    checksums_path.chmod(0o644)
    manifest_digest = _sha256(manifest)
    if manifest_digest != SEALED_MANIFEST_SHA256:
        raise RuntimeError(
            "Generated SiTU b552 bundle does not match the signed-off manifest: "
            f"expected {SEALED_MANIFEST_SHA256}, got {manifest_digest}"
        )
    manifest_path = output_root / "manifest.sha256"
    manifest_path.write_text(f"{manifest_digest}  checksums.txt\n")
    manifest_path.chmod(0o644)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--cutlass-root", type=Path, required=True)
    args = parser.parse_args()

    if not args.nvcc.is_file():
        raise RuntimeError(f"NVCC not found: {args.nvcc}")
    cuobjdump = args.nvcc.parent / "cuobjdump"
    if not cuobjdump.is_file():
        raise RuntimeError(f"cuobjdump not found next to NVCC: {cuobjdump}")
    ptxas = args.nvcc.parent / "ptxas"
    if not ptxas.is_file():
        raise RuntimeError(f"ptxas not found next to NVCC: {ptxas}")
    _require_cuda_toolchain_identity(args.nvcc, cuobjdump, ptxas)
    _require_clean_git_revision(
        args.source_root, SOURCE_COMMIT, "FlashInfer PR #2917 source"
    )
    _require_clean_git_revision(args.cutlass_root, CUTLASS_COMMIT, "CUTLASS")
    for relative in ("include", "tools/util/include"):
        if not (args.cutlass_root / relative).is_dir():
            raise RuntimeError(f"CUTLASS checkout is missing {relative}")
    # NVCC 13.2 derives CUDA-internal ELF symbol suffixes from the canonical
    # translation-unit path even when --frandom-seed is provided.  Keep that
    # path fixed and hold an owner-only process lock while its contents exist.
    # This makes generated paths identical across x86/ARM and clean rebuilds
    # without allowing concurrent builders to delete each other's inputs.
    with _locked_deterministic_work_root() as work_dir:
        source_files = _source_files(VENDORED_SOURCE_ROOT)
        generated = transform_sources(VENDORED_SOURCE_ROOT, work_dir / "generated/src")
        _, cuda_ptx_include, base_files = _download_base_inputs(work_dir)
        cubins = compile_cubins(
            generated,
            work_dir / "compiled",
            args.source_root,
            cuda_ptx_include,
            work_dir / "base/include/trtllmGen_bmm_export",
            args.nvcc,
            args.cutlass_root,
        )
        assemble_artifact(
            args.output_root,
            base_files,
            cubins,
            work_dir,
            VENDORED_SOURCE_ROOT,
            source_files,
            cuobjdump,
        )


if __name__ == "__main__":
    main()
