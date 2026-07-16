import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import types
import uuid
from enum import IntEnum
from typing import Optional

import pytest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fc1_name(index: int) -> str:
    return (
        "Bmm_MxE4m3_MxE2m1MxE4m3_Fp32_bA32_bB32_bC32_"
        f"case{index:03d}_situ_dynB_sm100f.cubin"
    )


def _fc2_name(index: int) -> str:
    return f"Bmm_Bfloat16_MxE2m1MxE4m3_Fp32_bA32_bB32_case{index:03d}_dynB_sm100f.cubin"


def _load_runtime_module(tmp_path: Path):
    project_root = Path(__file__).parents[1]
    package_name = f"_flashinfer_situ_test_{uuid.uuid4().hex}"
    package = types.ModuleType(package_name)
    package.__path__ = []
    sys.modules[package_name] = package

    jit_package = types.ModuleType(f"{package_name}.jit")
    jit_package.__path__ = []
    sys.modules[jit_package.__name__] = jit_package

    env = types.ModuleType(f"{package_name}.jit.env")
    env.FLASHINFER_CUBIN_DIR = tmp_path / "cubins"
    env.FLASHINFER_GEN_SRC_DIR = tmp_path / "generated"
    env.FLASHINFER_CSRC_DIR = project_root / "csrc"
    env.FLASHINFER_INCLUDE_DIR = project_root / "include"
    sys.modules[env.__name__] = env

    core = types.ModuleType(f"{package_name}.jit.core")
    core.JitSpec = object
    core.current_compilation_context = types.SimpleNamespace(
        get_nvcc_flags_list=lambda **_kwargs: []
    )
    core.gen_jit_spec = lambda *_args, **_kwargs: None
    sys.modules[core.__name__] = core

    cpp_ext = types.ModuleType(f"{package_name}.jit.cpp_ext")
    cpp_ext.get_cuda_path = lambda: "/usr/local/cuda"
    sys.modules[cpp_ext.__name__] = cpp_ext

    path = Path(__file__).parents[1] / "flashinfer/jit/situ_b552.py"
    name = f"{package_name}.jit.situ_b552"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, env


def _seal_bundle(root: Path) -> str:
    entries = []
    for path in sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"checksums.txt", "manifest.sha256"}
    ):
        entries.append(f"{_sha256(path)}  {path.relative_to(root).as_posix()}\n")
    manifest = "".join(entries)
    (root / "checksums.txt").write_text(manifest)
    digest = hashlib.sha256(manifest.encode()).hexdigest()
    (root / "manifest.sha256").write_text(f"{digest}  checksums.txt\n")
    return digest


def _make_valid_bundle(module, env) -> Path:
    root = env.FLASHINFER_CUBIN_DIR / module.ARTIFACT_RELATIVE_ROOT
    root.mkdir(parents=True)

    static_header_sha256 = {}
    for index, relative in enumerate(module.B552_HEADER_SHA256):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"header-{index}\n".encode())
        static_header_sha256[relative] = _sha256(path)
    # Tests use tiny deterministic fixtures; production values remain pinned in
    # both the generator and runtime modules.
    module.B552_HEADER_SHA256 = static_header_sha256

    provenance_sha256 = {}
    for index, relative in enumerate(module.PROVENANCE_SHA256):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"provenance-{index}\n".encode())
        provenance_sha256[relative] = _sha256(path)
    module.PROVENANCE_SHA256 = provenance_sha256

    metainfo = root / "include/flashinferMetaInfo.h"
    metainfo.parent.mkdir(parents=True, exist_ok=True)
    metainfo.write_text("// rewritten private metadata\n")
    header_sha256 = dict(static_header_sha256)
    header_sha256["include/flashinferMetaInfo.h"] = _sha256(metainfo)

    for index in range(102):
        (root / _fc1_name(index)).write_bytes(f"situ-{index}".encode())
    for index in range(101):
        (root / _fc2_name(index)).write_bytes(f"fc2-{index}".encode())

    contract = {
        "activation_abi": module.ACTIVATION_ABI,
        "artifact_root": module.ARTIFACT_RELATIVE_ROOT.as_posix(),
        "base_artifact_root": module.BASE_ARTIFACT_ROOT,
        "base_manifest_sha256": module.BASE_MANIFEST_SHA256,
        "cubin_compile_contract": module.CUBIN_COMPILE_CONTRACT,
        "cuda_arch": "sm_100f",
        "cuda_toolchain": module.CUDA_TOOLCHAIN_IDENTITY,
        "cutlass_commit": module.CUTLASS_COMMIT,
        "fc1_cubins": 102,
        "fc2_cubins": 101,
        "header_sha256": header_sha256,
        "kernel_params_abi": module.KERNEL_PARAMS_ABI,
        "module_name": module.MODULE_NAME,
        "native_ffi_abi": module.NATIVE_FFI_ABI,
        "native_module_abi_sha256": module.NATIVE_MODULE_ABI_SHA256,
        "native_runner_abi": module.NATIVE_RUNNER_ABI,
        "native_source_manifest_sha256": module.NATIVE_SOURCE_MANIFEST_SHA256,
        "routing_source_commit": module.ROUTING_SOURCE_COMMIT,
        "routing_source_manifest_sha256": module.ROUTING_SOURCE_MANIFEST_SHA256,
        "source_commit": module.SOURCE_COMMIT,
        "source_manifest_sha256": module.SOURCE_MANIFEST_SHA256,
    }
    (root / "contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n"
    )
    module.SEALED_MANIFEST_SHA256 = _seal_bundle(root)
    return root


def test_manifest_verification_and_artifact_corruption(tmp_path):
    module, env = _load_runtime_module(tmp_path)
    root = _make_valid_bundle(module, env)

    assert module.verify_situ_b552_artifacts() == _sha256(root / "checksums.txt")

    (root / _fc1_name(0)).write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="Missing or corrupt"):
        module.verify_situ_b552_artifacts()


def test_runtime_manifest_digest_is_pinned(tmp_path):
    module, _env = _load_runtime_module(tmp_path)
    assert module.SEALED_MANIFEST_SHA256 == (
        "504ca05b32d75242df92cd2beffb559837d994f63825903bfea4472751c80350"
    )


def test_native_module_abi_identity_is_canonical_and_namespaced(tmp_path):
    module, _env = _load_runtime_module(tmp_path)
    assert module._canonical_json_sha256(module._native_module_abi_contract()) == (
        module.NATIVE_MODULE_ABI_SHA256
    )
    assert (
        f"{module.MODULE_BASENAME}_{module.NATIVE_MODULE_ABI_SHA256[:12]}"
    ) == module.MODULE_NAME
    assert {
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
        "cubin_artifact_root": module.ARTIFACT_RELATIVE_ROOT.as_posix(),
        "cxx_standard": "c++17",
        "cuda_targets": ["sm_100a", "sm_103a"],
        "cuda_toolchain": {
            "build": "cuda_13.2.r13.2/compiler.37668154_0",
            "release": "13.2",
            "version": "13.2.78",
        },
        "debug": False,
        "fatbin_compression": "all",
        "host_visibility": "hidden",
        "lineinfo": False,
        "ndebug": True,
        "optimization": "O3",
        "runtime_mode": "aot_only",
        "use_fast_math": True,
    } == module.NATIVE_COMPILE_CONTRACT


def test_source_build_requires_exact_cuda_toolchain(tmp_path, monkeypatch):
    module, _env = _load_runtime_module(tmp_path)
    cuda_home = tmp_path / "cuda"
    nvcc = cuda_home / "bin/nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text("synthetic nvcc")
    cxx = tmp_path / "c++"
    cxx.write_text("synthetic cxx")
    monkeypatch.delenv("FLASHINFER_NVCC", raising=False)
    monkeypatch.setattr(module, "get_cuda_path", lambda: str(cuda_home))

    def fake_which(command):
        assert command == "c++"
        return str(cxx)

    monkeypatch.setattr(module.shutil, "which", fake_which)
    state = {"version": "13.2.78"}

    def fake_version(command, **kwargs):
        assert command == [str(nvcc.resolve()), "--version"]
        assert kwargs == {"check": True, "capture_output": True, "text": True}
        return types.SimpleNamespace(
            stdout=(
                "Cuda compilation tools, release 13.2, "
                f"V{state['version']}\n"
                "Build cuda_13.2.r13.2/compiler.37668154_0\n"
            )
        )

    monkeypatch.setattr(module.subprocess, "run", fake_version)
    assert module._require_source_build_cuda_toolchain() == {
        "cxx": str(cxx.resolve()),
        "linker": str(cxx.resolve()),
        "nvcc": str(nvcc.resolve()),
    }

    other_nvcc = tmp_path / "other-nvcc"
    other_nvcc.write_text("different synthetic nvcc")
    monkeypatch.setenv("FLASHINFER_NVCC", str(other_nvcc))
    with pytest.raises(
        RuntimeError, match="must resolve to the compiler under CUDA_HOME"
    ):
        module._require_source_build_cuda_toolchain()
    monkeypatch.delenv("FLASHINFER_NVCC")

    state["version"] = "13.2.79"
    with pytest.raises(RuntimeError, match="CUDA toolchain mismatch"):
        module._require_source_build_cuda_toolchain()


def test_source_build_requires_exact_release_targets_and_flags(tmp_path, monkeypatch):
    module, _env = _load_runtime_module(tmp_path)
    for name in (
        *module._SOURCE_BUILD_ENV_GUARDS,
        *module._SOURCE_BUILD_EXTRA_FLAG_ENV,
    ):
        monkeypatch.delenv(name, raising=False)

    class BuildContext:
        TARGET_CUDA_ARCHS = {(10, "0a"), (10, "3a"), (9, "0a")}

        def get_nvcc_flags_list(self, supported_major_versions=None):
            assert supported_major_versions == [10]
            return sorted(module._SOURCE_BUILD_GENCODE_FLAGS) + [
                "-DFLASHINFER_ENABLE_FP8_E8M0",
                "-DFLASHINFER_ENABLE_FP4_E2M1",
            ]

    module.current_compilation_context = BuildContext()
    flags = module._source_build_nvcc_flags()
    assert (
        frozenset(flag for flag in flags if flag.startswith("-gencode="))
        == module._SOURCE_BUILD_GENCODE_FLAGS
    )

    toolchain = {
        "cxx": "/toolchain/bin/c++",
        "linker": "/toolchain/bin/c++",
        "nvcc": "/cuda/bin/nvcc",
    }
    valid_spec = types.SimpleNamespace(
        extra_cflags=["-std=c++17", "-DNDEBUG", "-O3", "-fvisibility=hidden"],
        extra_cuda_cflags=[
            "-std=c++17",
            "-DNDEBUG",
            "-O3",
            "-use_fast_math",
            "-Xfatbin=-compress-all",
            "-Xcompiler=-fvisibility=hidden",
            *sorted(module._SOURCE_BUILD_GENCODE_FLAGS),
        ],
        extra_ldflags=None,
        needs_device_linking=False,
        cxx=toolchain["cxx"],
        nvcc=toolchain["nvcc"],
        cxx_launcher="",
        nvcc_launcher="",
        use_environment_flags=False,
    )
    module._verify_source_build_spec(valid_spec, toolchain)

    valid_spec.nvcc_launcher = "sccache"
    with pytest.raises(RuntimeError, match="compiler/linker command mismatch"):
        module._verify_source_build_spec(valid_spec, toolchain)
    valid_spec.nvcc_launcher = ""

    module.current_compilation_context.TARGET_CUDA_ARCHS = {(10, "0a")}
    with pytest.raises(RuntimeError, match="require exactly the filtered CUDA targets"):
        module._source_build_nvcc_flags()

    valid_spec.extra_cuda_cflags.append("-lineinfo")
    with pytest.raises(RuntimeError, match="forbid debug, verbose, and lineinfo"):
        module._verify_source_build_spec(valid_spec, toolchain)


@pytest.mark.parametrize(
    "environment_name",
    [
        "FLASHINFER_JIT_DEBUG",
        "FLASHINFER_JIT_VERBOSE",
        "FLASHINFER_JIT_LINEINFO",
    ],
)
def test_source_build_rejects_enabled_debug_environment(
    tmp_path, monkeypatch, environment_name
):
    module, _env = _load_runtime_module(tmp_path)
    module.current_compilation_context = types.SimpleNamespace(
        TARGET_CUDA_ARCHS={(10, "0a"), (10, "3a")},
        get_nvcc_flags_list=lambda **_kwargs: sorted(
            module._SOURCE_BUILD_GENCODE_FLAGS
        ),
    )
    monkeypatch.setenv(environment_name, "1")
    with pytest.raises(RuntimeError, match="release-only JIT settings"):
        module._source_build_nvcc_flags()


@pytest.mark.parametrize("minor", ["0a", "3a"])
def test_aot_only_runtime_accepts_one_local_blackwell_target(
    tmp_path, monkeypatch, minor
):
    module, env = _load_runtime_module(tmp_path)
    _make_valid_bundle(module, env)
    monkeypatch.delenv("FLASHINFER_SITU_B552_SOURCE_ROOT", raising=False)
    gencode = f"-gencode=arch=compute_10{minor},code=sm_10{minor}"
    module.current_compilation_context = types.SimpleNamespace(
        TARGET_CUDA_ARCHS={(10, minor)},
        get_nvcc_flags_list=lambda **_kwargs: [gencode],
    )
    captured = {}

    def capture_spec(*args, **kwargs):
        captured["kwargs"] = kwargs
        return object()

    module.gen_jit_spec = capture_spec
    module.gen_trtllm_gen_fused_moe_situ_b552_module()
    assert gencode in captured["kwargs"]["extra_cuda_cflags"]


def test_immutable_source_revision_rejects_untracked_files(tmp_path):
    module, _env = _load_runtime_module(tmp_path)
    root = tmp_path / "source"
    root.mkdir()
    subprocess.run(["git", "init", "-q", root], check=True)
    subprocess.run(
        ["git", "-C", root, "config", "user.email", "situ-test@example.invalid"],
        check=True,
    )
    subprocess.run(["git", "-C", root, "config", "user.name", "SiTU Test"], check=True)
    tracked = root / "tracked.txt"
    tracked.write_text("tracked\n")
    subprocess.run(["git", "-C", root, "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", root, "commit", "-qm", "fixture"], check=True)
    revision = subprocess.run(
        ["git", "-C", root, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    module._require_clean_git_revision(root, revision, "fixture")
    (root / "untracked.txt").write_text("untracked\n")
    with pytest.raises(RuntimeError, match="not clean"):
        module._require_clean_git_revision(root, revision, "fixture")


def test_manifest_rejects_internally_consistent_but_unpinned_bundle(tmp_path):
    module, env = _load_runtime_module(tmp_path)
    root = _make_valid_bundle(module, env)
    pinned_digest = module.SEALED_MANIFEST_SHA256

    (root / _fc1_name(0)).write_bytes(b"replacement")
    replacement_digest = _seal_bundle(root)
    assert replacement_digest != pinned_digest

    with pytest.raises(RuntimeError, match="sealed manifest digest mismatch"):
        module.verify_situ_b552_artifacts()


def test_manifest_requires_exact_fc1_and_fc2_counts(tmp_path):
    module, env = _load_runtime_module(tmp_path)
    root = _make_valid_bundle(module, env)
    (root / _fc1_name(101)).unlink()
    module.SEALED_MANIFEST_SHA256 = _seal_bundle(root)

    with pytest.raises(RuntimeError, match="102 SiTU FC1 and 101"):
        module.verify_situ_b552_artifacts()


def test_manifest_rejects_contract_abi_corruption(tmp_path):
    module, env = _load_runtime_module(tmp_path)
    root = _make_valid_bundle(module, env)
    contract_path = root / "contract.json"
    contract = json.loads(contract_path.read_text())
    contract["kernel_params_abi"]["sizeof"] += 128
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    module.SEALED_MANIFEST_SHA256 = _seal_bundle(root)

    with pytest.raises(RuntimeError, match="kernel_params_abi"):
        module.verify_situ_b552_artifacts()


def test_current_routing_closure_is_hash_pinned_for_k3_and_precomputed_ids(
    tmp_path,
):
    module, _env = _load_runtime_module(tmp_path)
    resolved = module._verify_current_routing_sources()
    assert tuple(resolved) == tuple(module.ROUTING_SOURCE_SHA256)
    assert module._ROUTING_RUNTIME_SOURCES == (
        "csrc/fused_moe/trtllm_backend/trtllm_fused_moe_routing_custom.cu",
        "csrc/fused_moe/trtllm_backend/trtllm_fused_moe_routing_common.cu",
        "csrc/fused_moe/trtllm_backend/trtllm_fused_moe_routing_deepseek.cu",
        "csrc/fused_moe/trtllm_backend/trtllm_fused_moe_routing_llama4.cu",
    )
    assert not any(
        "routingDeepSeek/" in source
        or "routingRenormalize/" in source
        or "routing_renormalize.cu" in source
        for source in module._B552_RUNTIME_SOURCES
    )

    manifest = "".join(
        f"{_sha256(resolved[relative])}  {relative}\n"
        for relative in module.ROUTING_SOURCE_SHA256
    )
    assert hashlib.sha256(manifest.encode()).hexdigest() == (
        module.ROUTING_SOURCE_MANIFEST_SHA256
    )

    runner = resolved["csrc/trtllm_fused_moe_runner.cu"].read_text()
    assert "routingMethodType == RoutingMethodType::DeepSeekV3 && nGroup <= 1" in runner
    assert "routingCustom::Data routingData;" in runner
    assert "RoutingPreprocessType::SigmoidBias" in runner
    assert "RoutingPostprocessType::ScaledSumNormalize" in runner
    assert "routingData.mPtrTopKIds = expertIds;" in runner

    policy = resolved[
        "include/flashinfer/trtllm/fused_moe/RoutingCustomPolicy.cuh"
    ].read_text()
    assert "Tier<1024, 32>" in policy

    custom = resolved[
        "csrc/fused_moe/trtllm_backend/trtllm_fused_moe_routing_custom.cu"
    ].read_text()
    assert "if (data.mPtrTopKIds != nullptr ||" in custom
    assert "runPostTopKPipeline(data, stream);" in custom

    current_runner = resolved["csrc/trtllm_fused_moe_runner.cu"]
    corrupted_runner = tmp_path / "corrupted-current-runner.cu"
    corrupted_runner.write_bytes(current_runner.read_bytes() + b"// modified\n")
    original_resolver = module._routing_source_path
    module._routing_source_path = lambda relative: (
        corrupted_runner
        if relative == "csrc/trtllm_fused_moe_runner.cu"
        else original_resolver(relative)
    )
    try:
        with pytest.raises(RuntimeError, match="routing source hash mismatch"):
            module._verify_current_routing_sources()
    finally:
        module._routing_source_path = original_resolver


def test_missing_aot_module_fails_without_jit_or_network(tmp_path, monkeypatch):
    module, env = _load_runtime_module(tmp_path)
    _make_valid_bundle(module, env)

    class MissingAotSpec:
        is_aot = False

        def build_and_load(self):
            raise AssertionError("AOT-only SiTU must not attempt a runtime build")

    module.gen_jit_spec = lambda *_args, **_kwargs: MissingAotSpec()
    monkeypatch.delenv("FLASHINFER_SITU_B552_SOURCE_ROOT", raising=False)
    monkeypatch.setenv("FLASHINFER_NO_DOWNLOAD", "1")

    with pytest.raises(RuntimeError, match="runtime JIT fallback is disabled"):
        module.load_trtllm_gen_fused_moe_situ_b552_module()


def test_full_manifest_digest_is_compiled_and_compared(tmp_path, monkeypatch):
    module, env = _load_runtime_module(tmp_path)
    root = _make_valid_bundle(module, env)
    digest = _sha256(root / "checksums.txt")
    chunks = module._manifest_digest_chunks(digest)
    assert len(chunks) == 5
    assert all(chunk < (1 << 63) for chunk in chunks)
    offset = 0
    reconstructed = []
    for width, chunk in zip(module._MANIFEST_DIGEST_CHUNK_WIDTHS, chunks, strict=False):
        reconstructed.append(f"{chunk:0{width}x}")
        offset += width
    assert offset == 64
    assert "".join(reconstructed) == digest
    native_chunks = module._manifest_digest_chunks(module.NATIVE_MODULE_ABI_SHA256)

    captured = {}

    def capture_spec(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return object()

    module.gen_jit_spec = capture_spec
    module.gen_trtllm_gen_fused_moe_situ_b552_module()
    flags = captured["kwargs"]["extra_cuda_cflags"]
    assert "-Xcompiler=-fvisibility=hidden" in flags
    assert captured["kwargs"]["extra_cflags"] == ["-fvisibility=hidden"]
    defines = [
        flag
        for flag in flags
        if flag.startswith("-DFLASHINFER_SITU_B552_MANIFEST_CHUNK_")
    ]
    assert len(defines) == 5
    native_defines = [
        flag
        for flag in flags
        if flag.startswith("-DFLASHINFER_SITU_B552_NATIVE_ABI_CHUNK_")
    ]
    assert len(native_defines) == 5
    assert not any("MANIFEST_TAG" in flag for flag in flags)

    bad_chunks = list(chunks)
    bad_chunks[-1] ^= 1

    class MismatchedAotSpec:
        is_aot = True

        def build_and_load(self):
            return types.SimpleNamespace(
                trtllm_situ_b552_native_abi_digest=lambda: native_chunks,
                trtllm_situ_b552_manifest_digest=lambda: bad_chunks,
            )

    module.gen_jit_spec = lambda *_args, **_kwargs: MismatchedAotSpec()
    monkeypatch.delenv("FLASHINFER_SITU_B552_SOURCE_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="JIT-cache/cubin manifest mismatch"):
        module.load_trtllm_gen_fused_moe_situ_b552_module()

    bad_native_chunks = list(native_chunks)
    bad_native_chunks[-1] ^= 1

    class MismatchedNativeAotSpec:
        is_aot = True

        def build_and_load(self):
            return types.SimpleNamespace(
                trtllm_situ_b552_native_abi_digest=lambda: bad_native_chunks,
                trtllm_situ_b552_manifest_digest=lambda: chunks,
            )

    module.gen_jit_spec = lambda *_args, **_kwargs: MismatchedNativeAotSpec()
    with pytest.raises(RuntimeError, match="JIT-cache/native module ABI mismatch"):
        module.load_trtllm_gen_fused_moe_situ_b552_module()

    class MissingFfiExportAotSpec:
        is_aot = True

        def build_and_load(self):
            exports = {
                export_name: (lambda *_args, **_kwargs: None)
                for export_name in module.NATIVE_FFI_ABI.values()
            }
            exports["trtllm_situ_b552_native_abi_digest"] = lambda: native_chunks
            exports["trtllm_situ_b552_manifest_digest"] = lambda: chunks
            del exports[module.NATIVE_FFI_ABI["pre_routed_moe"]]
            return types.SimpleNamespace(**exports)

    module.gen_jit_spec = lambda *_args, **_kwargs: MissingFfiExportAotSpec()
    with pytest.raises(RuntimeError, match="missing contract-bound FFI exports"):
        module.load_trtllm_gen_fused_moe_situ_b552_module()


def test_private_native_patch_ports_precomputed_routing_without_mutating_sources(
    tmp_path,
):
    module, _env = _load_runtime_module(tmp_path)
    source = tmp_path / "immutable"
    source.mkdir()

    header = source / "runner.h"
    header.write_text(
        "enum class RoutingMethodType : int64_t {\n"
        "  // TopK only (no softmax)\n"
        "  TopK = 5,\n"
        "  // Unspecified\n"
        "  Unspecified = 6,\n"
        "};\n"
        "inline std::string serializeMoeRoutingMethodType("
        "RoutingMethodType routingMethodType) {\n"
        "  switch (routingMethodType) {\n"
        "    case RoutingMethodType::TopK:\n"
        '      return "TopK";\n'
        "    default:\n"
        '      return "InvalidRountingMethod";\n'
        "  };\n"
        "}\n"
        "           bool useDeepSeekFp8, RoutingMethodType routingMethodType, "
        "cudaStream_t stream);\n"
    )
    runner = source / "runner.cu"
    runner.write_text(
        "// immutable b552 prefix\n"
        "namespace Routing {\n"
        "void Runner::run(\n"
        "                 RoutingMethodType routingMethodType, "
        "cudaStream_t stream) {\n"
        "  oldRouting();\n"
        "}  // namespace Routing\n"
        "namespace PermuteGemm1 { int old_b552_gemm = 1; }\n"
    )
    current_runner = module._routing_source_path("csrc/trtllm_fused_moe_runner.cu")
    launcher = source / "launcher.cu"
    launcher.write_text(
        "#include <cmath>\n"
        '#include "flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/'
        'GemmGatedActOptions.h"\n'
        "namespace flashinfer {\n"
        "void run_base() {\n"
        "    routing_runner.run(\n"
        "        static_cast<RoutingMethodType>(routing_method_type), routing_stream);\n"
        "}\n"
        "void run_fp8() {\n"
        "    routing_runner.run(\n"
        "        static_cast<RoutingMethodType>(routing_method_type), routing_stream);\n"
        "}\n"
        "void prepare_routing() {\n"
        "    if (routing_logits.has_value()) {\n"
        "      if (static_cast<RoutingMethodType>(routing_method_type) == "
        "RoutingMethodType::DeepSeekV3) {\n"
        "        TVM_FFI_ICHECK_EQ(routing_logits.value().dtype(), dl_float32)\n"
        '            << "routing_logits must be float.";\n'
        "        mDtypeScore = btg::Dtype::Fp32;\n"
        "      } else if (routing_logits.value().dtype() == dl_float32) {\n"
        "        mDtypeScore = btg::Dtype::Fp32;\n"
        "      } else {\n"
        "        mDtypeScore = btg::Dtype::Bfloat16;\n"
        "      }\n"
        "    }\n"
        "}\n"
        "class FP4BlockScaleLauncher {\n"
        "  FP4BlockScaleLauncher(\n"
        "      TensorView const& expert_indices,\n"
        "      TensorView const& expert_weights)\n"
        "      : expert_indices(expert_indices),\n"
        "        expert_weights(expert_weights) {}\n"
        "    auto routing_bias_dtype = routing_bias.has_value() ? "
        "routing_bias.value().dtype() : dl_bfloat16;\n"
        "    mRoutingBiasDtype = routing_bias_dtype == dl_bfloat16 ? "
        "btg::Dtype::Bfloat16 : btg::Dtype::Fp32;\n"
        "    workspace.routing_expert_indexes =\n"
        "        static_cast<int*>(const_cast<void*>(expert_indices.data_ptr()));\n"
        "    workspace.expert_weights = const_cast<void*>(expert_weights.data_ptr());\n"
        "    cudaStream_t routing_stream = get_stream(hidden_states.device());\n\n"
        "    routing_runner.run(\n"
        "        args->routed_scaling_factor, static_cast<int*>(expert_indices.data_ptr()),\n"
        "        static_cast<int*>(permuted_idx_to_token_idx.data_ptr()), "
        "expert_weights.data_ptr(),\n"
        "        static_cast<RoutingMethodType>(routing_method_type), routing_stream);\n"
        "  TensorView expert_indices;\n"
        "  TensorView expert_weights;\n"
        "  Array<Tensor> run() {\n"
        "    return {gemm2_output, FusedMoeLauncher::expert_weights, "
        "expanded_idx_to_permuted_idx};\n"
        "  }\n"
        "};\n"
        "Array<Tensor> trtllm_bf16_moe() {}\n"
        "Array<Tensor> trtllm_fp4_block_scale_moe(\n"
        "    Optional<TensorView> routing_logits, TensorView expert_indices, "
        "TensorView expert_weights,\n"
        "    Optional<TensorView> routing_bias) {}\n"
        "TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_bf16_moe, trtllm_bf16_moe);\n"
        "TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_fp8_per_tensor_scale_moe, "
        "trtllm_fp8_per_tensor_scale_moe);\n"
        "TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_fp8_block_scale_moe, "
        "trtllm_fp8_block_scale_moe);\n"
        "TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_fp4_block_scale_moe, "
        "trtllm_fp4_block_scale_moe);\n"
        "TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_mxint4_block_scale_moe, "
        "trtllm_mxint4_block_scale_moe);\n"
        "TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_get_valid_moe_configs, "
        "trtllm_get_valid_moe_configs);\n"
    )
    originals = {
        path: path.read_bytes() for path in (header, runner, current_runner, launcher)
    }

    patched_header = tmp_path / "generated/runner.h"
    patched_runner = tmp_path / "generated/runner.cu"
    patched_launcher = tmp_path / "generated/launcher.cu"
    module._patch_runner_header(header, patched_header)
    module._patch_runner_source(runner, current_runner, patched_runner)
    module._patch_launcher(launcher, patched_launcher)

    assert all(path.read_bytes() == contents for path, contents in originals.items())
    header_text = patched_header.read_text()
    runner_text = patched_runner.read_text()
    launcher_text = patched_launcher.read_text()
    assert header_text.count("int32_t* expertIds") == 1
    assert "SigmoidRenorm = 6" in header_text
    assert "MiniMax2 = 7" in header_text
    assert "Sigmoid = 8" in header_text
    assert "Unspecified = 9" in header_text
    assert 'return "SigmoidRenorm";' in header_text
    assert 'return "MiniMax2";' in header_text
    assert 'return "Sigmoid";' in header_text
    assert runner_text.count("int32_t* expertIds") == 1
    assert "auto const dtypeLogits = dtypeScore;" in runner_text
    assert "bool const normTopkProb = true;" in runner_text
    assert "int16_t* const routing_replay_out = nullptr;" in runner_text
    assert (
        "routingMethodType == RoutingMethodType::DeepSeekV3 && nGroup <= 1"
        in runner_text
    )
    assert "routingCustom::Data routingData;" in runner_text
    assert "RoutingPreprocessType::SigmoidBias" in runner_text
    assert "RoutingPostprocessType::ScaledSumNormalize" in runner_text
    assert runner_text.count("routingData.mPtrTopKIds = expertIds") == 5
    assert "namespace PermuteGemm1 { int old_b552_gemm = 1; }" in runner_text
    assert "oldRouting();" not in runner_text
    assert "Pre-routed IDs are read-only input" in launcher_text
    assert "FusedMoeLauncher::expert_indexes = alloc_tensor" in launcher_text
    assert "Tensor const& expert_weights" in launcher_text
    assert "Tensor routing_expert_weights;" in launcher_text
    assert (
        "workspace.expert_weights = routing_expert_weights.data_ptr();" in launcher_text
    )
    assert (
        "return {gemm2_output, routing_expert_weights, "
        "expanded_idx_to_permuted_idx};" in launcher_text
    )
    assert "FusedMoeLauncher::expert_weights" not in launcher_text
    assert (
        "Optional<TensorView> routing_logits, TensorView expert_indices, "
        "Tensor expert_weights," in launcher_text
    )
    assert (
        "args->routed_scaling_factor, workspace.routing_expert_indexes" in launcher_text
    )
    assert launcher_text.count("routing_runner.run(") == 3
    assert launcher_text.count("routing_stream, nullptr);") == 2
    assert launcher_text.count("routing_stream, expert_ids);") == 1
    assert "routing_logits must be float." not in launcher_text
    assert (
        "static_cast<RoutingMethodType>(routing_method_type) == "
        "RoutingMethodType::DeepSeekV3" not in launcher_text
    )
    assert launcher_text.count("mDtypeScore = btg::Dtype::Bfloat16;") == 1
    assert (
        "if (routing_logits.has_value()) {\n"
        "      mDtypeScore = routing_logits.value().dtype() == dl_float32\n"
        "                        ? btg::Dtype::Fp32\n"
        "                        : btg::Dtype::Bfloat16;\n"
        "    }" in launcher_text
    )
    assert launcher_text.count("static_assert(") == 10
    assert "sizeof(::batchedGemm::KernelParams) == 17408" in launcher_text
    assert "alignof(::batchedGemm::KernelParams) == 128" in launcher_text
    assert "MoE::ActivationType::Swiglu) == 3" in launcher_text
    assert "gemmGatedAct::ActType::SwiGlu) == 0" in launcher_text
    for member, offset in module.KERNEL_PARAMS_ABI["offsets"].items():
        assert (
            f"offsetof(::batchedGemm::KernelParams, {member}) == {offset}"
            in launcher_text
        )
    assert launcher_text.count("trtllm_fp4_block_scale_situ_logits_moe") == 1
    assert launcher_text.count("trtllm_fp4_block_scale_situ_routed_moe") == 1
    assert "trtllm_fp4_block_scale_situ_moe" not in launcher_text
    assert launcher_text.count("trtllm_situ_b552_manifest_digest") == 3
    assert launcher_text.count("trtllm_situ_b552_native_abi_digest") == 3
    assert "trtllm_situ_b552_manifest_tag" not in launcher_text


def test_private_include_overlay_contains_runner_sibling_headers(tmp_path):
    module, _env = _load_runtime_module(tmp_path)
    source_root = tmp_path / "immutable"
    fused_moe = source_root / "include/flashinfer/trtllm/fused_moe"
    fused_moe.mkdir(parents=True)
    runner = fused_moe / "runner.h"
    runner.write_text(
        '#include "DevKernel.h"\n'
        '#include "RoutingKernel.h"\n'
        "enum class RoutingMethodType : int64_t {\n"
        "  // TopK only (no softmax)\n"
        "  TopK = 5,\n"
        "  // Unspecified\n"
        "  Unspecified = 6,\n"
        "};\n"
        "inline std::string serializeMoeRoutingMethodType("
        "RoutingMethodType routingMethodType) {\n"
        "  switch (routingMethodType) {\n"
        "    case RoutingMethodType::TopK:\n"
        '      return "TopK";\n'
        "    default:\n"
        '      return "InvalidRountingMethod";\n'
        "  };\n"
        "}\n"
        "           bool useDeepSeekFp8, RoutingMethodType routingMethodType, "
        "cudaStream_t stream);\n"
    )
    (fused_moe / "DevKernel.h").write_text("// immutable DevKernel\n")
    (fused_moe / "RoutingKernel.h").write_text("// immutable RoutingKernel\n")
    cuda_utils = source_root / "include/flashinfer/trtllm/common/cudaUtils.h"
    cuda_utils.parent.mkdir(parents=True)
    cuda_utils.write_text(
        "#include <driver_types.h>\n"
        "namespace tensorrt_llm::common {\n"
        "inline int getMultiProcessorCount() {\n"
        "  return 1;\n"
        "}\n"
        "}  // namespace tensorrt_llm::common\n"
    )
    cuda_utils_original = cuda_utils.read_text()
    shared_header = source_root / "include/flashinfer/exception.h"
    shared_header.write_text("// immutable exception\n")
    cubin_loader = source_root / "include/flashinfer/cubin_loader.h"
    cubin_loader.write_text(
        'extern "C" void FlashInferSetCubinCallback(void (*callback)());\n'
        'extern "C" void FlashInferSetCurrentCubin(const char* binary, int size);\n'
    )
    cubin_loader_original = cubin_loader.read_text()
    stock_generated = (
        source_root / "include/flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export"
    )
    stock_generated.mkdir(parents=True)
    (stock_generated / "stock.h").write_text("// stock generated header\n")

    artifact_root = tmp_path / "cubins"
    (artifact_root / "include/trtllmGen_bmm_export").mkdir(parents=True)
    routing_sources = module._verify_current_routing_sources()
    routing_dev_kernel = routing_sources[
        "include/flashinfer/trtllm/fused_moe/RoutingDevKernel.h"
    ]
    routing_dev_kernel_original = routing_dev_kernel.read_bytes()
    include_root = module._prepare_include_overlay(
        tmp_path / "generated", artifact_root, source_root, routing_sources
    )
    stale = include_root / "flashinfer/trtllm/fused_moe/stale-manual-header.h"
    stale.write_text("must not survive overlay recreation\n")
    assert (
        module._prepare_include_overlay(
            tmp_path / "generated", artifact_root, source_root, routing_sources
        )
        == include_root
    )
    assert not stale.exists()
    overlay = include_root / "flashinfer/trtllm/fused_moe"

    assert (overlay / "DevKernel.h").read_text() == "// immutable DevKernel\n"
    assert (overlay / "RoutingKernel.h").read_bytes() == routing_sources[
        "include/flashinfer/trtllm/fused_moe/RoutingKernel.h"
    ].read_bytes()
    assert (fused_moe / "RoutingKernel.h").read_text() == (
        "// immutable RoutingKernel\n"
    )
    for relative in module._ROUTING_HEADERS:
        if relative.endswith("/RoutingDevKernel.h"):
            continue
        assert (
            include_root / Path(relative).relative_to("include")
        ).read_bytes() == routing_sources[relative].read_bytes()
    assert (
        include_root / "flashinfer/exception.h"
    ).read_text() == "// immutable exception\n"
    private_cubin_loader = (include_root / "flashinfer/cubin_loader.h").read_text()
    assert private_cubin_loader.count('visibility("default")') == 2
    assert "FlashInferSetCubinCallback" in private_cubin_loader
    assert "FlashInferSetCurrentCubin" in private_cubin_loader
    assert cubin_loader.read_text() == cubin_loader_original
    assert cuda_utils.read_text() == cuda_utils_original
    private_cuda_utils = (
        include_root / "flashinfer/trtllm/common/cudaUtils.h"
    ).read_text()
    assert "#ifdef ENABLE_FP8\n#include <cuda_fp8.h>\n#endif" in private_cuda_utils
    assert private_cuda_utils.count("inline int getSMVersion()") == 1
    assert private_cuda_utils.count("inline int getMultiProcessorCount()") == 1
    private_routing_dev_kernel = (overlay / "RoutingDevKernel.h").read_text()
    for macro in (
        "LAUNCH_ROUTING_LLAMA4",
        "LAUNCH_ROUTING_WITH_NUM_EXPERTS_FORCE_FLOAT_INPUT",
    ):
        assert private_routing_dev_kernel.count(f"#undef {macro}") == 1
        assert f"#undef {macro}\n#define {macro}" in private_routing_dev_kernel
    assert routing_dev_kernel.read_bytes() == routing_dev_kernel_original
    assert "int32_t* expertIds" in (overlay / "runner.h").read_text()
    assert "int32_t* expertIds" not in runner.read_text()
    assert (
        include_root / "flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export"
    ).resolve() == (artifact_root / "include/trtllmGen_bmm_export").resolve()


def test_private_cubin_overlay_preserves_the_complete_stock_tree(tmp_path):
    path = Path(__file__).parents[1] / "flashinfer-cubin/build_backend.py"
    tree = ast.parse(path.read_text())
    names = {
        "_SITU_OVERLAY_ROOT",
        "_snapshot_stock_cubin_tree",
        "_assert_stock_cubin_tree_unchanged",
    }
    nodes = [
        node
        for node in tree.body
        if (
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and any(
                isinstance(target, ast.Name) and target.id in names
                for target in (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
            )
        )
        or (isinstance(node, ast.FunctionDef) and node.name in names)
    ]
    namespace = {"Path": Path, "hashlib": hashlib}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), namespace)

    cubin_dir = tmp_path / "cubins"
    stock_cubin = cubin_dir / "stock/batched_gemm/kernel.cubin"
    stock_header = cubin_dir / "stock/batched_gemm/metadata.h"
    private_cubin = cubin_dir / "fireworks/situ_b552/private.cubin"
    for artifact, contents in (
        (stock_cubin, b"stock-cubin"),
        (stock_header, b"stock-header"),
        (private_cubin, b"private-v1"),
    ):
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(contents)

    snapshot = namespace["_snapshot_stock_cubin_tree"](cubin_dir)
    assert set(snapshot) == {
        "stock/batched_gemm/kernel.cubin",
        "stock/batched_gemm/metadata.h",
    }

    private_cubin.write_bytes(b"private-v2")
    namespace["_assert_stock_cubin_tree_unchanged"](
        snapshot, namespace["_snapshot_stock_cubin_tree"](cubin_dir)
    )

    stock_cubin.write_bytes(b"modified-stock-cubin")
    with pytest.raises(RuntimeError, match="modified=.*kernel.cubin"):
        namespace["_assert_stock_cubin_tree_unchanged"](
            snapshot, namespace["_snapshot_stock_cubin_tree"](cubin_dir)
        )

    download_source = path.read_text().split("def _download_cubins", 1)[1]
    staged_overlay_index = download_source.index(
        "_build_situ_b552_overlay(staged_cubin_dir)"
    )
    download_index = download_source.index("artifacts.download_artifacts()")
    authenticated_index = download_source.index(
        "_verify_authenticated_stock_cubin_tree("
    )
    install_index = download_source.index(
        "_install_staged_situ_b552_overlay(staged_cubin_dir, cubin_dir)"
    )
    after_index = download_source.index(
        "stock_after = _snapshot_stock_cubin_tree(cubin_dir)"
    )
    exact_after_index = download_source.index(
        "_assert_exact_stock_cubin_tree(expected_stock, stock_after)"
    )
    assert (
        staged_overlay_index
        < download_index
        < authenticated_index
        < install_index
        < after_index
        < exact_after_index
    )


def _function_args(tree: ast.Module, name: str) -> list[str]:
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    return [argument.arg for argument in function.args.args]


def _load_core_adapter_helpers():
    path = Path(__file__).parents[1] / "flashinfer/fused_moe/core.py"
    tree = ast.parse(path.read_text())
    names = {
        "_adapt_situ_b552_fp4_args",
        "_adapt_situ_b552_valid_config_args",
        "_make_situ_b552_topk_initializer",
        "_parse_situ_b552_pre_routed_inputs",
        "_situ_b552_autotune_cache_name",
        "_validate_situ_b552_boundary",
        "_validate_situ_b552_routing_tensors",
    }
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {function.name for function in functions} == names

    class RoutingInputMode(IntEnum):
        FromLogits = 0
        PackedPrecomputed = 1
        UnpackedPrecomputed = 2

    class RoutingMethodType(IntEnum):
        Renormalize = 1
        DeepSeekV3 = 2
        Llama4 = 3
        RenormalizeNaive = 4
        TopK = 5

    class ActivationType(IntEnum):
        Swiglu = 3

    class Device:
        def __init__(self, device_type, index=None):
            self.type = device_type
            self.index = index

        def __eq__(self, other):
            return (
                isinstance(other, Device)
                and self.type == other.type
                and self.index == other.index
            )

        def __repr__(self):
            suffix = "" if self.index is None else f":{self.index}"
            return f"{self.type}{suffix}"

    class BoolTensor:
        def __init__(self, values):
            self.values = list(values)

        def __and__(self, other):
            return BoolTensor(
                left and right
                for left, right in zip(self.values, other.values, strict=False)
            )

        def all(self):
            return all(self.values)

    class Tensor:
        def __init__(self, shape, dtype, device, contiguous=True, values=None):
            self.shape = shape
            self.dtype = dtype
            self.device = device
            self._contiguous = contiguous
            self.values = [1.0] if values is None else list(values)

        def is_contiguous(self):
            return self._contiguous

        def __gt__(self, lower_bound):
            return BoolTensor(value > lower_bound for value in self.values)

    class InitializerTensor:
        def __init__(self, values, dtype="int32", shape=None):
            self.values = list(values)
            self.dtype = dtype
            self.shape = shape

        def view(self, shape_or_dtype):
            if isinstance(shape_or_dtype, tuple):
                self.shape = shape_or_dtype
            else:
                self.dtype = shape_or_dtype
            return self

        def __lshift__(self, bits):
            return InitializerTensor(
                [value << bits for value in self.values], self.dtype, self.shape
            )

        def __or__(self, other):
            return InitializerTensor(
                [
                    left | right
                    for left, right in zip(self.values, other.values, strict=False)
                ],
                self.dtype,
                self.shape,
            )

    def make_random_topk_ids(num_experts, num_tokens, top_k, device):
        del device
        return InitializerTensor(
            [index % num_experts for index in range(num_tokens * top_k)]
        )

    def ones(shapes, dtype, device):
        del dtype, device
        return InitializerTensor([0x3F80] * math.prod(shapes), shape=shapes)

    def assert_async(condition, message):
        if not condition:
            raise ValueError(message)

    namespace = {
        "ActivationType": ActivationType,
        "RoutingMethodType": RoutingMethodType,
        "RoutingInputMode": RoutingInputMode,
        "Device": Device,
        "Tensor": Tensor,
        "get_compute_capability": lambda _device: (10, 3),
        "make_random_topk_ids": make_random_topk_ids,
        "math": math,
        "Optional": Optional,
        "torch": types.SimpleNamespace(
            _assert_async=assert_async,
            Tensor=Tensor,
            int16="int16",
            int32="int32",
            int64="int64",
            bfloat16="bfloat16",
            float32="float32",
            float8_e4m3fn="float8_e4m3fn",
            uint8="uint8",
            all=lambda value: value.all(),
            isfinite=lambda value: BoolTensor(
                math.isfinite(item) for item in value.values
            ),
            ones=ones,
        ),
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), path, "exec"), namespace)
    return namespace


def test_current_to_b552_abi_adapters_preserve_buffers_and_argument_order():
    helpers = _load_core_adapter_helpers()

    class Buffer:
        shape = (37, 8)

        def __init__(self, values=None, dtype=None):
            self.values = values
            self.dtype = dtype
            self.copied_from = None

        def __rshift__(self, bits):
            return Buffer([value >> bits for value in self.values], self.dtype)

        def __and__(self, mask):
            return Buffer([value & mask for value in self.values], self.dtype)

        def to(self, dtype):
            return Buffer(list(self.values), dtype)

        def view(self, dtype):
            return Buffer(list(self.values), dtype)

        def copy_(self, source):
            self.values = list(source.values)
            self.dtype = source.dtype
            self.copied_from = source
            return self

    routing_logits = object()
    topk_ids = Buffer(dtype="int32")
    topk_weights = Buffer(dtype="bfloat16")
    current_fp4_args = [object() for _ in range(36)]
    current_fp4_args[0] = 0  # RoutingInputMode.FromLogits
    current_fp4_args[1] = routing_logits
    current_fp4_args[2] = topk_ids
    current_fp4_args[3] = topk_weights
    current_fp4_args[19] = None  # per_token_scale
    current_fp4_args[34] = True  # norm_topk_prob
    current_fp4_args[35] = None  # routing_replay_out

    native_fp4_args = helpers["_adapt_situ_b552_fp4_args"](current_fp4_args)
    assert len(native_fp4_args) == 32
    assert native_fp4_args[0] is routing_logits
    assert native_fp4_args[1] is topk_ids
    assert native_fp4_args[2] is topk_weights
    assert native_fp4_args[1].shape == (37, 8)
    assert native_fp4_args[2].shape == (37, 8)
    assert native_fp4_args[3:18] == current_fp4_args[4:19]
    assert native_fp4_args[18:] == current_fp4_args[20:34]

    current_fp4_args[3] = Buffer(dtype="float32")
    with pytest.raises(RuntimeError, match="allocated as bfloat16"):
        helpers["_adapt_situ_b552_fp4_args"](current_fp4_args)
    current_fp4_args[3] = topk_weights

    packed_ids = Buffer([(7 << 16) | 0x3F80, (384 << 16) | 0xC000], "int32")
    decoded_weights_output = Buffer(dtype="bfloat16")
    current_fp4_args[0] = 1  # RoutingInputMode.PackedPrecomputed
    current_fp4_args[1] = None
    current_fp4_args[2] = packed_ids
    current_fp4_args[3] = decoded_weights_output
    native_packed_args = helpers["_adapt_situ_b552_fp4_args"](current_fp4_args)
    assert packed_ids.values == [(7 << 16) | 0x3F80, (384 << 16) | 0xC000]
    assert native_packed_args[1] is not packed_ids
    assert native_packed_args[1].values == [7, 384]
    assert native_packed_args[2] is decoded_weights_output
    assert decoded_weights_output.values == [0x3F80, 0xC000]
    assert decoded_weights_output.copied_from is not None

    unpacked_ids = Buffer([7, 384], "int32")
    unpacked_weights = Buffer([0x3F80, 0xC000], "bfloat16")
    current_fp4_args[0] = 2  # RoutingInputMode.UnpackedPrecomputed
    current_fp4_args[2] = unpacked_ids
    current_fp4_args[3] = unpacked_weights
    native_unpacked_args = helpers["_adapt_situ_b552_fp4_args"](current_fp4_args)
    assert native_unpacked_args[1] is unpacked_ids
    assert native_unpacked_args[2] is unpacked_weights

    current_config_args = [object() for _ in range(13)]
    current_config_args[10] = False  # use_per_token_scaling
    current_config_args[12] = False  # has_gemm1_lora_delta
    native_config_args = helpers["_adapt_situ_b552_valid_config_args"](
        current_config_args
    )
    assert len(native_config_args) == 11
    assert native_config_args[:10] == tuple(current_config_args[:10])
    assert native_config_args[10] is current_config_args[11]  # num_tokens

    current_config_args[10] = True
    with pytest.raises(NotImplementedError, match="per-token scaling"):
        helpers["_adapt_situ_b552_valid_config_args"](current_config_args)
    current_config_args[10] = False
    current_config_args[12] = True
    with pytest.raises(NotImplementedError, match="expert LoRA delta"):
        helpers["_adapt_situ_b552_valid_config_args"](current_config_args)


def test_private_pre_routed_inputs_fail_closed_before_native_access():
    helpers = _load_core_adapter_helpers()
    Device = helpers["Device"]
    Tensor = helpers["Tensor"]
    parse = helpers["_parse_situ_b552_pre_routed_inputs"]
    cuda0 = Device("cuda", 0)
    hidden = Tensor((4, 128), "mxfp8", cuda0)

    packed = Tensor((4, 2), "int32", cuda0)
    mode, ids, weights = parse(packed, hidden, 2)
    assert mode == 1
    assert ids is packed
    assert weights is None

    unpacked_ids = Tensor((4, 2), "int32", cuda0)
    unpacked_weights = Tensor((4, 2), "bfloat16", cuda0)
    mode, ids, weights = parse((unpacked_ids, unpacked_weights), hidden, 2)
    assert mode == 2
    assert ids is unpacked_ids
    assert weights is unpacked_weights

    with pytest.raises(ValueError, match="must have shape"):
        parse(Tensor((4, 1), "int32", cuda0), hidden, 2)
    with pytest.raises(ValueError, match="must have shape"):
        parse(Tensor((8,), "int32", cuda0), hidden, 2)
    with pytest.raises(ValueError, match="must have dtype int32"):
        parse(Tensor((4, 2), "int64", cuda0), hidden, 2)
    with pytest.raises(ValueError, match="must be a CUDA tensor"):
        parse(Tensor((4, 2), "int32", Device("cpu")), hidden, 2)
    with pytest.raises(ValueError, match="must be on cuda:0"):
        parse(Tensor((4, 2), "int32", Device("cuda", 1)), hidden, 2)
    with pytest.raises(ValueError, match="must be contiguous"):
        parse(Tensor((4, 2), "int32", cuda0, contiguous=False), hidden, 2)
    with pytest.raises(ValueError, match="exactly a .* tuple"):
        parse((unpacked_ids,), hidden, 2)
    with pytest.raises(TypeError, match="topk_weights must be a torch.Tensor"):
        parse((unpacked_ids, object()), hidden, 2)
    with pytest.raises(ValueError, match="topk_weights must have dtype bfloat16"):
        parse((unpacked_ids, Tensor((4, 2), "float32", cuda0)), hidden, 2)
    with pytest.raises(ValueError, match="topk_ids must have shape"):
        parse((Tensor((8,), "int32", cuda0), unpacked_weights), hidden, 2)
    with pytest.raises(ValueError, match="topk_weights must be a CUDA tensor"):
        parse((unpacked_ids, Tensor((4, 2), "bfloat16", Device("cpu"))), hidden, 2)
    with pytest.raises(ValueError, match="topk_weights must be on cuda:0"):
        parse(
            (unpacked_ids, Tensor((4, 2), "bfloat16", Device("cuda", 1))),
            hidden,
            2,
        )
    with pytest.raises(ValueError, match="topk_weights must be contiguous"):
        parse(
            (
                unpacked_ids,
                Tensor((4, 2), "bfloat16", cuda0, contiguous=False),
            ),
            hidden,
            2,
        )
    with pytest.raises(ValueError, match="exactly a .* tuple"):
        parse((unpacked_ids, unpacked_weights, unpacked_weights), hidden, 2)
    with pytest.raises(TypeError, match="must be an int32 packed tensor"):
        parse([unpacked_ids, unpacked_weights], hidden, 2)


def test_private_boundary_enforces_exact_mxfp_contract_and_situ_parameters():
    helpers = _load_core_adapter_helpers()
    Device = helpers["Device"]
    Tensor = helpers["Tensor"]
    validate = helpers["_validate_situ_b552_boundary"]
    cuda0 = Device("cuda", 0)

    def tensor(shape, dtype, **kwargs):
        return Tensor(shape, dtype, cuda0, **kwargs)

    def valid_args():
        return {
            "routing_input_mode": 0,
            "routing_logits": tensor((4, 8), "float32"),
            "topk_ids": None,
            "topk_weights": None,
            "routing_bias": tensor((8,), "float32"),
            "hidden_states": tensor((4, 128), "float8_e4m3fn"),
            "hidden_states_scale": tensor((4, 4), "float8_e4m3fn"),
            "gemm1_weights": tensor((4, 128, 64), "uint8"),
            "gemm1_weights_scale": tensor((4, 128, 4), "float8_e4m3fn"),
            "gemm1_bias": None,
            "gemm1_alpha": tensor((4,), "float32", values=[4.0] * 4),
            "gemm1_beta": tensor((4,), "float32", values=[25.0] * 4),
            "gemm1_clamp_limit": None,
            "gemm2_weights": tensor((4, 128, 32), "uint8"),
            "gemm2_weights_scale": tensor((4, 128, 2), "float8_e4m3fn"),
            "gemm2_bias": None,
            "output1_scale_scalar": tensor((4,), "float32"),
            "output1_scale_gate_scalar": tensor((4,), "float32"),
            "output2_scale_scalar": tensor((4,), "float32"),
            "per_token_scale": None,
            "num_experts": 8,
            "top_k": 2,
            "n_group": None,
            "topk_group": None,
            "intermediate_size": 64,
            "local_expert_offset": 2,
            "local_num_experts": 4,
            "routed_scaling_factor": None,
            "routing_method_type": 1,
            "do_finalize": True,
            "enable_pdl": True,
            "activation_type": 3,
            "output": tensor((4, 128), "bfloat16"),
            "tune_max_num_tokens": 32,
            "norm_topk_prob": True,
            "routing_replay_out": None,
        }

    assert validate(**valid_args()) == 0
    without_linear_clamp = valid_args()
    without_linear_clamp["gemm1_beta"] = None
    assert validate(**without_linear_clamp) == 0

    packed = valid_args()
    packed.update(
        routing_input_mode=1,
        routing_logits=None,
        topk_ids=tensor((4, 2), "int32"),
    )
    assert validate(**packed) == 1

    unpacked = valid_args()
    unpacked.update(
        routing_input_mode=2,
        routing_logits=None,
        topk_ids=tensor((4, 2), "int32"),
        topk_weights=tensor((4, 2), "bfloat16"),
    )
    assert validate(**unpacked) == 2

    invalid_tensors = (
        (
            "hidden_states_scale",
            tensor((4, 8), "float8_e4m3fn"),
            "hidden_states_scale must have shape",
        ),
        (
            "gemm1_weights",
            tensor((4, 128, 64), "float8_e4m3fn"),
            "gemm1_weights must have dtype uint8",
        ),
        (
            "gemm1_weights_scale",
            Tensor((4, 128, 4), "float8_e4m3fn", Device("cuda", 1)),
            "gemm1_weights_scale must be on cuda:0",
        ),
        (
            "gemm2_weights",
            tensor((4, 128, 64), "uint8"),
            "gemm2_weights must have shape",
        ),
        (
            "gemm2_weights_scale",
            tensor((4, 128, 2), "float8_e4m3fn", contiguous=False),
            "gemm2_weights_scale must be contiguous",
        ),
        (
            "output1_scale_scalar",
            None,
            "output1_scale_scalar is required",
        ),
        (
            "output",
            tensor((4, 64), "bfloat16"),
            "output must have shape",
        ),
        (
            "routing_logits",
            tensor((4, 8), "float8_e4m3fn"),
            "routing_logits must have dtype",
        ),
    )
    for name, value, match in invalid_tensors:
        args = valid_args()
        args[name] = value
        with pytest.raises(ValueError, match=match):
            validate(**args)

    for values in ([0.0] * 4, [-1.0] * 4, [float("inf")] * 4, [float("nan")] * 4):
        args = valid_args()
        args["gemm1_alpha"] = tensor((4,), "float32", values=values)
        with pytest.raises(ValueError, match="finite, strictly positive"):
            validate(**args)

    invalid_private_options = (
        ("activation_type", 4, "activation ABI sentinel"),
        ("per_token_scale", object(), "per-token scaling"),
        ("norm_topk_prob", False, "norm_topk_prob=True"),
        ("routing_replay_out", object(), "routing replay output"),
        ("gemm1_bias", object(), "expert GEMM bias"),
        ("gemm1_clamp_limit", object(), "gemm1_clamp_limit=None"),
    )
    for name, value, match in invalid_private_options:
        args = valid_args()
        args[name] = value
        with pytest.raises((ValueError, NotImplementedError), match=match):
            validate(**args)


def test_private_autotune_topk_initializers_and_cache_keys_are_mode_specific():
    helpers = _load_core_adapter_helpers()
    make_initializer = helpers["_make_situ_b552_topk_initializer"]
    cache_name = helpers["_situ_b552_autotune_cache_name"]
    device = helpers["Device"]("cuda", 0)

    packed = make_initializer(1, 8)((3, 2), "int32", device)
    assert packed.shape == (3, 2)
    assert [value >> 16 for value in packed.values] == list(range(6))
    assert [value & 0xFFFF for value in packed.values] == [0x3F80] * 6

    unpacked = make_initializer(2, 8)((3, 2), "int32", device)
    logits_output = make_initializer(0, 8)((3, 2), "int32", device)
    assert unpacked.values == list(range(6))
    assert logits_output.values == list(range(6))
    assert all(value < 8 for value in unpacked.values)

    names = [cache_name(mode) for mode in range(3)]
    assert len(set(names)) == 3
    assert names == [
        "flashinfer::trtllm_fp4_block_scale_moe_situ_b552::fromlogits",
        "flashinfer::trtllm_fp4_block_scale_moe_situ_b552::packedprecomputed",
        "flashinfer::trtllm_fp4_block_scale_moe_situ_b552::unpackedprecomputed",
    ]


def test_private_situ_apis_have_no_activation_type_or_public_export():
    path = Path(__file__).parents[1] / "flashinfer/fused_moe/core.py"
    tree = ast.parse(path.read_text())

    logits = _function_args(tree, "trtllm_fp4_block_scale_situ_moe")
    routed = _function_args(tree, "trtllm_fp4_block_scale_situ_routed_moe")
    assert "activation_type" not in logits
    assert "activation_type" not in routed

    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    for name in (
        "trtllm_fp4_block_scale_situ_moe",
        "trtllm_fp4_block_scale_situ_routed_moe",
    ):
        assert functions[name].decorator_list == []

    logits_source = ast.get_source_segment(
        path.read_text(), functions["trtllm_fp4_block_scale_situ_moe"]
    )
    routed_source = ast.get_source_segment(
        path.read_text(), functions["trtllm_fp4_block_scale_situ_routed_moe"]
    )
    assert logits_source is not None
    assert routed_source is not None
    assert ".trtllm_fp4_block_scale_moe(" in logits_source
    assert ".trtllm_fp4_block_scale_routed_moe(" in routed_source
    assert "_parse_situ_b552_pre_routed_inputs(" in routed_source

    package_root = Path(__file__).parents[1] / "flashinfer"
    root_exports = (package_root / "__init__.py").read_text()
    fused_exports = (package_root / "fused_moe/__init__.py").read_text()
    for name in (
        "trtllm_fp4_block_scale_situ_moe",
        "trtllm_fp4_block_scale_situ_routed_moe",
    ):
        assert name not in root_exports
        assert name not in fused_exports

    private_surface = (package_root / "fused_moe/_situ_b552.py").read_text()
    assert "Unsupported Fireworks-private ABI surface" in private_surface
    assert "trtllm_fp4_block_scale_situ_moe" in private_surface
    assert "trtllm_fp4_block_scale_situ_routed_moe" in private_surface

    stock_logits = _function_args(tree, "trtllm_fp4_block_scale_moe")
    stock_routed = _function_args(tree, "trtllm_fp4_block_scale_routed_moe")
    assert logits == [name for name in stock_logits if name != "activation_type"]
    assert routed == [name for name in stock_routed if name != "activation_type"]


def test_private_factory_registers_only_fp4_and_exposes_valid_configs():
    path = Path(__file__).parents[1] / "flashinfer/fused_moe/core.py"
    tree = ast.parse(path.read_text())
    factory = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_get_trtllm_moe_sm100_module"
    )

    situ_custom_ops = [
        node.value
        for node in ast.walk(factory)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "situ_custom_ops"
            for target in node.targets
        )
    ]
    assert len(situ_custom_ops) == 1
    assert isinstance(situ_custom_ops[0], ast.Set)
    assert {element.value for element in situ_custom_ops[0].elts} == {
        "trtllm_fp4_block_scale_moe"
    }

    namespace_shapes = [
        {keyword.arg for keyword in node.keywords}
        for node in ast.walk(factory)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "SimpleNamespace"
    ]
    native_private_shape = {
        "trtllm_fp4_block_scale_logits_moe",
        "trtllm_fp4_block_scale_routed_moe",
        "trtllm_get_valid_moe_configs",
    }
    exposed_private_shape = {
        "trtllm_fp4_block_scale_moe",
        "trtllm_fp4_block_scale_routed_moe",
        "trtllm_get_valid_moe_configs",
    }
    assert namespace_shapes.count(native_private_shape) == 1
    assert namespace_shapes.count(exposed_private_shape) == 1
    assert {
        "trtllm_bf16_moe",
        "trtllm_fp8_per_tensor_scale_moe",
        "trtllm_fp8_block_scale_moe",
        "trtllm_fp4_block_scale_moe",
        "trtllm_mxint4_block_scale_moe",
    } in namespace_shapes

    nested_functions = {
        node.name: node for node in factory.body if isinstance(node, ast.FunctionDef)
    }
    for helper_name in ("register_variant_custom_op", "register_variant_fake_op"):
        helper_source = ast.get_source_segment(
            path.read_text(), nested_functions[helper_name]
        )
        assert helper_source is not None
        assert 'variant == "situ_b552" and name not in situ_custom_ops' in helper_source
        assert "return lambda function: function" in helper_source

    factory_source = ast.get_source_segment(path.read_text(), factory)
    assert factory_source is not None
    assert "private_op.trtllm_fp4_block_scale_situ_logits_moe" in factory_source
    assert "private_op.trtllm_fp4_block_scale_situ_routed_moe" in factory_source
    assert "private_op.trtllm_fp4_block_scale_situ_moe" not in factory_source
    assert "private_op.trtllm_get_valid_situ_moe_configs" in factory_source

    private_fp4 = next(
        node
        for node in ast.walk(factory)
        if isinstance(node, ast.FunctionDef)
        and node.name == "trtllm_fp4_block_scale_moe_op"
    )
    private_fp4_source = ast.get_source_segment(path.read_text(), private_fp4)
    assert private_fp4_source is not None
    boundary_index = private_fp4_source.index("_validate_situ_b552_boundary(")
    tune_index = private_fp4_source.index("tuner.choose_one(")
    assert boundary_index < tune_index
    assert "routing_input_mode=private_routing_mode" in private_fp4_source
    assert "_situ_b552_autotune_cache_name(private_routing_mode)" in private_fp4_source
