import importlib.util
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

_SWIGLU_SNIPPET = """            cutlass::Array<float, 2> fusedScaleArray{(mGatedActAlpha) * (float{1.442695}),
                                                             (mGatedActAlpha) * (float{1.442695})};
            cutlass::Array<float, 2> betaScaleGateArray{mGatedActBeta, mGatedActBeta};
            cutlass::Array<float, 2> scaleGateArray{float{1}, float{1}};
            cutlass::Array<float, 2> x0ScaleGateArray;
            x0ScaleGateArray = trtllm::dev::ffma2(x0Array, scaleGateArray, betaScaleGateArray);
            cutlass::Array<float, 2> x1ScaledArray;
            x1ScaledArray = trtllm::dev::fmul2(x1Array, fusedScaleArray);
            cutlass::Array<float, 2> actArray;
            actArray = trtllm::dev::sigmoid2_base2(x1ScaledArray);
            cutlass::Array<float, 2> swishArray;
            swishArray = trtllm::dev::fmul2(x1Array, actArray);
            cutlass::Array<float, 2> gatedActArray;
            gatedActArray = trtllm::dev::fmul2(x0ScaleGateArray, swishArray);"""


def _load_script_module():
    path = Path(__file__).parents[1] / "scripts/situ_b552/build.py"
    spec = importlib.util.spec_from_file_location("situ_b552_build", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_source_snapshot(module, root: Path) -> list[Path]:
    source_dir = root / module.SOURCE_RELATIVE_DIR
    source_dir.mkdir(parents=True)
    for index in range(102):
        stem = (
            "Bmm_MxE4m3_MxE2m1MxE4m3_Fp32_bA32_bB32_bC32_"
            f"case{index:03d}_swiGlu_dynB_sm100f"
        )
        # 92 * 7 + 10 * 6 = the exact 704 PR #2917 epilogues.
        num_epilogues = 7 if index < 92 else 6
        (source_dir / f"{stem}.cu").write_text(
            ("\n".join([_SWIGLU_SNIPPET] * num_epilogues)) + "\n"
        )
        (source_dir / f"{stem}.h").write_text(f"// generated symbol: {stem}\n")
    return sorted(source_dir.iterdir())


def _accept_synthetic_snapshot(module, root: Path) -> list[Path]:
    files = _make_source_snapshot(module, root)
    module.SOURCE_MANIFEST_SHA256 = module._source_manifest_sha256(files, root)
    return files


def test_situ_epilogue_rewrite_is_exact():
    module = _load_script_module()
    rewritten, count = module._SWIGLU_EPILOGUE.subn(
        module._situ_epilogue, _SWIGLU_SNIPPET
    )
    assert count == 1
    assert "gateTanhArray" in rewritten
    assert "gateSigmoidArray" in rewritten
    assert "upArray = x0Array" in rewritten
    assert "if (mGatedActBeta > float{0})" in rewritten
    assert "swishArray" not in rewritten
    assert "x0ScaleGateArray" not in rewritten


def test_exact_source_parent_child_counts_and_formula_invariants(tmp_path):
    module = _load_script_module()
    source_root = tmp_path / "source"
    parents = _accept_synthetic_snapshot(module, source_root)
    output_dir = tmp_path / "generated"

    cuda_children = module.transform_sources(source_root, output_dir)
    children = sorted(output_dir.iterdir())

    assert len(parents) == 204
    assert len([path for path in parents if path.suffix == ".cu"]) == 102
    assert len(cuda_children) == 102
    assert len(children) == 204
    assert all("_swiGlu_" in path.name for path in parents)
    assert all("_situ_" in path.name for path in children)

    generated_cuda = "\n".join(path.read_text() for path in cuda_children)
    assert generated_cuda.count("gateTanhInput =") == 704
    assert generated_cuda.count("upArray = x0Array") == 704
    assert (
        generated_cuda.count(
            "gateSigmoidInput = trtllm::dev::fmul2(x1Array, log2eArray)"
        )
        == 704
    )
    assert (
        generated_cuda.count(
            "gatedActArray = trtllm::dev::fmul2(upArray, gatedActArray)"
        )
        == 704
    )
    assert "swiGlu" not in generated_cuda
    assert "swishArray" not in generated_cuda


def test_source_aggregate_hash_rejects_modified_snapshot(tmp_path):
    module = _load_script_module()
    source_root = tmp_path / "source"
    files = _accept_synthetic_snapshot(module, source_root)
    assert module._source_files(source_root) == files

    files[0].write_text(files[0].read_text() + "// corruption\n")
    with pytest.raises(RuntimeError, match="not the immutable PR #2917 input"):
        module._source_files(source_root)


def test_artifact_file_modes_are_world_readable_and_non_executable(tmp_path):
    module = _load_script_module()
    artifact_root = tmp_path / "artifact"
    nested = artifact_root / "include/generated.h"
    nested.parent.mkdir(parents=True)
    nested.write_text("header\n")
    cubin = artifact_root / "kernel.cubin"
    cubin.write_bytes(b"cubin")
    outside = tmp_path / "immutable-source.h"
    outside.write_text("source\n")

    nested.chmod(0o600)
    cubin.chmod(0o755)
    outside.chmod(0o600)
    module._normalize_artifact_file_modes(artifact_root)

    for path in (nested, cubin):
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode == 0o644
        assert mode & stat.S_IROTH
        assert not mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    assert stat.S_IMODE(outside.stat().st_mode) == 0o600


def test_deterministic_work_root_is_locked_private_and_cleaned(tmp_path, monkeypatch):
    module = _load_script_module()
    work_root = tmp_path / "stable-work-root"
    monkeypatch.setattr(module, "DETERMINISTIC_WORK_ROOT", work_root)

    with module._locked_deterministic_work_root() as acquired:
        assert acquired == work_root
        assert stat.S_IMODE(work_root.stat().st_mode) == 0o700
        lock = work_root / module._WORK_ROOT_LOCK
        assert lock.is_file()
        assert stat.S_IMODE(lock.stat().st_mode) == 0o600
        (work_root / "generated").mkdir()
        (work_root / "generated/source.cu").write_text("kernel\n")

    assert work_root.is_dir()
    assert (work_root / module._WORK_ROOT_LOCK).is_file()
    assert sorted(path.name for path in work_root.iterdir()) == [module._WORK_ROOT_LOCK]


def test_deterministic_work_root_rejects_symlink(tmp_path, monkeypatch):
    module = _load_script_module()
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    work_root = tmp_path / "stable-work-root"
    work_root.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(module, "DETERMINISTIC_WORK_ROOT", work_root)

    with (
        pytest.raises(RuntimeError, match="private directory owned"),
        module._locked_deterministic_work_root(),
    ):
        raise AssertionError("unsafe work root was accepted")


def test_metainfo_rewrite_updates_only_symbol_and_hash(tmp_path):
    module = _load_script_module()
    cubin = tmp_path / (
        "Bmm_MxE4m3_MxE2m1MxE4m3_Fp32_bA32_bB32_bC32_"
        "t128x16x256_s3_situ_dynB_sm100f.cubin"
    )
    cubin.write_bytes(b"new-cubin")
    old_symbol = "bmm_" + cubin.stem.removeprefix("Bmm_").replace("_situ_", "_swiGlu_")
    metainfo = (
        '{nullptr, 0, 221504, "'
        + old_symbol
        + '", 512, "'
        + ("a" * 64)
        + '", "", nullptr},\n'
    )
    rewritten = module._rewrite_metainfo(metainfo, [cubin])
    assert old_symbol not in rewritten
    assert old_symbol.replace("_swiGlu_", "_situ_") in rewritten
    assert "{nullptr, 0, 221504," in rewritten
    assert f"{{nullptr, 0, {cubin.stat().st_size}," not in rewritten
    assert module._sha256(cubin.read_bytes()) in rewritten


def test_private_contract_counts_and_abi_are_pinned():
    module = _load_script_module()
    assert module.EXPECTED_CUDA_SOURCES == 102
    assert module.EXPECTED_FC2_CUBINS == 101
    assert module.EXPECTED_SOURCE_FILES == 204
    assert module.EXPECTED_EPILOGUES == 704
    assert module.EXPECTED_B552_HEADERS == 18
    assert module.B552_GENERATED_HEADERS_MANIFEST_SHA256 == (
        "82d97df674784b3632463fbfc5935a80b74bc9a50d2f8aaebb5b7c81ae2c99b6"
    )
    assert module.CUDA_TOOLCHAIN_IDENTITY == {
        "build": "cuda_13.2.r13.2/compiler.37668154_0",
        "release": "13.2",
        "version": "13.2.78",
    }
    assert module.CUBIN_COMPILE_CONTRACT == {
        "architecture": "sm_100f",
        "cxx_standard": "c++17",
        "fast_math": True,
        "ndebug": True,
        "optimization": "O3",
        "output": "cubin",
        "random_seed": "sha256(source_basename)[:16]",
        "work_root": "/tmp/flashinfer-situ-b552-64ed071e-sm100f",
    }
    assert module.ARTIFACT_RELATIVE_ROOT.as_posix() == (
        "fireworks/situ_b552/batched_gemm-b5521162-64ed071e"
    )
    assert module.KERNEL_PARAMS_ABI == {
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
    assert module.ACTIVATION_ABI["runner_enum_value"] == 3
    assert module.ACTIVATION_ABI["generated_enum_value"] == 0
    assert module.ACTIVATION_ABI["private_semantics"] == "situ"
    assert module.NATIVE_RUNNER_ABI == "b552_situ_precomputed_ids_v1"
    assert module.NATIVE_FFI_ABI == {
        "logits_moe": "trtllm_fp4_block_scale_situ_logits_moe",
        "pre_routed_moe": "trtllm_fp4_block_scale_situ_routed_moe",
        "valid_configs": "trtllm_get_valid_situ_moe_configs",
        "manifest_digest": "trtllm_situ_b552_manifest_digest",
        "native_abi_digest": "trtllm_situ_b552_native_abi_digest",
    }
    assert module.NATIVE_SOURCE_MANIFEST_SHA256 == (
        "2f8d9bdf2439dc04bc39c5a51c6a375398ee8e4e5d3c53e44e08ce7058c041ac"
    )
    assert module.NATIVE_MODULE_ABI_SHA256 == (
        "4136c08c87d437e4c94367088ebd2c4ea485190eae0bfd854a0142893a2ff4a2"
    )
    assert module.MODULE_NAME == (
        "fused_moe_trtllm_sm100_situ_b552_64ed071e_4136c08c87d4"
    )
    assert module.CUTLASS_COMMIT == "da5e086dab31d63815acafdac9a9c5893b1c69e2"
    assert module.SOURCE_LICENSE_SHA256 == (
        "cb67c224f503e0a063908950b12f89a7280c6e527dcffac972aa114e4bf3c5de"
    )
    assert module.SOURCE_NOTICE_SHA256 == (
        "90bb9e1dec06f26a34f8f8ce98c53ded10b4fd8e1a9e91e2360e6be354e81db3"
    )
    assert module.ROUTING_SOURCE_COMMIT == ("57ba7eeb7ea3003a2d6ad5d9a057c4f952709bac")
    assert module.ROUTING_SOURCE_MANIFEST_SHA256 == (
        "8a6aadfccc8cd04a563cdb266c40844e0997f9fdc6de14587a22247b6298790d"
    )
    assert module.SEALED_MANIFEST_SHA256 == (
        "504ca05b32d75242df92cd2beffb559837d994f63825903bfea4472751c80350"
    )
    assert set(module.B552_HEADER_SHA256) == {
        "include/trtllmGen_bmm_export/BatchedGemmInterface.h",
        "include/trtllmGen_bmm_export/GemmGatedActOptions.h",
        "include/trtllmGen_bmm_export/KernelParams.h",
        "include/trtllmGen_bmm_export/KernelParamsDecl.h",
    }


def test_vendored_pr_sources_are_exact_and_complete():
    module = _load_script_module()
    files = module._source_files(module.VENDORED_SOURCE_ROOT)

    assert len(files) == 204
    assert len([path for path in files if path.suffix == ".cu"]) == 102
    assert len([path for path in files if path.suffix == ".h"]) == 102
    assert (module.VENDORED_SOURCE_ROOT / "LICENSE").is_file()
    assert (module.VENDORED_SOURCE_ROOT / "NOTICE").is_file()
    assert (
        module._sha256((module.VENDORED_SOURCE_ROOT / "LICENSE").read_bytes())
        == module.SOURCE_LICENSE_SHA256
    )
    assert (
        module._sha256((module.VENDORED_SOURCE_ROOT / "NOTICE").read_bytes())
        == module.SOURCE_NOTICE_SHA256
    )


def test_cuda_toolchain_identity_is_exact(tmp_path, monkeypatch):
    module = _load_script_module()
    state = {"version": "13.2.78"}

    def fake_version(command, **kwargs):
        assert kwargs == {"check": True, "capture_output": True, "text": True}
        assert command[1] == "--version"
        return SimpleNamespace(
            stdout=(
                "Cuda compilation tools, release 13.2, "
                f"V{state['version']}\n"
                "Build cuda_13.2.r13.2/compiler.37668154_0\n"
            )
        )

    monkeypatch.setattr(module.subprocess, "run", fake_version)
    module._require_cuda_toolchain_identity(tmp_path / "nvcc", tmp_path / "cuobjdump")

    state["version"] = "13.2.79"
    with pytest.raises(RuntimeError, match="CUDA toolchain mismatch"):
        module._require_cuda_toolchain_identity(
            tmp_path / "nvcc", tmp_path / "cuobjdump"
        )


def test_vendored_b552_headers_match_pinned_manifest():
    module = _load_script_module()
    paths = sorted(
        path for path in module.VENDORED_B552_ROOT.rglob("*") if path.is_file()
    )
    manifest = {
        path.relative_to(module.VENDORED_B552_ROOT).as_posix(): module._sha256(
            path.read_bytes()
        )
        for path in paths
    }

    assert module._vendored_b552_headers(manifest) == paths


def test_cubin_validation_binds_sm100_elf_symbols_and_metadata(tmp_path, monkeypatch):
    module = _load_script_module()
    cubins = [
        tmp_path / "Bmm_case000_situ_dynB_sm100f.cubin",
        tmp_path / "Bmm_Bfloat16_MxE2m1MxE4m3_Fp32_bA32_bB32_case000_dynB_sm100f.cubin",
    ]
    for cubin in cubins:
        cubin.write_bytes(b"synthetic cubin")
    symbols = [module._cubin_kernel_symbol(cubin) for cubin in cubins]
    metainfo = "\n".join(
        f'{{nullptr, 0, 1, "{symbol}", 0, "{"0" * 64}"}},' for symbol in symbols
    )

    state = {"arch": "100", "omit_symbol": None}

    def fake_cuobjdump(command, **kwargs):
        assert kwargs == {"check": True, "capture_output": True, "text": True}
        assert command[1] == "--dump-elf"
        cubin = Path(command[2])
        assert cubin in cubins
        symbol = module._cubin_kernel_symbol(cubin)
        exported = [
            entry
            for entry in (symbol, f"{symbol}GetSmemSize")
            if entry != state["omit_symbol"]
        ]
        stdout = f"64-bit ELF: type=ET_EXEC, ABI=8, sm={state['arch']}\n"
        stdout += "".join(
            f" 0x1 0 0x100 0x12 0x10 0x12 {entry}\n" for entry in exported
        )
        return SimpleNamespace(stdout=stdout)

    monkeypatch.setattr(module.subprocess, "run", fake_cuobjdump)
    module._validate_cubin_binaries(cubins, metainfo, tmp_path / "cuobjdump")

    state["arch"] = "103"
    with pytest.raises(RuntimeError, match="exactly one SM100 ELF image"):
        module._validate_cubin_binaries(cubins, metainfo, tmp_path / "cuobjdump")
    state["arch"] = "100"

    state["omit_symbol"] = f"{symbols[0]}GetSmemSize"
    with pytest.raises(RuntimeError, match="cubin ELF entry mismatch"):
        module._validate_cubin_binaries(cubins, metainfo, tmp_path / "cuobjdump")
    state["omit_symbol"] = None

    with pytest.raises(RuntimeError, match="metadata must resolve"):
        module._validate_cubin_binaries(
            cubins,
            metainfo.replace(f'"{symbols[0]}"', '"missing"'),
            tmp_path / "cuobjdump",
        )
