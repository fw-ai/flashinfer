from __future__ import annotations

import ast
import hashlib
import os
import shutil
import sys
import types
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest


_BACKEND_PATH = Path(__file__).parents[1] / "flashinfer-jit-cache" / "build_backend.py"


def _load_helpers():
    tree = ast.parse(_BACKEND_PATH.read_text())
    names = {
        "_SITU_BUILD_FLAG",
        "_SITU_SOURCE_ENV",
        "_SITU_CUTLASS_ENV",
        "_situ_b552_build_enabled",
        "_sha256_file",
        "_require_regular_elf",
        "_build_and_stage_situ_b552_aot",
        "_verify_situ_b552_wheel",
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
    namespace = {
        "Path": Path,
        "hashlib": hashlib,
        "os": os,
        "shutil": shutil,
        "zipfile": zipfile,
    }
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), _BACKEND_PATH, "exec"),
        namespace,
    )
    assert names <= namespace.keys()
    return namespace


def _install_fake_situ_modules(monkeypatch, module_name, generator, builder):
    flashinfer = types.ModuleType("flashinfer")
    flashinfer.__path__ = []
    jit = types.ModuleType("flashinfer.jit")
    jit.__path__ = []
    jit.build_jit_specs = builder
    situ = types.ModuleType("flashinfer.jit.situ_b552")
    situ.MODULE_NAME = module_name
    situ.gen_trtllm_gen_fused_moe_situ_b552_module = generator
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer)
    monkeypatch.setitem(sys.modules, "flashinfer.jit", jit)
    monkeypatch.setitem(sys.modules, "flashinfer.jit.situ_b552", situ)


def test_private_module_is_explicitly_source_built_and_staged(tmp_path, monkeypatch):
    helpers = _load_helpers()
    module_name = "fused_moe_trtllm_sm100_situ_b552_test"
    source = tmp_path / "build" / f"{module_name}.so"
    calls = []
    spec = SimpleNamespace(name=module_name, jit_library_path=source)

    def build(specs, verbose, skip_prebuilt):
        calls.append((specs, verbose, skip_prebuilt))
        source.parent.mkdir(parents=True)
        source.write_bytes(b"\x7fELFsource-built-situ")

    _install_fake_situ_modules(monkeypatch, module_name, lambda: spec, build)
    monkeypatch.setenv("FLASHINFER_BUILD_SITU_B552", "1")
    monkeypatch.setenv("FLASHINFER_SITU_B552_SOURCE_ROOT", str(tmp_path / "source"))
    monkeypatch.setenv("FLASHINFER_SITU_B552_CUTLASS_ROOT", str(tmp_path / "cutlass"))

    result = helpers["_build_and_stage_situ_b552_aot"](tmp_path / "wheel", True)

    destination = tmp_path / "wheel" / module_name / f"{module_name}.so"
    assert result == (module_name, destination)
    assert destination.read_bytes() == source.read_bytes()
    assert calls == [([spec], True, False)]


def test_private_module_build_is_disabled_without_opt_in(tmp_path, monkeypatch):
    helpers = _load_helpers()
    monkeypatch.delenv("FLASHINFER_BUILD_SITU_B552", raising=False)

    assert helpers["_build_and_stage_situ_b552_aot"](tmp_path, True) is None


def test_private_module_build_requires_source_roots(tmp_path, monkeypatch):
    helpers = _load_helpers()
    monkeypatch.setenv("FLASHINFER_BUILD_SITU_B552", "true")
    monkeypatch.delenv("FLASHINFER_SITU_B552_SOURCE_ROOT", raising=False)
    monkeypatch.delenv("FLASHINFER_SITU_B552_CUTLASS_ROOT", raising=False)

    with pytest.raises(RuntimeError, match="SOURCE_ROOT.*CUTLASS_ROOT"):
        helpers["_build_and_stage_situ_b552_aot"](tmp_path, False)


def test_private_module_build_rejects_wrong_spec_name(tmp_path, monkeypatch):
    helpers = _load_helpers()
    module_name = "expected_situ_module"
    spec = SimpleNamespace(name="wrong_module", jit_library_path=tmp_path / "wrong.so")
    _install_fake_situ_modules(
        monkeypatch, module_name, lambda: spec, lambda *args, **kwargs: None
    )
    monkeypatch.setenv("FLASHINFER_BUILD_SITU_B552", "1")
    monkeypatch.setenv("FLASHINFER_SITU_B552_SOURCE_ROOT", "source")
    monkeypatch.setenv("FLASHINFER_SITU_B552_CUTLASS_ROOT", "cutlass")

    with pytest.raises(RuntimeError, match="JIT spec name mismatch"):
        helpers["_build_and_stage_situ_b552_aot"](tmp_path, False)


@pytest.mark.parametrize("contents", [b"", b"not-elf"])
def test_private_module_build_rejects_invalid_output(tmp_path, monkeypatch, contents):
    helpers = _load_helpers()
    module_name = "situ_module"
    source = tmp_path / "source.so"
    spec = SimpleNamespace(name=module_name, jit_library_path=source)

    def build(*args, **kwargs):
        source.write_bytes(contents)

    _install_fake_situ_modules(monkeypatch, module_name, lambda: spec, build)
    monkeypatch.setenv("FLASHINFER_BUILD_SITU_B552", "1")
    monkeypatch.setenv("FLASHINFER_SITU_B552_SOURCE_ROOT", "source")
    monkeypatch.setenv("FLASHINFER_SITU_B552_CUTLASS_ROOT", "cutlass")

    with pytest.raises(RuntimeError, match="empty or truncated|not an ELF"):
        helpers["_build_and_stage_situ_b552_aot"](tmp_path / "wheel", False)


def _write_wheel(path: Path, entries: list[tuple[str, bytes]]) -> None:
    with zipfile.ZipFile(path, "w") as wheel:
        for name, contents in entries:
            wheel.writestr(name, contents)


def test_wheel_verifier_accepts_exact_staged_elf(tmp_path):
    helpers = _load_helpers()
    module_name = "situ_module"
    contents = b"\x7fELFmatched-module"
    staged = tmp_path / "staged.so"
    staged.write_bytes(contents)
    member = f"flashinfer_jit_cache/jit_cache/{module_name}/{module_name}.so"
    wheel = tmp_path / "jit.whl"
    _write_wheel(wheel, [(member, contents)])

    helpers["_verify_situ_b552_wheel"](wheel, module_name, staged)


@pytest.mark.parametrize(
    "wheel_contents,match",
    [
        (None, "missing exact"),
        (b"\x7fELFmismatched", "size differs"),
        (b"\x7fELFmatched-modulf", "differs from staged"),
        (b"\x00ELFmatched-module", "not an ELF"),
    ],
)
def test_wheel_verifier_rejects_missing_or_mismatched_member(
    tmp_path, wheel_contents, match
):
    helpers = _load_helpers()
    module_name = "situ_module"
    contents = b"\x7fELFmatched-module"
    staged = tmp_path / "staged.so"
    staged.write_bytes(contents)
    member = f"flashinfer_jit_cache/jit_cache/{module_name}/{module_name}.so"
    wheel = tmp_path / "jit.whl"
    entries = [] if wheel_contents is None else [(member, wheel_contents)]
    _write_wheel(wheel, entries)

    with pytest.raises(RuntimeError, match=match):
        helpers["_verify_situ_b552_wheel"](wheel, module_name, staged)


def test_wheel_verifier_rejects_duplicate_member(tmp_path):
    helpers = _load_helpers()
    module_name = "situ_module"
    contents = b"\x7fELFmatched-module"
    staged = tmp_path / "staged.so"
    staged.write_bytes(contents)
    member = f"flashinfer_jit_cache/jit_cache/{module_name}/{module_name}.so"
    wheel = tmp_path / "jit.whl"
    with pytest.warns(UserWarning, match="Duplicate name"):
        _write_wheel(wheel, [(member, contents), (member, contents)])

    with pytest.raises(RuntimeError, match="duplicate ZIP members"):
        helpers["_verify_situ_b552_wheel"](wheel, module_name, staged)


def test_backend_orders_explicit_private_stage_after_generic_copy():
    source = _BACKEND_PATH.read_text()
    function = source.split("def _compile_jit_cache", 1)[1].split(
        "def _build_aot_modules", 1
    )[0]

    assert function.index("aot.compile_and_package_modules(") < function.index(
        "_build_and_stage_situ_b552_aot(output_dir, verbose)"
    )
    explicit = source.split("def _build_and_stage_situ_b552_aot", 1)[1].split(
        "def _verify_situ_b552_wheel", 1
    )[0]
    assert "skip_prebuilt=False" in explicit
    assert "source = spec.jit_library_path" in explicit
    assert "spec.get_library_path(" not in explicit

    verifier = source.split("def _verify_situ_b552_wheel", 1)[1].split(
        "def _create_build_metadata", 1
    )[0]
    assert "wheel.open(info)" in verifier
    assert "wheel.read(info)" not in verifier


def test_package_data_uses_exact_one_level_module_layout():
    pyproject = (_BACKEND_PATH.parent / "pyproject.toml").read_text()

    assert 'flashinfer_jit_cache = ["jit_cache/*/*.so"]' in pyproject
