from __future__ import annotations

import ast
import hashlib
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest


_BACKEND_PATH = Path(__file__).parents[1] / "flashinfer-cubin/build_backend.py"


def _load_backend_helpers():
    tree = ast.parse(_BACKEND_PATH.read_text())
    names = {
        "_SITU_OVERLAY_ROOT",
        "_assert_exact_stock_cubin_tree",
        "_authenticated_stock_cubin_closure",
        "_remove_download_lockfiles",
        "_snapshot_stock_cubin_tree",
        "_stock_manifest_digests",
        "_stock_relative_path",
        "_verify_authenticated_stock_cubin_tree",
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
        "PurePosixPath": PurePosixPath,
        "hashlib": hashlib,
    }
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), _BACKEND_PATH, "exec"),
        namespace,
    )
    assert names <= namespace.keys()
    return namespace


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_authenticated_tree(tmp_path: Path, helpers):
    cubin_dir = tmp_path / "cubins"
    artifact_root = cubin_dir / "stock/batched_gemm"
    header = artifact_root / "include/metadata.h"
    cubin = artifact_root / "kernel.cubin"
    header.parent.mkdir(parents=True)
    header.write_bytes(b"authenticated-header")
    cubin.write_bytes(b"authenticated-cubin")

    manifest_relative = "stock/batched_gemm/checksums.txt"
    manifest = cubin_dir / manifest_relative
    manifest.write_text(
        f"{_sha256(cubin.read_bytes())}  kernel.cubin\n"
        f"{_sha256(header.read_bytes())}  include/metadata.h\n"
        f"{_sha256(b'jit-cache-binary')}  excluded-jit-cache.so\n"
    )
    manifest_digests = {
        manifest_relative: _sha256(manifest.read_bytes()),
    }
    expected = helpers["_authenticated_stock_cubin_closure"](
        cubin_dir, manifest_digests
    )
    return cubin_dir, manifest_digests, expected, cubin, header


def _assert_exact(helpers, cubin_dir: Path, expected: dict[str, str]):
    actual = helpers["_snapshot_stock_cubin_tree"](cubin_dir)
    helpers["_assert_exact_stock_cubin_tree"](expected, actual)


def test_authenticated_closure_is_manifest_complete_and_excludes_jit_so(
    tmp_path,
):
    helpers = _load_backend_helpers()
    cubin_dir, _, expected, cubin, header = _make_authenticated_tree(tmp_path, helpers)

    assert expected == {
        "stock/batched_gemm/checksums.txt": _sha256(
            (cubin_dir / "stock/batched_gemm/checksums.txt").read_bytes()
        ),
        "stock/batched_gemm/include/metadata.h": _sha256(header.read_bytes()),
        "stock/batched_gemm/kernel.cubin": _sha256(cubin.read_bytes()),
    }
    assert not any(path.endswith(".so") for path in expected)
    _assert_exact(helpers, cubin_dir, expected)


def test_exact_closure_rejects_missing_stock_member(tmp_path):
    helpers = _load_backend_helpers()
    cubin_dir, _, expected, _, header = _make_authenticated_tree(tmp_path, helpers)
    header.unlink()

    with pytest.raises(RuntimeError, match=r"missing=.*metadata\.h"):
        _assert_exact(helpers, cubin_dir, expected)


def test_exact_closure_rejects_extra_stock_member(tmp_path):
    helpers = _load_backend_helpers()
    cubin_dir, _, expected, _, _ = _make_authenticated_tree(tmp_path, helpers)
    extra = cubin_dir / "stock/batched_gemm/stale.cubin"
    extra.write_bytes(b"not authenticated")

    with pytest.raises(RuntimeError, match=r"extra=.*stale\.cubin"):
        _assert_exact(helpers, cubin_dir, expected)


def test_exact_closure_rejects_stale_stock_bytes(tmp_path):
    helpers = _load_backend_helpers()
    cubin_dir, _, expected, cubin, _ = _make_authenticated_tree(tmp_path, helpers)
    cubin.write_bytes(b"stale-cubin")

    with pytest.raises(RuntimeError, match=r"stale=.*kernel\.cubin"):
        _assert_exact(helpers, cubin_dir, expected)


def test_closure_rejects_stale_authenticated_manifest(tmp_path):
    helpers = _load_backend_helpers()
    cubin_dir, manifest_digests, _, _, _ = _make_authenticated_tree(tmp_path, helpers)
    manifest = cubin_dir / next(iter(manifest_digests))
    manifest.write_text(manifest.read_text() + "\n")

    with pytest.raises(RuntimeError, match="manifest hash mismatch"):
        helpers["_authenticated_stock_cubin_closure"](cubin_dir, manifest_digests)


def test_closure_rejects_unsupported_manifest_member(tmp_path):
    helpers = _load_backend_helpers()
    cubin_dir = tmp_path / "cubins"
    manifest = cubin_dir / "stock/checksums.txt"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(f"{_sha256(b'json')}  kernel_map.json\n")

    with pytest.raises(RuntimeError, match="Unsupported member.*kernel_map.json"):
        helpers["_authenticated_stock_cubin_closure"](
            cubin_dir,
            {"stock/checksums.txt": _sha256(manifest.read_bytes())},
        )


def test_closure_checks_path_safety_before_excluding_so(tmp_path):
    helpers = _load_backend_helpers()
    cubin_dir = tmp_path / "cubins"
    manifest = cubin_dir / "stock/checksums.txt"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(f"{_sha256(b'jit')}  ../escape.so\n")

    with pytest.raises(RuntimeError, match="Unsafe .* member path"):
        helpers["_authenticated_stock_cubin_closure"](
            cubin_dir,
            {"stock/checksums.txt": _sha256(manifest.read_bytes())},
        )


def test_only_authenticated_downloader_locks_are_removed(tmp_path):
    helpers = _load_backend_helpers()
    cubin_dir, _, expected, cubin, _ = _make_authenticated_tree(tmp_path, helpers)
    expected_lock = cubin.with_name(cubin.name + ".lock")
    orphan_lock = cubin_dir / "orphan.lock"
    expected_lock.write_bytes(b"")
    orphan_lock.write_bytes(b"")

    helpers["_remove_download_lockfiles"](cubin_dir, expected)

    assert not expected_lock.exists()
    assert orphan_lock.exists()
    with pytest.raises(RuntimeError, match=r"extra=.*orphan\.lock"):
        _assert_exact(helpers, cubin_dir, expected)


def test_pinned_manifest_selection_uses_only_the_host_dsl_architecture():
    helpers = _load_backend_helpers()
    global_digest = "1" * 64
    x86_digest = "2" * 64
    arm_digest = "3" * 64
    artifacts = SimpleNamespace(
        _get_host_cpu_arch=lambda: "x86_64",
        ArtifactPath=SimpleNamespace(DSL_FMHA="dsl/"),
        CheckSumHash=SimpleNamespace(
            map_checksums={
                "stock/checksums.txt": global_digest,
                "dsl/x86_64/sm_100a/checksums.txt": x86_digest,
                "dsl/aarch64/sm_100a/checksums.txt": arm_digest,
            }
        ),
    )

    assert helpers["_stock_manifest_digests"](artifacts) == {
        "stock/checksums.txt": global_digest,
        "dsl/x86_64/sm_100a/checksums.txt": x86_digest,
    }


def test_complete_verifier_removes_expected_locks_and_proves_exact_tree(
    tmp_path,
):
    helpers = _load_backend_helpers()
    cubin_dir, manifest_digests, expected, cubin, _ = _make_authenticated_tree(
        tmp_path, helpers
    )
    cubin_lock = cubin.with_name(cubin.name + ".lock")
    cubin_lock.write_bytes(b"")
    artifacts = SimpleNamespace(
        _get_host_cpu_arch=lambda: "x86_64",
        ArtifactPath=SimpleNamespace(DSL_FMHA="dsl/"),
        CheckSumHash=SimpleNamespace(map_checksums=manifest_digests),
    )

    verified_expected, verified_actual = helpers[
        "_verify_authenticated_stock_cubin_tree"
    ](cubin_dir, artifacts)

    assert verified_expected == expected
    assert verified_actual == expected
    assert not cubin_lock.exists()
