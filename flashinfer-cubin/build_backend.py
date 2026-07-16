"""
Custom build backend that downloads cubins before building the package.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

from setuptools import build_meta as _orig

# Add parent directory to path to import artifacts module
sys.path.insert(0, str(Path(__file__).parent.parent))

from build_utils import get_git_version

# Skip version check when building flashinfer-cubin package
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"

_SITU_OVERLAY_ROOT = Path("fireworks/situ_b552")


def _situ_b552_enabled() -> bool:
    return os.environ.get("FLASHINFER_BUILD_SITU_B552", "").lower() in (
        "1",
        "true",
    )


def _stock_relative_path(raw: str, context: str) -> PurePosixPath:
    """Return one canonical, wheel-relative artifact path."""
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise RuntimeError(f"Unsafe {context} path: {raw!r}")
    relative = PurePosixPath(raw)
    if (
        relative.is_absolute()
        or relative.as_posix() != raw
        or any(part in ("", ".", "..") for part in relative.parts)
    ):
        raise RuntimeError(f"Unsafe {context} path: {raw!r}")
    return relative


def _stock_manifest_digests(artifacts_module) -> dict[str, str]:
    """Select the pinned upstream manifests for the current CPU architecture."""
    cpu_arch = artifacts_module._get_host_cpu_arch()
    if cpu_arch not in ("x86_64", "aarch64"):
        raise RuntimeError(f"Unsupported cubin-wheel CPU architecture: {cpu_arch}")

    dsl_root = _stock_relative_path(
        artifacts_module.ArtifactPath.DSL_FMHA.rstrip("/"),
        "DSL FMHA root",
    )
    manifests: dict[str, str] = {}
    for raw_path, digest in artifacts_module.CheckSumHash.map_checksums.items():
        relative = _stock_relative_path(raw_path, "pinned stock manifest")
        if relative.name != "checksums.txt":
            raise RuntimeError(
                f"Pinned stock manifest is not checksums.txt: {relative}"
            )
        if relative == dsl_root or dsl_root in relative.parents:
            dsl_relative = relative.relative_to(dsl_root)
            if not dsl_relative.parts or dsl_relative.parts[0] != cpu_arch:
                continue
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise RuntimeError(
                f"Invalid pinned digest for stock manifest {relative}: {digest!r}"
            )
        normalized = relative.as_posix()
        if normalized in manifests:
            raise RuntimeError(f"Duplicate pinned stock manifest: {normalized}")
        manifests[normalized] = digest

    if not manifests:
        raise RuntimeError(f"No authenticated stock manifests selected for {cpu_arch}")
    return manifests


def _authenticated_stock_cubin_closure(
    cubin_dir: Path, manifest_digests: dict[str, str]
) -> dict[str, str]:
    """Expand pinned manifests into the exact stock cubin-wheel file closure.

    The upstream cubin wheel contains each authenticated checksums.txt plus all
    safe .cubin and .h members named by those manifests. Safe .so members belong
    to the separate JIT-cache artifact and are intentionally excluded. Any
    other manifest member type fails closed.
    """
    closure: dict[str, str] = {}
    manifests: list[tuple[PurePosixPath, str]] = []
    for raw_path, expected_digest in sorted(manifest_digests.items()):
        relative = _stock_relative_path(raw_path, "stock manifest")
        if relative.name != "checksums.txt":
            raise RuntimeError(f"Stock manifest is not checksums.txt: {relative}")
        if (
            not isinstance(expected_digest, str)
            or len(expected_digest) != 64
            or any(character not in "0123456789abcdef" for character in expected_digest)
        ):
            raise RuntimeError(
                f"Invalid authenticated digest for {relative}: {expected_digest!r}"
            )
        normalized = relative.as_posix()
        if normalized in closure:
            raise RuntimeError(f"Duplicate authenticated stock manifest: {normalized}")
        closure[normalized] = expected_digest
        manifests.append((relative, expected_digest))

    for manifest_relative, expected_digest in manifests:
        manifest_path = cubin_dir / manifest_relative
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise RuntimeError(
                f"Missing authenticated stock manifest: {manifest_relative}"
            )
        manifest_bytes = manifest_path.read_bytes()
        actual_digest = hashlib.sha256(manifest_bytes).hexdigest()
        if actual_digest != expected_digest:
            raise RuntimeError(
                "Authenticated stock manifest hash mismatch for "
                f"{manifest_relative}: expected={expected_digest}, "
                f"actual={actual_digest}"
            )
        try:
            manifest_text = manifest_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise RuntimeError(
                f"Authenticated stock manifest is not UTF-8: {manifest_relative}"
            ) from error

        manifest_members: set[str] = set()
        for line_number, line in enumerate(manifest_text.splitlines(), 1):
            fields = line.split()
            if len(fields) != 2:
                raise RuntimeError(f"Malformed {manifest_relative} line {line_number}")
            member_digest, raw_member = fields
            if len(member_digest) != 64 or any(
                character not in "0123456789abcdef" for character in member_digest
            ):
                raise RuntimeError(
                    f"Invalid digest in {manifest_relative} line {line_number}"
                )
            member_relative = _stock_relative_path(
                raw_member,
                f"{manifest_relative} member",
            )
            member = manifest_relative.parent / member_relative
            normalized = member.as_posix()
            if normalized in manifest_members:
                raise RuntimeError(
                    f"Duplicate member in {manifest_relative}: {raw_member}"
                )
            manifest_members.add(normalized)

            if member.suffix == ".so":
                continue
            if member.suffix not in (".cubin", ".h"):
                raise RuntimeError(
                    f"Unsupported member in {manifest_relative}: {raw_member}"
                )
            if normalized in closure:
                raise RuntimeError(
                    f"Stock manifests authenticate the same path twice: {normalized}"
                )
            closure[normalized] = member_digest
    return closure


def _remove_download_lockfiles(
    cubin_dir: Path, authenticated_closure: dict[str, str]
) -> None:
    """Remove only FileLock byproducts corresponding to authenticated files."""
    for relative in authenticated_closure:
        artifact = cubin_dir / PurePosixPath(relative)
        lock = artifact.with_name(artifact.name + ".lock")
        if lock.is_symlink() or (lock.exists() and not lock.is_file()):
            raise RuntimeError(f"Unsafe downloader lockfile: {lock}")
        if lock.exists():
            lock.unlink()


def _snapshot_stock_cubin_tree(cubin_dir: Path) -> dict[str, str]:
    """Hash every stock file while excluding only the private SiTU subtree."""
    snapshot = {}
    for path in sorted(cubin_dir.rglob("*")):
        relative = path.relative_to(cubin_dir)
        if relative == _SITU_OVERLAY_ROOT or _SITU_OVERLAY_ROOT in relative.parents:
            continue
        if path.is_symlink():
            raise RuntimeError(f"Stock cubin tree contains a symlink: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise RuntimeError(
                f"Stock cubin tree contains a non-regular file: {relative}"
            )
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        snapshot[relative.as_posix()] = digest.hexdigest()
    return snapshot


def _assert_exact_stock_cubin_tree(
    expected: dict[str, str], actual: dict[str, str]
) -> None:
    """Require exact paths and bytes from the authenticated upstream closure."""
    expected_paths = set(expected)
    actual_paths = set(actual)
    missing = sorted(expected_paths - actual_paths)
    extra = sorted(actual_paths - expected_paths)
    stale = sorted(
        path for path in expected_paths & actual_paths if expected[path] != actual[path]
    )
    if not missing and not extra and not stale:
        return
    raise RuntimeError(
        "Stock cubin tree does not match the authenticated upstream closure: "
        f"missing={missing}, extra={extra}, stale={stale}"
    )


def _verify_authenticated_stock_cubin_tree(
    cubin_dir: Path, artifacts_module
) -> tuple[dict[str, str], dict[str, str]]:
    manifest_digests = _stock_manifest_digests(artifacts_module)
    expected = _authenticated_stock_cubin_closure(cubin_dir, manifest_digests)
    _remove_download_lockfiles(cubin_dir, expected)
    actual = _snapshot_stock_cubin_tree(cubin_dir)
    _assert_exact_stock_cubin_tree(expected, actual)
    cubin_count = sum(PurePosixPath(path).suffix == ".cubin" for path in expected)
    header_count = sum(PurePosixPath(path).suffix == ".h" for path in expected)
    print(
        "Authenticated complete upstream stock cubin closure: "
        f"{cubin_count} cubins, {header_count} headers, "
        f"{len(manifest_digests)} manifests"
    )
    return expected, actual


def _assert_stock_cubin_tree_unchanged(
    before: dict[str, str], after: dict[str, str]
) -> None:
    if before == after:
        return
    before_paths = set(before)
    after_paths = set(after)
    added = sorted(after_paths - before_paths)
    removed = sorted(before_paths - after_paths)
    modified = sorted(
        path for path in before_paths & after_paths if before[path] != after[path]
    )
    raise RuntimeError(
        "The private SiTU overlay changed stock d2c artifacts: "
        f"added={added}, removed={removed}, modified={modified}"
    )


def _build_situ_b552_overlay(cubin_dir: Path) -> None:
    """Build the opt-in Fireworks SiTU bundle into the cubin wheel tree."""
    if not _situ_b552_enabled():
        return

    source_root = os.environ.get("FLASHINFER_SITU_B552_SOURCE_ROOT")
    if not source_root:
        raise RuntimeError(
            "FLASHINFER_BUILD_SITU_B552 requires "
            "FLASHINFER_SITU_B552_SOURCE_ROOT to point at PR #2917 commit "
            "64ed071e23bf8d5d2d5af5c91577e5b8e036a1cf"
        )
    cutlass_root = os.environ.get("FLASHINFER_SITU_B552_CUTLASS_ROOT")
    if not cutlass_root:
        raise RuntimeError(
            "FLASHINFER_BUILD_SITU_B552 requires "
            "FLASHINFER_SITU_B552_CUTLASS_ROOT to point at NVIDIA CUTLASS "
            "commit da5e086dab31d63815acafdac9a9c5893b1c69e2"
        )

    project_root = Path(__file__).parent.parent
    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    nvcc = Path(os.environ.get("FLASHINFER_NVCC", str(cuda_home / "bin/nvcc")))
    output_root = (
        cubin_dir / "fireworks" / "situ_b552" / "batched_gemm-b5521162-64ed071e"
    )
    command = [
        sys.executable,
        str(project_root / "scripts/situ_b552/build.py"),
        "--source-root",
        source_root,
        "--output-root",
        str(output_root),
        "--nvcc",
        str(nvcc),
        "--cutlass-root",
        cutlass_root,
    ]
    subprocess.run(command, check=True, cwd=project_root)

    required = [
        output_root / "checksums.txt",
        output_root / "manifest.sha256",
        output_root / "contract.json",
        output_root / "include/flashinferMetaInfo.h",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"SiTU b552 build did not produce required files: {missing}")


def _install_staged_situ_b552_overlay(staged_cubin_dir: Path, cubin_dir: Path) -> None:
    """Replace only the private overlay with a staged build.

    Building the private closure before the much larger stock download avoids
    starting its FC2 fetches immediately after roughly 16,000 public artifact
    requests. Keep the staged tree outside the cubin directory until the stock
    snapshot is authenticated, then prove installing it did not alter stock.
    """
    source = staged_cubin_dir / _SITU_OVERLAY_ROOT
    destination = cubin_dir / _SITU_OVERLAY_ROOT
    if source.is_symlink() or not source.is_dir():
        raise RuntimeError(f"Missing or unsafe staged SiTU overlay: {source}")
    for path in source.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"Staged SiTU overlay contains a symlink: {path}")

    private_parent = destination.parent
    if private_parent.is_symlink():
        raise RuntimeError(f"Unsafe private overlay parent: {private_parent}")
    private_parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise RuntimeError(f"Unsafe existing SiTU overlay: {destination}")
    if destination.exists():
        if not destination.is_dir():
            raise RuntimeError(
                f"Existing SiTU overlay is not a directory: {destination}"
            )
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def _download_cubins():
    """Download cubins to the source directory before building."""
    from flashinfer import artifacts

    # Create cubins directory in the source tree
    cubin_dir = Path(__file__).parent / "flashinfer_cubin" / "cubins"
    cubin_dir.mkdir(parents=True, exist_ok=True)

    # Set environment variable to download to our package directory
    original_cubin_dir = os.environ.get("FLASHINFER_CUBIN_DIR")
    os.environ["FLASHINFER_CUBIN_DIR"] = str(cubin_dir)

    staged_directory = None
    try:
        staged_cubin_dir = None
        if _situ_b552_enabled():
            # Fetch and compile the small hash-pinned b552 closure before the
            # stock downloader issues roughly 16,000 public artifact requests.
            # The ARM wheel build otherwise received a transient HTTP 403 when
            # it started the private FC2 fetches.
            staged_directory = tempfile.TemporaryDirectory(
                prefix="flashinfer-situ-b552-stage-"
            )
            staged_cubin_dir = Path(staged_directory.name)
            _build_situ_b552_overlay(staged_cubin_dir)

        print(f"Downloading cubins to {cubin_dir}...")
        artifacts.download_artifacts()
        print(f"Successfully downloaded cubins to {cubin_dir}")

        # Count the downloaded files
        cubin_files = list(cubin_dir.rglob("*.cubin"))
        print(f"Downloaded {len(cubin_files)} cubin files")

        expected_stock, stock_before = _verify_authenticated_stock_cubin_tree(
            cubin_dir, artifacts
        )
        if staged_cubin_dir is not None:
            # The private bundle is additive.  Bind that invariant to every
            # wheel build so no stock d2c cubin, header, or metadata file can
            # be replaced, removed, or added accidentally.
            _install_staged_situ_b552_overlay(staged_cubin_dir, cubin_dir)
            stock_after = _snapshot_stock_cubin_tree(cubin_dir)
            _assert_stock_cubin_tree_unchanged(stock_before, stock_after)
            _assert_exact_stock_cubin_tree(expected_stock, stock_after)

    finally:
        if staged_directory is not None:
            staged_directory.cleanup()
        # Restore original environment variable
        if original_cubin_dir:
            os.environ["FLASHINFER_CUBIN_DIR"] = original_cubin_dir
        else:
            os.environ.pop("FLASHINFER_CUBIN_DIR", None)


def _create_build_metadata():
    """Create build metadata file with version information."""
    version_file = Path(__file__).parent.parent / "version.txt"
    if version_file.exists():
        with open(version_file, "r") as f:
            version = f.read().strip()
    else:
        version = "0.0.0+unknown"

    # Add dev suffix if specified
    dev_suffix = os.environ.get("FLASHINFER_DEV_RELEASE_SUFFIX", "")
    if dev_suffix:
        version = f"{version}.dev{dev_suffix}"

    # Get git version
    git_version = get_git_version(cwd=Path(__file__).parent.parent)

    # Append local version suffix if available
    local_version = os.environ.get("FLASHINFER_LOCAL_VERSION")
    if local_version:
        # Use + to create a local version identifier that will appear in wheel name
        version = f"{version}+{local_version}"

    # Create build metadata in the source tree
    package_dir = Path(__file__).parent / "flashinfer_cubin"
    build_meta_file = package_dir / "_build_meta.py"

    # Check if we're in a git repository
    git_dir = Path(__file__).parent.parent / ".git"
    in_git_repo = git_dir.exists()

    # If file exists and not in git repo (installing from sdist), keep existing file
    if build_meta_file.exists() and not in_git_repo:
        print("Build metadata file already exists (not in git repo), keeping it")
        return version

    # In git repo (editable) or file doesn't exist, create/update it
    with open(build_meta_file, "w") as f:
        f.write('"""Build metadata for flashinfer-cubin package."""\n')
        f.write(f'__version__ = "{version}"\n')
        f.write(f'__git_version__ = "{git_version}"\n')

    print(f"Created build metadata file with version {version}")
    return version


# Create build metadata as soon as this module is imported
_create_build_metadata()


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    """Build a wheel, downloading cubins first."""
    _download_cubins()
    return _orig.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    """Build an editable install, downloading cubins first."""
    _download_cubins()
    return _orig.build_editable(wheel_directory, config_settings, metadata_directory)


# Pass through all other hooks
get_requires_for_build_wheel = _orig.get_requires_for_build_wheel
get_requires_for_build_editable = _orig.get_requires_for_build_editable
prepare_metadata_for_build_wheel = _orig.prepare_metadata_for_build_wheel
prepare_metadata_for_build_editable = _orig.prepare_metadata_for_build_editable
