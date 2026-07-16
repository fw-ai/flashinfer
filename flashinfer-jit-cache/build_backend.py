"""
Copyright (c) 2025 by FlashInfer team.

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

import hashlib
import os
import platform
import shutil
import sys
import zipfile
from pathlib import Path
from setuptools import build_meta as _orig
from wheel.bdist_wheel import bdist_wheel

# Add parent directory to path to import flashinfer modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from build_utils import get_git_version

# Skip version check when building flashinfer-jit-cache package
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"


_SITU_BUILD_FLAG = "FLASHINFER_BUILD_SITU_B552"
_SITU_SOURCE_ENV = "FLASHINFER_SITU_B552_SOURCE_ROOT"
_SITU_CUTLASS_ENV = "FLASHINFER_SITU_B552_CUTLASS_ROOT"


def _situ_b552_build_enabled() -> bool:
    return os.environ.get(_SITU_BUILD_FLAG, "").lower() in ("1", "true")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_regular_elf(path: Path, label: str) -> tuple[int, str]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} must be a regular non-symlink file: {path}")
    size = path.stat().st_size
    if size <= 4:
        raise RuntimeError(f"{label} is empty or truncated: {path}")
    with path.open("rb") as stream:
        if stream.read(4) != b"\x7fELF":
            raise RuntimeError(f"{label} is not an ELF shared object: {path}")
    return size, _sha256_file(path)


def _build_and_stage_situ_b552_aot(
    output_dir: Path, verbose: bool
) -> tuple[str, Path] | None:
    """Build the private module explicitly and add it to the wheel source tree.

    This is intentionally independent from aot.gen_all_modules. The SiTU wheel
    contract must not silently depend on a generic module-selection predicate,
    and an installed AOT cache must not satisfy this source build.
    """
    if not _situ_b552_build_enabled():
        return None

    missing_env = [
        name
        for name in (_SITU_SOURCE_ENV, _SITU_CUTLASS_ENV)
        if not os.environ.get(name)
    ]
    if missing_env:
        raise RuntimeError(
            "SiTU b552 JIT-cache builds require: " + ", ".join(missing_env)
        )

    from flashinfer.jit import build_jit_specs
    from flashinfer.jit.situ_b552 import (
        MODULE_NAME,
        gen_trtllm_gen_fused_moe_situ_b552_module,
    )

    spec = gen_trtllm_gen_fused_moe_situ_b552_module()
    if spec.name != MODULE_NAME:
        raise RuntimeError(
            f"SiTU b552 JIT spec name mismatch: {spec.name!r} != {MODULE_NAME!r}"
        )

    # Always target the source-build path. get_library_path may resolve an
    # already-installed AOT wheel, which would defeat the matched-wheel build.
    build_jit_specs([spec], verbose=verbose, skip_prebuilt=False)
    source = spec.jit_library_path
    source_size, source_digest = _require_regular_elf(
        source, "source-built SiTU b552 module"
    )

    destination = output_dir / MODULE_NAME / f"{MODULE_NAME}.so"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise RuntimeError(
            f"staged SiTU b552 module destination must not be a symlink: {destination}"
        )
    shutil.copy2(source, destination)
    staged_size, staged_digest = _require_regular_elf(
        destination, "staged SiTU b552 module"
    )
    if (staged_size, staged_digest) != (source_size, source_digest):
        raise RuntimeError(
            "Staged SiTU b552 module differs from its source-built shared object"
        )
    return MODULE_NAME, destination


def _verify_situ_b552_wheel(
    wheel_path: Path, module_name: str, staged_module: Path
) -> None:
    staged_size, staged_digest = _require_regular_elf(
        staged_module, "staged SiTU b552 module"
    )
    expected_member = f"flashinfer_jit_cache/jit_cache/{module_name}/{module_name}.so"
    with zipfile.ZipFile(wheel_path) as wheel:
        names = wheel.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError(
                f"JIT-cache wheel has duplicate ZIP members: {wheel_path}"
            )
        if names.count(expected_member) != 1:
            raise RuntimeError(
                f"JIT-cache wheel is missing exact SiTU b552 member {expected_member}"
            )
        info = wheel.getinfo(expected_member)
        if info.file_size != staged_size:
            raise RuntimeError(
                "JIT-cache wheel SiTU b552 module size differs from staged module: "
                f"{info.file_size} != {staged_size}"
            )
        member_digest = hashlib.sha256()
        with wheel.open(info) as stream:
            magic = stream.read(4)
            member_digest.update(magic)
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                member_digest.update(chunk)
    if magic != b"\x7fELF":
        raise RuntimeError(
            "JIT-cache wheel SiTU b552 member is not an ELF shared object"
        )
    if member_digest.hexdigest() != staged_digest:
        raise RuntimeError(
            "JIT-cache wheel SiTU b552 member differs from staged source-built module"
        )


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
    build_meta_file = Path(__file__).parent / "flashinfer_jit_cache" / "_build_meta.py"

    # Check if we're in a git repository
    git_dir = Path(__file__).parent.parent / ".git"
    in_git_repo = git_dir.exists()

    # If file exists and not in git repo (installing from sdist), keep existing file
    if build_meta_file.exists() and not in_git_repo:
        print("Build metadata file already exists (not in git repo), keeping it")
        return version

    # In git repo (editable) or file doesn't exist, create/update it
    with open(build_meta_file, "w") as f:
        f.write('"""Build metadata for flashinfer-jit-cache package."""\n')
        f.write(f'__version__ = "{version}"\n')
        f.write(f'__git_version__ = "{git_version}"\n')

    print(f"Created build metadata file with version {version}")
    return version


# Create build metadata as soon as this module is imported
_create_build_metadata()


def _compile_jit_cache(output_dir: Path, verbose: bool = True):
    """Compile AOT modules using flashinfer.aot functions directly."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent

    # Ensure 3rdparty submodules are populated (may be empty in CI Docker images).
    # Skip if submodules are already present or if git metadata is incomplete
    # (e.g., Docker builds where .git points to a parent repo not in the context).
    import subprocess

    submodule_check_paths = [
        project_root / "3rdparty" / "cutlass" / "include",
        project_root / "3rdparty" / "spdlog" / "include",
        project_root / "3rdparty" / "cccl" / "cub",
    ]
    if not all(p.exists() for p in submodule_check_paths):
        result = subprocess.run(
            ["git", "submodule", "update", "--init", "--recursive"],
            cwd=str(project_root),
            capture_output=True,
        )
        if result.returncode != 0:
            missing = [str(p) for p in submodule_check_paths if not p.exists()]
            if missing:
                raise RuntimeError(
                    f"git submodule update failed and submodules are missing: {missing}\n"
                    f"git stderr: {result.stderr.decode().strip()}"
                )

    # Ensure flashinfer/data/ symlinks exist (normally created by the main
    # package's build_backend, but jit-cache builds may not install the main
    # package first). Use importlib to avoid name collision with this file.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "main_build_backend", project_root / "build_backend.py"
    )
    main_build_backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main_build_backend)
    main_build_backend._create_data_dir(use_symlinks=True)

    from flashinfer import aot

    # Set up build directory
    build_dir = project_root / "build" / "aot"

    # Use the centralized compilation function from aot.py
    aot.compile_and_package_modules(
        out_dir=output_dir,
        build_dir=build_dir,
        project_root=project_root,
        config=None,  # Use default config
        verbose=verbose,
        skip_prebuilt=False,
    )
    _build_and_stage_situ_b552_aot(output_dir, verbose)


def _build_aot_modules():
    # First, ensure AOT modules are compiled
    aot_package_dir = Path(__file__).parent / "flashinfer_jit_cache" / "jit_cache"
    aot_package_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Compile AOT modules
        _compile_jit_cache(aot_package_dir)

        # Verify that some modules were actually compiled
        so_files = list(aot_package_dir.rglob("*.so"))
        if not so_files:
            raise RuntimeError("No .so files were generated during AOT compilation")

        print(f"Successfully compiled {len(so_files)} AOT modules")

    except Exception as e:
        print(f"Failed to compile AOT modules: {e}")
        raise


def _prepare_build():
    """Shared preparation logic for both wheel and editable builds."""
    _build_aot_modules()


class PlatformSpecificBdistWheel(bdist_wheel):
    """Custom wheel builder that uses py_limited_api for cp39+."""

    def finalize_options(self):
        super().finalize_options()
        # Force platform-specific wheel (not pure Python)
        self.root_is_pure = False
        # Use py_limited_api for cp39 (Python 3.9+)
        self.py_limited_api = "cp39"

    def get_tag(self):
        # Use py_limited_api tags
        python_tag = "cp39"
        abi_tag = "abi3"  # Stable ABI tag

        # Get platform tag
        machine = platform.machine()
        if platform.system() == "Linux":
            # Use manylinux_2_28 as specified
            if machine == "x86_64":
                plat_tag = "manylinux_2_28_x86_64"
            elif machine == "aarch64":
                plat_tag = "manylinux_2_28_aarch64"
            else:
                plat_tag = f"linux_{machine}"
        else:
            # For non-Linux platforms, use the default
            import distutils.util

            plat_tag = distutils.util.get_platform().replace("-", "_").replace(".", "_")

        return python_tag, abi_tag, plat_tag


class _MonkeyPatchBdistWheel:
    """Context manager to temporarily replace bdist_wheel with our custom class."""

    def __enter__(self):
        from setuptools.command import bdist_wheel as setuptools_bdist_wheel

        self.original_bdist_wheel = setuptools_bdist_wheel.bdist_wheel
        setuptools_bdist_wheel.bdist_wheel = PlatformSpecificBdistWheel

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        from setuptools.command import bdist_wheel as setuptools_bdist_wheel

        setuptools_bdist_wheel.bdist_wheel = self.original_bdist_wheel


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    """Build wheel with custom AOT module compilation."""
    print("Building flashinfer-jit-cache wheel...")

    _prepare_build()

    with _MonkeyPatchBdistWheel():
        wheel_name = _orig.build_wheel(
            wheel_directory, config_settings, metadata_directory
        )

    if _situ_b552_build_enabled():
        from flashinfer.jit.situ_b552 import MODULE_NAME

        staged_module = (
            Path(__file__).parent
            / "flashinfer_jit_cache"
            / "jit_cache"
            / MODULE_NAME
            / f"{MODULE_NAME}.so"
        )
        _verify_situ_b552_wheel(
            Path(wheel_directory) / wheel_name, MODULE_NAME, staged_module
        )
    return wheel_name


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    """Build editable install with custom AOT module compilation."""
    print("Building flashinfer-jit-cache in editable mode...")

    _prepare_build()

    # Now build the editable install using setuptools
    _orig_build_editable = getattr(_orig, "build_editable", None)
    if _orig_build_editable is None:
        raise RuntimeError("build_editable not supported by setuptools backend")

    result = _orig_build_editable(wheel_directory, config_settings, metadata_directory)

    return result


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    """Prepare metadata with platform-specific wheel tags."""
    with _MonkeyPatchBdistWheel():
        return _orig.prepare_metadata_for_build_wheel(
            metadata_directory, config_settings
        )


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    """Prepare metadata for editable install."""
    with _MonkeyPatchBdistWheel():
        return _orig.prepare_metadata_for_build_editable(
            metadata_directory, config_settings
        )


# Export the required interface
get_requires_for_build_wheel = _orig.get_requires_for_build_wheel
get_requires_for_build_editable = getattr(
    _orig, "get_requires_for_build_editable", None
)
