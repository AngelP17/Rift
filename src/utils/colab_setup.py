"""Google Colab environment detection, setup, and artifact persistence.

Usage in a Colab notebook cell:
    from utils.colab_setup import setup_rift
    setup_rift()  # clones repo, installs deps, configures PYTHONPATH
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/AngelP17/Rift.git"
REPO_DIR = "/content/Rift"
DRIVE_MOUNT = "/content/drive"
DRIVE_RIFT_DIR = "/content/drive/MyDrive/rift_artifacts"


def is_colab() -> bool:
    """Detect whether we are running inside Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def is_gpu_available() -> bool:
    """Check for GPU availability (CUDA)."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_device() -> str:
    """Return 'cuda' if GPU available, else 'cpu'."""
    return "cuda" if is_gpu_available() else "cpu"


def clone_repo(branch: str = "main", force: bool = False) -> Path:
    """Clone the Rift repo into Colab's filesystem."""
    repo_path = Path(REPO_DIR)
    if repo_path.exists() and not force:
        print(f"Repo already exists at {repo_path}")
        subprocess.run(["git", "-C", str(repo_path), "pull"], check=False)
        return repo_path

    if repo_path.exists():
        import shutil
        shutil.rmtree(repo_path)

    subprocess.run(
        ["git", "clone", "--depth", "1", "-b", branch, REPO_URL, str(repo_path)],
        check=True,
    )
    print(f"Cloned Rift ({branch}) to {repo_path}")
    return repo_path


def install_deps(extras: str = "") -> None:
    """Install Rift dependencies. Uses GPU torch if available."""
    repo_path = Path(REPO_DIR)
    if not repo_path.exists():
        clone_repo()

    cmds = [
        [sys.executable, "-m", "pip", "install", "-q", "-e", f"{repo_path}[dev]"],
        [sys.executable, "-m", "pip", "install", "-q",
         "polars", "numpy", "pandas", "scikit-learn", "xgboost",
         "duckdb", "shap", "structlog", "python-dotenv", "rich", "jinja2", "pyarrow"],
    ]

    if is_gpu_available():
        print("GPU detected -- installing PyTorch with CUDA support")
    else:
        cmds.append([
            sys.executable, "-m", "pip", "install", "-q",
            "torch", "--index-url", "https://download.pytorch.org/whl/cpu",
        ])

    for cmd in cmds:
        subprocess.run(cmd, check=False)

    print("Dependencies installed")


def configure_pythonpath() -> None:
    """Add src/ to Python path for bare module imports."""
    src_path = str(Path(REPO_DIR) / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    os.environ["PYTHONPATH"] = src_path
    print(f"PYTHONPATH set to {src_path}")


def mount_drive() -> Path | None:
    """Mount Google Drive for artifact persistence (Colab only)."""
    if not is_colab():
        print("Not in Colab -- skipping Drive mount")
        return None

    from google.colab import drive
    drive.mount(DRIVE_MOUNT)
    artifact_dir = Path(DRIVE_RIFT_DIR)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    print(f"Drive mounted. Artifacts dir: {artifact_dir}")
    return artifact_dir


def setup_rift(
    branch: str = "main",
    mount_gdrive: bool = False,
    force_clone: bool = False,
) -> dict:
    """One-call setup for Rift in Google Colab.

    Returns a dict with paths and device info.
    """
    print("=" * 60)
    print("  Rift: Colab Environment Setup")
    print("=" * 60)

    env_info = {
        "is_colab": is_colab(),
        "device": get_device(),
        "gpu_available": is_gpu_available(),
    }
    print(f"Environment: {'Colab' if env_info['is_colab'] else 'Local'}")
    print(f"Device: {env_info['device']}")

    repo_path = clone_repo(branch=branch, force=force_clone)
    env_info["repo_path"] = str(repo_path)

    install_deps()
    configure_pythonpath()

    env_info["src_path"] = str(repo_path / "src")
    env_info["data_dir"] = str(repo_path / "data")
    env_info["artifacts_dir"] = str(repo_path / "artifacts")

    if mount_gdrive:
        drive_dir = mount_drive()
        if drive_dir:
            env_info["drive_artifacts"] = str(drive_dir)

    print("=" * 60)
    print("  Setup complete! You can now import Rift modules.")
    print("  Example: from data.generator import generate_transactions")
    print("=" * 60)

    return env_info
