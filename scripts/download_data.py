"""Download the garbage classification dataset from Kaggle and resolve data_dir."""
import os
import zipfile
import subprocess
from pathlib import Path


def setup_kaggle_credentials() -> None:
    kaggle_src = Path("/content/kaggle.json")
    kaggle_dst = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_src.exists() and not kaggle_dst.exists():
        kaggle_dst.parent.mkdir(parents=True, exist_ok=True)
        kaggle_dst.write_bytes(kaggle_src.read_bytes())
        kaggle_dst.chmod(0o600)
        print("[DATA] Kaggle credentials configured.")


def download_and_extract(dest: Path) -> Path:
    """Download dataset and return the resolved data_dir."""
    dest.mkdir(parents=True, exist_ok=True)

    zip_path = dest / "garbage-classification.zip"
    if not zip_path.exists():
        print("[DATA] Downloading dataset from Kaggle…")
        try:
            subprocess.run(
                [
                    "kaggle", "datasets", "download",
                    "-d", "hassnainzaidi/garbage-classification",
                    "-p", str(dest),
                ],
                check=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"[DATA] Kaggle download failed: {exc}\n"
                "Make sure kaggle.json is at /content/kaggle.json (Colab) "
                "or ~/.kaggle/kaggle.json."
            ) from exc

    print("[DATA] Extracting…")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)

    # Resolve actual top-level folder
    candidates = [p for p in dest.iterdir() if p.is_dir()]
    print(f"[DATA] Directories after extraction: {[c.name for c in candidates]}")

    # Look for a folder that contains class subdirectories
    data_dir = None
    for candidate in candidates:
        subdirs = [p for p in candidate.iterdir() if p.is_dir()]
        if subdirs:
            data_dir = candidate
            break

    if data_dir is None:
        data_dir = dest  # files extracted flat

    print(f"[DATA] Resolved data_dir = {data_dir}")
    print(f"[DATA] Classes found: {[p.name for p in sorted(data_dir.iterdir()) if p.is_dir()]}")
    return data_dir


if __name__ == "__main__":
    setup_kaggle_credentials()
    dest = Path("/content/data/raw")
    data_dir = download_and_extract(dest)
    print(f"[DATA] Use data_dir='{data_dir}' in your training scripts.")
