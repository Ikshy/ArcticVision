import os
import sys
import shutil
from pathlib import Path

# ── Minimum Python version ────────────────────────────────────────────────────
MIN_PYTHON = (3, 9)
if sys.version_info < MIN_PYTHON:
    sys.exit(f"[ERROR] Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required. "
             f"Found: {sys.version}")

ROOT = Path(__file__).parent


def check_env_file() -> None:
    """Ensure .env exists; create from .env.example if missing."""
    env_path = ROOT / ".env"
    example_path = ROOT / ".env.example"
    if not env_path.exists():
        if example_path.exists():
            shutil.copy(example_path, env_path)
            print("[SETUP]  Created .env from .env.example — "
                  "please fill in your API credentials.")
        else:
            print("[WARN ]  No .env or .env.example found.")
    else:
        print("[OK   ]  .env file exists.")


def create_directories() -> None:
    """Guarantee all expected project directories exist."""
    dirs = [
        "data/raw", "data/processed", "data/external",
        "outputs/plots", "outputs/animations",
        "outputs/dashboards", "outputs/models",
        "reports", "notebooks",
    ]
    for d in dirs:
        path = ROOT / d
        path.mkdir(parents=True, exist_ok=True)
    print("[OK   ]  All project directories verified.")


def check_dependencies() -> None:
    """Spot-check key packages are importable."""
    critical = ["numpy", "pandas", "torch", "xarray",
                 "rasterio", "plotly", "sklearn", "yaml"]
    missing = []
    for pkg in critical:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[WARN ]  Missing packages: {missing}")
        print("         Run:  pip install -r requirements.txt")
    else:
        print("[OK   ]  All core dependencies found.")


def validate_config() -> None:
    """Load configs/config.yaml and report key settings."""
    try:
        import yaml
        with open(ROOT / "configs" / "config.yaml") as f:
            cfg = yaml.safe_load(f)
        print(f"[OK   ]  Config loaded — project: {cfg['project']['name']} "
              f"v{cfg['project']['version']}")
    except Exception as e:
        print(f"[WARN ]  Could not load config.yaml: {e}")


def main() -> None:
    print("\n" + "=" * 60)
    print("  ArcticVision — Project Setup Verification")
    print("=" * 60)
    print(f"  Python  : {sys.version.split()[0]}")
    print(f"  Root    : {ROOT}\n")

    check_env_file()
    create_directories()
    check_dependencies()
    validate_config()

    print("\n" + "=" * 60)
    print("  Setup complete.  Next steps:")
    print("  1. Edit .env with your NASA / GEE credentials")
    print("  2. Run:  python run_system.py --help")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()