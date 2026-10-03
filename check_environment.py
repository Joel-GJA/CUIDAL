"""
Boot-Cached Requirement Checker for PILSS Custom Dataset Labeller.
Verifies python packages once per Windows boot session.
If the PC restarts, GetTickCount64() resets, automatically triggering a fresh verification.
"""

import os
import sys
import time
import tempfile
import ctypes
import subprocess

REQUIRED_PACKAGES = [
    ("opencv-python", "cv2"),
    ("numpy", "numpy"),
    ("Pillow", "PIL"),
    ("ultralytics", "ultralytics"),
]


def get_system_boot_timestamp() -> float:
    """Returns the approximate system boot timestamp using Windows kernel tick count."""
    try:
        uptime_ms = ctypes.windll.kernel32.GetTickCount64()
        return time.time() - (uptime_ms / 1000.0)
    except Exception:
        return 0.0


def is_boot_session_cached(cache_file: str, current_boot_ts: float) -> bool:
    """Checks if requirements were already verified in the current system boot session."""
    if not os.path.exists(cache_file) or current_boot_ts <= 0:
        return False
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            cached_boot_ts = float(f.read().strip())
            return abs(cached_boot_ts - current_boot_ts) < 120.0
    except Exception:
        return False


def save_boot_cache(cache_file: str, current_boot_ts: float):
    """Saves the current boot timestamp to the temporary cache file."""
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            f.write(str(current_boot_ts))
    except Exception:
        pass


def verify_and_install_dependencies():
    cache_file = os.path.join(tempfile.gettempdir(), "pilss_labeller_boot.tag")
    current_boot_ts = get_system_boot_timestamp()

    # Fast path: Check if already verified during this boot session
    if is_boot_session_cached(cache_file, current_boot_ts):
        print("[FAST START] Dependencies verified for current Windows session (cached).")
        return True

    print("[INITIAL CHECK] Verifying package dependencies for current boot session...")
    missing = []
    for pkg_name, import_name in REQUIRED_PACKAGES:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg_name)

    if missing:
        print(f"\n[MISSING PACKAGES] Found {len(missing)} uninstalled packages: {', '.join(missing)}")
        req_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
        if os.path.exists(req_file):
            print(f"[INSTALLING] Installing dependencies via pip...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
                print("[SUCCESS] All packages installed successfully.")
            except Exception as e:
                print(f"[ERROR] Failed to install requirements: {e}")
                return False
        else:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
                print("[SUCCESS] Missing packages installed.")
            except Exception as e:
                print(f"[ERROR] Failed to install missing packages: {e}")
                return False
    else:
        print("[OK] All required packages are already installed.")

    # Save to boot cache
    save_boot_cache(cache_file, current_boot_ts)
    print("[CACHED] Requirement check cached until next system reboot.\n")
    return True


if __name__ == "__main__":
    success = verify_and_install_dependencies()
    sys.exit(0 if success else 1)
