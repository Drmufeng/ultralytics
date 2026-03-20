import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    cmd = [sys.executable, "-m", "ui.qt5.app_stub"]
    print("[INFO] 启动Qt5界面:", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(root))


if __name__ == "__main__":
    raise SystemExit(main())
