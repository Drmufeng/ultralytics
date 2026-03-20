import argparse
import importlib
import subprocess
import sys
from pathlib import Path
from shutil import which


IMPORT_TO_PACKAGE = {
    "ultralytics": "ultralytics",
    "torch": "torch",
    "torchvision": "torchvision",
    "torchaudio": "torchaudio",
    "numpy": "numpy",
    "cv2": "opencv-python",
    "yaml": "PyYAML",
    "tqdm": "tqdm",
    "matplotlib": "matplotlib",
    "lap": "lap",
    "einops": "einops",
    "h5py": "h5py",
    "skvideo": "scikit-video",
    "PyQt5": "PyQt5",
}


def parse_requirements(req_path: Path):
    # 读取依赖清单，并保留版本约束字符串供自动安装使用。
    specs = {}
    if not req_path.exists():
        return specs
    for raw in req_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key = line
        for sep in ("==", ">=", "<=", "~=", "!=", ">", "<"):
            if sep in line:
                key = line.split(sep, 1)[0].strip()
                break
        specs[key.lower()] = line
    return specs


def missing_imports():
    # 基于import测试判定缺失依赖，避免仅凭pip列表误判。
    missing = []
    for module_name in IMPORT_TO_PACKAGE:
        try:
            importlib.import_module(module_name)
        except Exception:
            missing.append(module_name)
    return missing


def install_missing(missing_modules, req_specs, mirror):
    if not missing_modules:
        return 0
    packages = []
    for mod in missing_modules:
        pkg = IMPORT_TO_PACKAGE[mod]
        packages.append(req_specs.get(pkg.lower(), pkg))
    cmd = [sys.executable, "-m", "pip", "install", "-i", mirror, *packages]
    print("[INFO] Installing:", ", ".join(packages))
    return subprocess.call(cmd)


def ensure_torch_gpu_if_needed(prefer_gpu: bool) -> int:
    # 用户开启prefer-gpu且检测到NVIDIA时，优先安装CUDA版PyTorch。
    if not prefer_gpu or which("nvidia-smi") is None:
        return 0
    try:
        import torch

        if torch.cuda.is_available():
            return 0
    except Exception:
        pass
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--index-url",
        "https://download.pytorch.org/whl/cu121",
        "--upgrade",
        "--force-reinstall",
        "--no-deps",
        "torch==2.4.1",
        "torchvision==0.19.1",
        "torchaudio==2.4.1",
    ]
    print("[INFO] Installing CUDA PyTorch (cu121)...")
    return subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser(description="Check project environment and auto-fix")
    parser.add_argument("--fix", action="store_true")
    parser.add_argument("--prefer-gpu", action="store_true")
    parser.add_argument("--mirror", default="https://pypi.tuna.tsinghua.edu.cn/simple")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    req = root / "env" / "requirements-py310.txt"
    req_specs = parse_requirements(req)

    print("[INFO] Python:", sys.version.split()[0], "| exe:", sys.executable)

    if args.fix:
        # 先处理GPU版PyTorch，再补装普通依赖，减少冲突。
        code = ensure_torch_gpu_if_needed(args.prefer_gpu)
        if code != 0:
            print("[WARN] CUDA torch install failed, continue.")

    missing = missing_imports()
    if missing and args.fix:
        code = install_missing(missing, req_specs, args.mirror)
        if code != 0:
            print("[ERROR] dependency install failed")
            return code
        missing = missing_imports()

    if missing:
        print("[ERROR] Missing:", ", ".join(missing))
        return 1

    import torch

    print("[INFO] Torch:", torch.__version__)
    print("[INFO] CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("[INFO] GPU:", torch.cuda.get_device_name(0))
    else:
        print("[INFO] Will fallback to CPU.")

    key_files = [
        # 关键路径检查：保证训练与推理基础文件齐全。
        root / "src" / "ultralytics-main" / "main.py",
        root / "scripts" / "infer" / "run_pose_stgcn.py",
        root / "configs" / "train" / "train_stgcn_out3.yaml",
        root / "configs" / "infer" / "runtime_out3.json",
        root / "data" / "stgcn_dataset_out3" / "train_data_c3v18.npy",
        root / "data" / "stgcn_dataset_out3" / "train_label.pkl",
    ]
    missing_files = [str(p) for p in key_files if not p.exists()]
    if missing_files:
        print("[ERROR] Missing files:")
        for p in missing_files:
            print(" -", p)
        return 2

    print("[OK] Environment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
