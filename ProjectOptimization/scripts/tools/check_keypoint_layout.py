import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def infer_expected_v(config_path: Path, layout_override: str) -> tuple[str, int]:
    suffix = config_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        layout = (
            data.get("model_args", {})
            .get("graph_args", {})
            .get("layout", "yolopose")
        )
    elif suffix == ".json":
        data = json.loads(config_path.read_text(encoding="utf-8"))
        layout = data.get("graph_layout", "yolopose")
    else:
        raise ValueError(f"Unsupported config type: {config_path}")

    if layout_override in {"yolopose", "openpose"}:
        layout = layout_override

    expected_v = 17 if layout == "yolopose" else 18 if layout == "openpose" else -1
    return layout, expected_v


def main() -> int:
    parser = argparse.ArgumentParser(description="检查关键点维度与配置layout是否一致")
    parser.add_argument("--data", required=True, help="npy数据路径，例如 train_data_*.npy")
    parser.add_argument("--config", required=True, help="训练yaml或推理json配置路径")
    parser.add_argument("--layout", choices=["auto", "yolopose", "openpose"], default="auto", help="可选覆盖layout")
    args = parser.parse_args()

    data_path = Path(args.data)
    cfg_path = Path(args.config)

    if not data_path.exists():
        print(f"[ERROR] 数据文件不存在: {data_path}")
        return 1
    if not cfg_path.exists():
        print(f"[ERROR] 配置文件不存在: {cfg_path}")
        return 1

    arr = np.load(str(data_path), mmap_mode="r")
    if arr.ndim < 4:
        print(f"[ERROR] 数据维度异常，期望至少4维，实际: {arr.shape}")
        return 2

    v = int(arr.shape[3])
    layout, expected_v = infer_expected_v(cfg_path, args.layout)

    if args.layout == "auto":
        if v == 17:
            layout, expected_v = "yolopose", 17
        elif v == 18:
            layout, expected_v = "openpose", 18

    print(f"[INFO] 数据文件: {data_path}")
    print(f"[INFO] 数据shape: {arr.shape}")
    print(f"[INFO] 关键点数V: {v}")
    print(f"[INFO] 配置layout: {layout}")
    print(f"[INFO] 期望关键点数: {expected_v}")

    if expected_v == -1:
        print("[WARN] 未知layout，无法自动判定。")
        return 3

    if v != expected_v:
        print("[ERROR] 关键点数与配置不一致，请修正数据或layout。")
        return 4

    print("[OK] 关键点数与配置一致。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
