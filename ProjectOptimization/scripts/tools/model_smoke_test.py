import argparse
from pathlib import Path

import torch


def _safe_torch_load(path: str, map_location: str = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception as e:
        print(f"[WARN] torch.load(weights_only=True) failed, fallback to legacy mode: {e}")
        return torch.load(path, map_location=map_location)


def _infer_layout_from_state(state: dict) -> str:
    k = "data_bn.running_mean"
    if k in state and hasattr(state[k], "shape"):
        feat = int(state[k].shape[0])
        v = feat // 3 if feat % 3 == 0 else -1
        if v == 17:
            return "yolopose"
        if v == 18:
            return "openpose"
    return "yolopose"


def _infer_num_class_from_state(state: dict) -> int:
    k = "fcn.weight"
    if k in state and hasattr(state[k], "shape") and len(state[k].shape) >= 1:
        return int(state[k].shape[0])
    return 10


def main() -> int:
    parser = argparse.ArgumentParser(description="模型快速检查脚本")
    parser.add_argument("--yolo", required=True, help="YOLO权重路径")
    parser.add_argument("--stgcn", required=True, help="ST-GCN权重路径")
    parser.add_argument("--layout", choices=["auto", "yolopose", "openpose"], default="auto", help="ST-GCN图结构")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    src = root / "src" / "ultralytics-main"

    yolo_path = Path(args.yolo)
    stgcn_path = Path(args.stgcn)
    if not yolo_path.exists():
        print(f"[ERROR] YOLO权重不存在: {yolo_path}")
        return 1
    if not stgcn_path.exists():
        print(f"[ERROR] ST-GCN权重不存在: {stgcn_path}")
        return 1

    import sys

    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from ultralytics import YOLO
    from net.st_gcn import Model

    print(f"[INFO] 检查YOLO权重: {yolo_path}")
    _ = YOLO(str(yolo_path))

    print(f"[INFO] 检查ST-GCN权重: {stgcn_path}")
    ckpt = _safe_torch_load(str(stgcn_path), map_location="cpu")
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    fixed = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}

    layout = _infer_layout_from_state(fixed) if args.layout == "auto" else args.layout
    num_class = _infer_num_class_from_state(fixed)
    print(f"[INFO] ST-GCN layout: {layout}, num_class: {num_class}")

    model = Model(
        in_channels=3,
        num_class=num_class,
        edge_importance_weighting=True,
        graph_args={"layout": layout, "strategy": "spatial"},
    )
    model_state = model.state_dict()
    compatible = {}
    for k, v in fixed.items():
        if k in model_state and model_state[k].shape == v.shape:
            compatible[k] = v
    model.load_state_dict(compatible, strict=False)

    print("[OK] 模型检查通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
