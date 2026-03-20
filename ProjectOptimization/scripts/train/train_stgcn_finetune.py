import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml


def _to_abs(root: Path, p: str) -> Path:
    obj = Path(p)
    return obj if obj.is_absolute() else (root / obj)


def _resolve_cfg_rel(cfg_path: Path, root: Path, raw: str, require_exists: bool = True) -> Path:
    obj = Path(raw)
    if obj.is_absolute():
        return obj
    by_cfg = (cfg_path.parent / obj).resolve()
    if (not require_exists) or by_cfg.exists():
        return by_cfg
    return (root / obj).resolve()


def _expected_v(layout: str) -> int:
    return 17 if layout == "yolopose" else 18 if layout == "openpose" else -1


def _resolve_layout(config_layout: str, data_v: int, layout_arg: str) -> str:
    if layout_arg in {"yolopose", "openpose"}:
        return layout_arg
    if layout_arg == "auto":
        if data_v == 17:
            return "yolopose"
        if data_v == 18:
            return "openpose"
    return config_layout


def _prepare_config(root: Path, config_rel: str, layout_arg: str) -> tuple[Path, str]:
    cfg_path = _to_abs(root, config_rel).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    config_layout = cfg.get("model_args", {}).get("graph_args", {}).get("layout", "yolopose")
    train_data_raw = cfg.get("train_feeder_args", {}).get("data_path", "")
    test_data_raw = cfg.get("test_feeder_args", {}).get("data_path", "")
    train_label_raw = cfg.get("train_feeder_args", {}).get("label_path", "")
    test_label_raw = cfg.get("test_feeder_args", {}).get("label_path", "")
    work_dir_raw = cfg.get("work_dir", "")

    data_path = _resolve_cfg_rel(cfg_path, root, train_data_raw)
    if not data_path.exists():
        raise FileNotFoundError(f"训练数据不存在: {data_path}")

    arr = np.load(str(data_path), mmap_mode="r")
    data_v = int(arr.shape[3])

    chosen_layout = _resolve_layout(config_layout, data_v, layout_arg)
    exp_v = _expected_v(chosen_layout)
    if exp_v == -1:
        raise ValueError(f"未知layout: {chosen_layout}")
    if data_v != exp_v:
        raise ValueError(
            f"关键点数不匹配: 数据V={data_v}, layout={chosen_layout}期望V={exp_v}。"
            f"如需17点请先准备V=17数据，或选择openpose。"
        )

    cfg.setdefault("model_args", {}).setdefault("graph_args", {})["layout"] = chosen_layout
    if work_dir_raw:
        cfg["work_dir"] = str(_resolve_cfg_rel(cfg_path, root, work_dir_raw, require_exists=False))
    cfg.setdefault("train_feeder_args", {})["data_path"] = str(data_path)
    if train_label_raw:
        cfg["train_feeder_args"]["label_path"] = str(_resolve_cfg_rel(cfg_path, root, train_label_raw))
    if test_data_raw:
        cfg.setdefault("test_feeder_args", {})["data_path"] = str(_resolve_cfg_rel(cfg_path, root, test_data_raw))
    if test_label_raw:
        cfg.setdefault("test_feeder_args", {})["label_path"] = str(_resolve_cfg_rel(cfg_path, root, test_label_raw))

    tmp_dir = root / "logs" / "run_logs"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_cfg = tmp_dir / f"_tmp_train_finetune_{int(time.time())}.yaml"
    tmp_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return tmp_cfg, chosen_layout


def main() -> int:
    # 微调阶段：基于上一阶段权重继续训练，学习率通常更小。
    parser = argparse.ArgumentParser(description="ST-GCN finetune training")
    parser.add_argument("--config", default="configs/train/train_stgcn_out3_finetune.yaml")
    parser.add_argument("--weights", default="models/stgcn/out3/best.pt")
    parser.add_argument("--layout", choices=["auto", "yolopose", "openpose"], default="auto")
    parser.add_argument("--log-interval", type=int, default=1)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    src = root / "src" / "ultralytics-main"

    try:
        cfg_path, chosen_layout = _prepare_config(root, args.config, args.layout)
    except Exception as e:
        print(f"[ERROR] 配置检查失败: {e}")
        return 2

    weights_path = _to_abs(root, args.weights).resolve()
    if not weights_path.exists():
        print(f"[WARN] 微调初始权重不存在，将按给定路径尝试加载: {weights_path}")

    print(f"[INFO] 训练layout: {chosen_layout}")
    # 保持入口一致，复用同一底层训练逻辑。
    cmd = [
        sys.executable,
        str(src / "main.py"),
        "recognition",
        "-c",
        str(cfg_path),
        "--weights",
        str(weights_path),
        "--log_interval",
        str(max(1, int(args.log_interval))),
    ]
    print("[INFO]", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(src))


if __name__ == "__main__":
    raise SystemExit(main())
