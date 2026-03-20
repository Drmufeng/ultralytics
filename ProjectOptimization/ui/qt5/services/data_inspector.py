from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class DataStatus:
    ok: bool
    message: str
    level: str
    keypoints: int | None


def _detect_out3_layout(data_dir: Path) -> DataStatus | None:
    required = [
        data_dir / "train_data_c3v18.npy",
        data_dir / "train_label.pkl",
        data_dir / "val_data_c3v18.npy",
        data_dir / "val_label.pkl",
        data_dir / "label_map.json",
    ]
    if not all(p.exists() for p in required):
        return None
    try:
        arr = np.load(str(data_dir / "train_data_c3v18.npy"), mmap_mode="r")
        v = int(arr.shape[3])
    except Exception as e:
        return DataStatus(False, f"数据检查异常: {e}", "error", None)

    if v == 17:
        return DataStatus(True, "数据完整，检测到V=17（建议yolopose）", "ok", 17)
    if v == 18:
        return DataStatus(True, "数据完整，检测到V=18（建议openpose；若需17请先转换）", "ok", 18)
    return DataStatus(False, f"数据完整，但关键点V={v}不在17/18范围", "error", v)


def _detect_out17_layout(data_dir: Path) -> DataStatus | None:
    required = [
        data_dir / "train_data.npy",
        data_dir / "train_label.pkl",
        data_dir / "val_data.npy",
        data_dir / "val_label.pkl",
    ]
    if not all(p.exists() for p in required):
        return None
    try:
        arr = np.load(str(data_dir / "train_data.npy"), mmap_mode="r")
        v = int(arr.shape[3])
    except Exception as e:
        return DataStatus(False, f"数据检查异常: {e}", "error", None)

    if v == 17:
        return DataStatus(True, "数据完整，检测到V=17（建议yolopose）", "ok", 17)
    if v == 18:
        return DataStatus(True, "数据完整，检测到V=18（建议openpose）", "ok", 18)
    return DataStatus(False, f"数据完整，但关键点V={v}不在17/18范围", "error", v)


def inspect_stgcn_dataset(data_dir: Path) -> DataStatus:
    if not data_dir.exists():
        return DataStatus(False, f"数据目录不存在: {data_dir}", "error", None)

    status = _detect_out3_layout(data_dir)
    if status is not None:
        return status

    status = _detect_out17_layout(data_dir)
    if status is not None:
        return status

    known = [
        "train_data_c3v18.npy",
        "val_data_c3v18.npy",
        "train_data.npy",
        "val_data.npy",
        "train_label.pkl",
        "val_label.pkl",
    ]
    missing = [name for name in known if not (data_dir / name).exists()]
    return DataStatus(False, "未识别的数据集结构，可能缺失: " + ", ".join(missing), "error", None)
