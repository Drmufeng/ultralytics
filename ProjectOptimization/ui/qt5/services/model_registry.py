import re
from pathlib import Path


def _list_pt_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(folder.glob("*.pt"), key=lambda p: ("best" not in p.name.lower(), p.name.lower()))


def _unique_files(files: list[Path]) -> list[Path]:
    seen = set()
    out = []
    for p in files:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _is_epoch_file(p: Path) -> bool:
    return re.match(r"epoch\d+_model\.pt$", p.name.lower()) is not None


def _latest_epoch_file(files: list[Path]) -> list[Path]:
    epochs = []
    for p in files:
        m = re.match(r"epoch(\d+)_model\.pt$", p.name.lower())
        if m:
            epochs.append((int(m.group(1)), p))
    if not epochs:
        return []
    epochs.sort(key=lambda x: x[0], reverse=True)
    return [epochs[0][1]]


def _compact_files(files: list[Path]) -> list[Path]:
    """精简模式：优先best，其次非epoch文件，再补最新epoch。"""
    files = _unique_files(files)
    best = [p for p in files if "best" in p.name.lower()]
    non_epoch = [p for p in files if not _is_epoch_file(p) and p not in best]
    latest_epoch = [p for p in _latest_epoch_file(files) if p not in best and p not in non_epoch]
    return _unique_files(best + non_epoch + latest_epoch)


def _list_recursive_pt_files(folder: Path, pattern: str = "**/*.pt") -> list[Path]:
    if not folder.exists():
        return []
    return sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)


def _latest_ablation_best(files: list[Path]) -> list[Path]:
    cands = [p for p in files if p.name.lower() == "best.pt"]
    if not cands:
        return []
    cands.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    return [cands[0]]


class ModelRegistry:
    """模型目录扫描与默认选择策略。"""

    def __init__(self, root: Path):
        self.root = root

    def yolo_files(self, show_all: bool) -> list[Path]:
        base_files = _list_pt_files(self.root / "models" / "yolo" / "custom") + _list_pt_files(
            self.root / "models" / "yolo" / "pretrained"
        )
        ablation_files = _list_recursive_pt_files(self.root / "outputs" / "eval" / "ablation", "**/weights/*.pt")
        if show_all:
            return _unique_files(base_files + ablation_files)
        return _compact_files(base_files + _latest_ablation_best(ablation_files))

    def stgcn_files(self, show_all: bool) -> list[Path]:
        files = (
            _list_pt_files(self.root / "models" / "stgcn" / "out17")
            + _list_pt_files(self.root / "models" / "stgcn" / "out17_finetune")
            + _list_pt_files(self.root / "models" / "stgcn" / "out3")
            + _list_pt_files(self.root / "models" / "stgcn" / "out3_finetune")
            + _list_pt_files(self.root / "models" / "stgcn" / "legacy")
            + _list_pt_files(self.root / "models" / "stgcn" / "pretrained")
        )
        return _unique_files(files) if show_all else _compact_files(files)

    def stgcn_pretrain_files(self, show_all: bool) -> list[Path]:
        files = _list_pt_files(self.root / "models" / "stgcn" / "pretrained") + self.stgcn_files(show_all)
        return _unique_files(files)

    def preferred_yolo(self) -> Path:
        return (self.root / "models" / "yolo" / "custom" / "yolo_pose_best.pt").resolve()

    def preferred_stgcn(self) -> Path:
        candidates = [
            self.root / "models" / "stgcn" / "out17" / "best.pt",
            self.root / "models" / "stgcn" / "out17_finetune" / "best.pt",
            self.root / "models" / "stgcn" / "out3" / "best.pt",
            self.root / "models" / "stgcn" / "out3_finetune" / "best.pt",
        ]
        for p in candidates:
            if p.exists():
                return p.resolve()
        return candidates[0].resolve()
