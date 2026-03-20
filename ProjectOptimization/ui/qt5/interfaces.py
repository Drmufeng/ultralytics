from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class TrainEvent:
    # 训练过程事件数据结构，供Qt5界面消费。
    epoch: int
    metrics: Dict[str, Any]


class Qt5Hooks:
    """Qt5 UI钩子接口占位，当前只定义回调契约。"""

    def on_train_start(self, config: Dict[str, Any]) -> None:
        pass

    def on_train_epoch(self, event: TrainEvent) -> None:
        pass

    def on_best_model(self, path: str, score: float) -> None:
        pass

    def on_infer_frame(self, payload: Dict[str, Any]) -> None:
        pass

    def on_error(self, message: str) -> None:
        pass
