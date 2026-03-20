from typing import Any, Dict, List

from .interfaces import Qt5Hooks, TrainEvent


class EventBus:
    def __init__(self):
        # 维护UI监听器列表，训练/推理过程通过事件总线广播。
        self._hooks: List[Qt5Hooks] = []

    def register(self, hook: Qt5Hooks) -> None:
        self._hooks.append(hook)

    def emit_train_start(self, config: Dict[str, Any]) -> None:
        for h in self._hooks:
            h.on_train_start(config)

    def emit_train_epoch(self, epoch: int, metrics: Dict[str, Any]) -> None:
        event = TrainEvent(epoch=epoch, metrics=metrics)
        for h in self._hooks:
            h.on_train_epoch(event)

    def emit_best_model(self, path: str, score: float) -> None:
        for h in self._hooks:
            h.on_best_model(path, score)

    def emit_infer_frame(self, payload: Dict[str, Any]) -> None:
        for h in self._hooks:
            h.on_infer_frame(payload)

    def emit_error(self, message: str) -> None:
        for h in self._hooks:
            h.on_error(message)
