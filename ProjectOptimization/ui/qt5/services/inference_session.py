import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
from PyQt5.QtCore import QTimer
from ultralytics import YOLO

from scripts.infer.run_pose_stgcn import (
    build_stgcn_model,
    classify_action,
    draw_person_label,
    draw_pose,
    extract_multi_persons,
)


class InferenceSession:
    """内嵌推理会话：管理模型、视频流、定时器与结果回调。"""

    def __init__(
        self,
        parent,
        on_log,
        on_frame,
        on_table,
        on_state,
    ):
        self.parent = parent
        self.on_log = on_log
        self.on_frame = on_frame
        self.on_table = on_table
        self.on_state = on_state

        self.timer = None
        self.cap = None
        self.writer = None
        self.cfg = None
        self.yolo_model = None
        self.stgcn_model = None

        self.frame_idx = 0
        self.start_time = 0.0
        self.track_buffers = defaultdict(lambda: deque(maxlen=30))
        self.track_last_seen = {}
        self.track_pred = {}
        self.timer_interval_ms = 15
        self.enable_cv_window = False

    @property
    def is_running(self) -> bool:
        return self.timer is not None

    def start(self, cfg):
        if self.is_running:
            self.stop("已有推理会话在运行，已重置。")

        self.cfg = cfg
        try:
            self.yolo_model = YOLO(cfg.yolo_pose_weights)
            self.stgcn_model = build_stgcn_model(cfg)
        except Exception as e:
            self.stop(f"模型加载失败: {e}")
            return False

        self.cap = cv2.VideoCapture(cfg.video_path)
        if self.cap is None or not self.cap.isOpened():
            self.stop(f"视频打开失败: {cfg.video_path}")
            return False

        fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if cfg.save_path:
            out_path = Path(cfg.save_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        self.track_buffers = defaultdict(lambda: deque(maxlen=cfg.window_size))
        self.track_last_seen = {}
        self.track_pred = {}
        self.frame_idx = 0
        self.start_time = time.time()
        self.timer_interval_ms = max(10, int(1000 / max(float(fps), 1.0)))

        self.timer = QTimer(self.parent)
        self.timer.timeout.connect(self._tick)
        self.timer.start(self.timer_interval_ms)

        self.on_state("running")
        self.on_log(f"内嵌推理启动，视频: {cfg.video_path}")
        return True

    def stop(self, reason: str = ""):
        if self.timer is not None:
            self.timer.stop()
            self.timer.deleteLater()
            self.timer = None

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if self.writer is not None:
            self.writer.release()
            self.writer = None

        cv2.destroyAllWindows()
        self.cfg = None
        self.yolo_model = None
        self.stgcn_model = None
        self.track_buffers = defaultdict(lambda: deque(maxlen=30))
        self.track_last_seen = {}
        self.track_pred = {}

        self.on_table([])
        self.on_state("idle")
        if reason:
            self.on_log(reason)

    def _tick(self):
        try:
            if self.cap is None or self.cfg is None:
                self.stop("推理状态异常，已停止。")
                return

            ok, frame = self.cap.read()
            if not ok:
                self.stop("视频读取结束，推理完成。")
                return

            self.frame_idx += 1
            h, w = frame.shape[:2]

            results = self.yolo_model.track(
                frame,
                persist=True,
                tracker=self.cfg.tracker_yaml,
                verbose=False,
            )
            result = results[0]
            persons = extract_multi_persons(result, w, h, self.cfg)

            current_ids = set()
            for tid, info in persons.items():
                current_ids.add(tid)
                self.track_buffers[tid].append(info["kpt"])
                self.track_last_seen[tid] = self.frame_idx
                if len(self.track_buffers[tid]) == self.cfg.window_size and (self.frame_idx % self.cfg.stride == 0):
                    pred_id, pred_score = classify_action(self.stgcn_model, list(self.track_buffers[tid]), self.cfg)
                    pred_name = (
                        self.cfg.class_names[pred_id] if 0 <= pred_id < len(self.cfg.class_names) else f"class_{pred_id}"
                    )
                    self.track_pred[tid] = (pred_name, pred_score)

            for tid in list(self.track_buffers.keys()):
                if tid not in current_ids and len(self.track_buffers[tid]) > 0:
                    self.track_buffers[tid].append(self.track_buffers[tid][-1])

            for tid, last in list(self.track_last_seen.items()):
                if self.frame_idx - last > self.cfg.max_missing_frames:
                    self.track_buffers.pop(tid, None)
                    self.track_last_seen.pop(tid, None)
                    self.track_pred.pop(tid, None)

            rows = []
            for tid, info in sorted(persons.items(), key=lambda x: x[0]):
                draw_pose(frame, info["kpt"], conf_thres=self.cfg.kpt_conf_thres)
                name, score = self.track_pred.get(tid, ("warming...", 0.0))
                draw_person_label(frame, info["box"], tid, name, score)

                box = info.get("box")
                bbox = "-"
                if box is not None:
                    x1, y1, x2, y2 = [int(v) for v in box[:4]]
                    bbox = f"{x1},{y1},{x2},{y2}"
                rows.append(
                    {
                        "id": tid,
                        "det_conf": float(info.get("det_conf", 0.0)),
                        "action": name,
                        "action_conf": float(score),
                        "bbox": bbox,
                    }
                )

            elapsed = max(time.time() - self.start_time, 1e-6)
            fps_now = self.frame_idx / elapsed
            cv2.putText(
                frame,
                f"Frame:{self.frame_idx} FPS:{fps_now:.2f} Persons:{len(persons)}",
                (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            if self.writer is not None:
                self.writer.write(frame)

            self.on_frame(frame)
            self.on_table(rows)

            if self.cfg.show_window and self.enable_cv_window:
                cv2.imshow("Pose+ByteTrack+STGCN", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    self.stop("用户在OpenCV窗口中终止推理。")
        except Exception as e:
            self.stop(f"推理异常，已停止: {e}")
