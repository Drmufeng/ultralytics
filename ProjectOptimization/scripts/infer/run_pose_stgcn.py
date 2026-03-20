import argparse
import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch


def project_paths():
    # 统一解析工程根目录与源码目录，避免硬编码绝对路径。
    root = Path(__file__).resolve().parents[2]
    src = root / "src" / "ultralytics-main"
    return root, src


ROOT, SRC = project_paths()
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from net.st_gcn import Model  # noqa: E402
from ultralytics import YOLO  # noqa: E402


@dataclass
class RuntimeConfig:
    # 推理运行配置，全部来自JSON配置文件，可由命令行覆盖部分参数。
    yolo_pose_weights: str
    stgcn_weights: str
    video_path: str
    save_path: str | None
    tracker_yaml: str
    label_map_path: str | None
    num_class: int
    in_channels: int
    window_size: int
    stride: int
    class_names: list[str]
    person_conf_thres: float
    kpt_conf_thres: float
    max_missing_frames: int
    min_valid_kpts: int
    show_window: bool
    graph_layout: str
    graph_strategy: str
    device: str


def _resolve_path(raw_path: str | None) -> str | None:
    if raw_path is None:
        return None
    p = Path(raw_path)
    if p.is_absolute():
        return str(p)
    return str((SRC / p).resolve())


def load_runtime_config(config_path: Path) -> RuntimeConfig:
    # 读取并解析推理配置。
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    return RuntimeConfig(
        yolo_pose_weights=_resolve_path(raw["yolo_pose_weights"]),
        stgcn_weights=_resolve_path(raw["stgcn_weights"]),
        video_path=_resolve_path(raw["video_path"]),
        save_path=_resolve_path(raw.get("save_path")),
        tracker_yaml=str(raw.get("tracker_yaml", "bytetrack.yaml")),
        label_map_path=_resolve_path(raw.get("label_map_path")),
        num_class=int(raw.get("num_class", 0)),
        in_channels=int(raw.get("in_channels", 3)),
        window_size=int(raw.get("window_size", 30)),
        stride=int(raw.get("stride", 5)),
        class_names=list(raw.get("class_names", [])),
        person_conf_thres=float(raw.get("person_conf_thres", 0.25)),
        kpt_conf_thres=float(raw.get("kpt_conf_thres", 0.3)),
        max_missing_frames=int(raw.get("max_missing_frames", 30)),
        min_valid_kpts=int(raw.get("min_valid_kpts", 3)),
        show_window=bool(raw.get("show_window", True)),
        graph_layout=str(raw.get("graph_layout", "yolopose")),
        graph_strategy=str(raw.get("graph_strategy", "spatial")),
        device=str(raw.get("device", "cuda" if torch.cuda.is_available() else "cpu")),
    )


def finalize_config(cfg: RuntimeConfig) -> RuntimeConfig:
    # 补全类别名称、类别数与设备信息。
    if cfg.label_map_path and Path(cfg.label_map_path).exists() and not cfg.class_names:
        data = json.loads(Path(cfg.label_map_path).read_text(encoding="utf-8"))
        cfg.class_names = [str(x) for x in data.get("labels", [])]
    if cfg.num_class <= 0:
        cfg.num_class = len(cfg.class_names)
    if cfg.num_class <= 0:
        raise ValueError("num_class must be set or derivable from label_map")
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        cfg.device = "cpu"
    return cfg


def safe_torch_load(path: str, map_location: str = "cpu"):
    """优先使用 weights_only 安全加载，失败时回退旧模式以兼容历史权重。"""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception as e:
        print(f"[WARN] torch.load(weights_only=True) failed, fallback to legacy mode: {e}")
        return torch.load(path, map_location=map_location)


def infer_layout_from_stgcn_weights(weight_path: str) -> str | None:
    """从ST-GCN权重推断图结构：51->yolopose(17), 54->openpose(18)。"""
    p = Path(weight_path)
    if not p.exists():
        return None
    ckpt = safe_torch_load(str(p), map_location="cpu")
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    fixed = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    key = "data_bn.running_mean"
    if key not in fixed or not hasattr(fixed[key], "shape"):
        return None
    feat = int(fixed[key].shape[0])
    if feat % 3 != 0:
        return None
    v = feat // 3
    if v == 17:
        return "yolopose"
    if v == 18:
        return "openpose"
    return None


def infer_num_class_from_stgcn_weights(weight_path: str) -> int | None:
    """从ST-GCN权重推断类别数（读取fcn.weight输出通道）。"""
    p = Path(weight_path)
    if not p.exists():
        return None
    ckpt = safe_torch_load(str(p), map_location="cpu")
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    fixed = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    key = "fcn.weight"
    if key not in fixed or not hasattr(fixed[key], "shape"):
        return None
    if len(fixed[key].shape) < 1:
        return None
    return int(fixed[key].shape[0])


COCO_SKELETON = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def convert_coco17_to_layout(kpt17: np.ndarray, layout: str) -> np.ndarray:
    # 将YOLO COCO-17关键点转换为训练图结构（openpose-18或yolopose-17）。
    if layout == "yolopose":
        return kpt17
    if layout == "openpose":
        kpt18 = np.zeros((18, 3), dtype=np.float32)
        kpt18[0] = kpt17[0]
        neck_xy = 0.5 * (kpt17[5, :2] + kpt17[6, :2])
        neck_sc = 0.5 * (kpt17[5, 2] + kpt17[6, 2])
        kpt18[1] = np.array([neck_xy[0], neck_xy[1], neck_sc], dtype=np.float32)
        kpt18[2] = kpt17[6]
        kpt18[3] = kpt17[8]
        kpt18[4] = kpt17[10]
        kpt18[5] = kpt17[5]
        kpt18[6] = kpt17[7]
        kpt18[7] = kpt17[9]
        kpt18[8] = kpt17[12]
        kpt18[9] = kpt17[14]
        kpt18[10] = kpt17[16]
        kpt18[11] = kpt17[11]
        kpt18[12] = kpt17[13]
        kpt18[13] = kpt17[15]
        kpt18[14] = kpt17[2]
        kpt18[15] = kpt17[1]
        kpt18[16] = kpt17[4]
        kpt18[17] = kpt17[3]
        return kpt18
    raise ValueError(f"Unsupported graph_layout: {layout}")


def build_stgcn_model(cfg: RuntimeConfig):
    # 构建并加载ST-GCN权重（兼容module.前缀）。
    model = Model(
        in_channels=cfg.in_channels,
        num_class=cfg.num_class,
        graph_args={"layout": cfg.graph_layout, "strategy": cfg.graph_strategy},
        edge_importance_weighting=True,
    )
    ckpt = safe_torch_load(cfg.stgcn_weights, map_location="cpu")
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    fixed = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(fixed, strict=False)
    model.to(cfg.device).eval()
    return model


def build_stgcn_input(seq_kpts, device):
    t = len(seq_kpts)
    v = int(seq_kpts[0].shape[0])
    data = np.zeros((1, 3, t, v, 1), dtype=np.float32)
    for i, k in enumerate(seq_kpts):
        data[0, 0, i, :, 0] = k[:, 0]
        data[0, 1, i, :, 0] = k[:, 1]
        data[0, 2, i, :, 0] = k[:, 2]
    return torch.from_numpy(data).to(device)


def classify_action(model, seq_kpts, cfg: RuntimeConfig):
    x = build_stgcn_input(seq_kpts, cfg.device)
    with torch.no_grad():
        out = model(x)
        prob = torch.softmax(out, dim=1)[0]
    pred = int(torch.argmax(prob).item())
    score = float(prob[pred].item())
    return pred, score


def extract_multi_persons(result, img_w, img_h, cfg: RuntimeConfig):
    # 解析YOLO+ByteTrack结果，过滤低质量目标并输出标准骨架格式。
    persons = {}
    if result.boxes is None or result.keypoints is None or result.boxes.id is None:
        return persons
    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    boxes_conf = result.boxes.conf.cpu().numpy()
    track_ids = result.boxes.id.int().cpu().numpy()
    kpts_all = result.keypoints.data.cpu().numpy()

    for i in range(len(track_ids)):
        tid = int(track_ids[i])
        conf_box = float(boxes_conf[i])
        if conf_box < cfg.person_conf_thres:
            continue
        box = boxes_xyxy[i]
        kpt_raw = kpts_all[i]
        kpt_norm = np.zeros((17, 3), dtype=np.float32)
        valid_count = 0
        for j in range(17):
            x, y, s = kpt_raw[j]
            x = np.clip(x, 0, img_w - 1)
            y = np.clip(y, 0, img_h - 1)
            kpt_norm[j, 0] = (x / img_w) - 0.5
            kpt_norm[j, 1] = (y / img_h) - 0.5
            kpt_norm[j, 2] = s
            if s > cfg.kpt_conf_thres:
                valid_count += 1
        kpt_layout = convert_coco17_to_layout(kpt_norm, cfg.graph_layout)
        valid_count = int((kpt_layout[:, 2] > cfg.kpt_conf_thres).sum())
        if valid_count < cfg.min_valid_kpts:
            continue
        persons[tid] = {"box": box, "kpt": kpt_layout, "det_conf": conf_box}
    return persons


def draw_pose(img, pts, conf_thres=0.3):
    for a, b in COCO_SKELETON:
        if a >= len(pts) or b >= len(pts):
            continue
        xa, ya, sa = pts[a]
        xb, yb, sb = pts[b]
        if sa > conf_thres and sb > conf_thres:
            h, w = img.shape[:2]
            cv2.line(
                img,
                (int((xa + 0.5) * w), int((ya + 0.5) * h)),
                (int((xb + 0.5) * w), int((yb + 0.5) * h)),
                (255, 0, 0),
                2,
            )


def draw_person_label(img, box, track_id, action_name, action_score):
    x1, y1, x2, y2 = map(int, box[:4])
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
    label = f"ID {track_id} | {action_name} {action_score:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y_top = max(0, y1 - th - 8)
    cv2.rectangle(img, (x1, y_top), (x1 + tw + 6, y1), (0, 0, 0), -1)
    cv2.putText(img, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


def parse_args():
    # 提供配置文件与权重覆盖参数。
    parser = argparse.ArgumentParser(description="YOLOPose + ByteTrack + ST-GCN")
    parser.add_argument("--config", default=str(ROOT / "configs" / "infer" / "runtime_out3.json"))
    parser.add_argument("--stgcn-weights", default=None)
    parser.add_argument("--yolo-weights", default=None)
    parser.add_argument("--video-path", default=None)
    parser.add_argument("--save-path", default=None)
    parser.add_argument("--show-window", default=None)
    parser.add_argument("--layout", choices=["auto", "yolopose", "openpose"], default="auto")
    return parser.parse_args()


def main():
    # 主流程：检测跟踪 -> 骨架缓存 -> ST-GCN分类 -> 可视化/保存。
    args = parse_args()
    cfg = finalize_config(load_runtime_config(Path(args.config)))
    if args.yolo_weights:
        cfg.yolo_pose_weights = str(Path(args.yolo_weights).resolve())
    if args.stgcn_weights:
        cfg.stgcn_weights = str(Path(args.stgcn_weights).resolve())
    if args.video_path:
        cfg.video_path = str(Path(args.video_path).resolve())
    if args.save_path:
        cfg.save_path = str(Path(args.save_path).resolve())
    if args.show_window is not None:
        cfg.show_window = str(args.show_window).lower() in {"1", "true", "yes", "y"}

    if args.layout != "auto":
        cfg.graph_layout = args.layout
    else:
        inferred = infer_layout_from_stgcn_weights(cfg.stgcn_weights)
        if inferred in {"yolopose", "openpose"}:
            cfg.graph_layout = inferred
            print(f"[INFO] auto layout from ST-GCN weight: {cfg.graph_layout}")

    inferred_num_class = infer_num_class_from_stgcn_weights(cfg.stgcn_weights)
    if inferred_num_class and inferred_num_class > 0:
        if cfg.num_class != inferred_num_class:
            print(f"[INFO] auto num_class from ST-GCN weight: {inferred_num_class}")
            cfg.num_class = inferred_num_class
        if len(cfg.class_names) != cfg.num_class:
            if len(cfg.class_names) > cfg.num_class:
                cfg.class_names = cfg.class_names[: cfg.num_class]
            else:
                cfg.class_names = [f"class_{i}" for i in range(cfg.num_class)]

    yolo_model = YOLO(cfg.yolo_pose_weights)
    stgcn_model = build_stgcn_model(cfg)

    cap = cv2.VideoCapture(cfg.video_path)
    if not cap.isOpened():
        raise FileNotFoundError(cfg.video_path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if cfg.save_path:
        Path(cfg.save_path).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(cfg.save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    track_buffers = defaultdict(lambda: deque(maxlen=cfg.window_size))
    track_last_seen = {}
    track_pred = {}

    frame_idx = 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        results = yolo_model.track(frame, persist=True, tracker=cfg.tracker_yaml, verbose=False)
        result = results[0]
        persons = extract_multi_persons(result, w, h, cfg)

        current_ids = set()
        for tid, info in persons.items():
            current_ids.add(tid)
            track_buffers[tid].append(info["kpt"])
            track_last_seen[tid] = frame_idx
            if len(track_buffers[tid]) == cfg.window_size and (frame_idx % cfg.stride == 0):
                pred_id, pred_score = classify_action(stgcn_model, list(track_buffers[tid]), cfg)
                pred_name = cfg.class_names[pred_id] if 0 <= pred_id < len(cfg.class_names) else f"class_{pred_id}"
                track_pred[tid] = (pred_name, pred_score)

        for tid in list(track_buffers.keys()):
            if tid not in current_ids and len(track_buffers[tid]) > 0:
                track_buffers[tid].append(track_buffers[tid][-1])

        for tid, last in list(track_last_seen.items()):
            if frame_idx - last > cfg.max_missing_frames:
                track_buffers.pop(tid, None)
                track_last_seen.pop(tid, None)
                track_pred.pop(tid, None)

        for tid, info in persons.items():
            draw_pose(frame, info["kpt"], conf_thres=cfg.kpt_conf_thres)
            name, score = track_pred.get(tid, ("warming...", 0.0))
            draw_person_label(frame, info["box"], tid, name, score)

        elapsed = time.time() - t0
        cur_fps = frame_idx / max(elapsed, 1e-6)
        cv2.putText(frame, f"Frame:{frame_idx} FPS:{cur_fps:.2f} Persons:{len(persons)}", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if writer is not None:
            writer.write(frame)
        if cfg.show_window:
            cv2.imshow("Pose+ByteTrack+STGCN", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    if cfg.save_path:
        print(f"[INFO] saved: {cfg.save_path}")


if __name__ == "__main__":
    main()
