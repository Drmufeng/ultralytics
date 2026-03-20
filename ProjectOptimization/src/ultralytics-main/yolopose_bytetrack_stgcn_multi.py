import os
import cv2
import time
import torch
import numpy as np
from collections import deque, defaultdict
from ultralytics import YOLO

# =========================
# 1) 配置区（按你的工程修改）
# =========================
YOLO_POSE_WEIGHTS = r"runs/pose/train/weights/best.pt"
STGCN_WEIGHTS = r"work_dir-gcn/recognition/kinetics_skeleton/ST_GCN/epoch50_model.pt"   # 改成你的ST-GCN权重
VIDEO_PATH = r"dataset/GCNData/BaseballPitch/v_BaseballPitch_g04_c01.avi"
SAVE_PATH = r"output_multi_action.mp4"                                # 不保存可设为 None

# ST-GCN 模型导入（按你的工程）
from net.st_gcn import Model

# =========================
# 2) ST-GCN参数（和训练保持一致）
# =========================
NUM_CLASS = 4
IN_CHANNELS = 3
WINDOW_SIZE = 30        # 和训练样本长度一致（例如30/50）
STRIDE = 5              # 每隔多少帧更新一次分类
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = ["BaseballPitch", "Basketball", "Bowling", "CricketBowling"]  # 改成你的动作类别名

# 阈值
PERSON_CONF_THRES = 0.25
KPT_CONF_THRES = 0.3

# 多人轨迹管理
MAX_MISSING_FRAMES = 30   # 某个ID连续多少帧没出现就删除
MIN_VALID_KPTS = 3        # 一帧至少多少个有效点才算“有效骨架”

SHOW_WINDOW = True

# =========================
# 3) COCO 17关键点骨架连接
# =========================
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

# =========================
# 4) ST-GCN模型构建
# =========================
def build_stgcn_model():
    model = Model(
        in_channels=IN_CHANNELS,
        num_class=NUM_CLASS,
        graph_args={"layout": "yolopose", "strategy": "spatial"},
        edge_importance_weighting=True
    )

    ckpt = torch.load(STGCN_WEIGHTS, map_location="cpu")

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    new_sd = {}
    for k, v in state_dict.items():
        new_sd[k[7:]] = v if k.startswith("module.") else v
        if not k.startswith("module."):
            new_sd[k] = v

    # 上面写法会重复插入非module键，这里修正一下
    fixed_sd = {}
    for k, v in state_dict.items():
        fixed_sd[k[7:] if k.startswith("module.") else k] = v

    model.load_state_dict(fixed_sd, strict=False)
    model.to(DEVICE)
    model.eval()
    return model

# =========================
# 5) ST-GCN输入构建
#    seq_kpts: list[T], 每个元素(17,3) [x_norm, y_norm, score]
#    输出: (1,3,T,17,1)
# =========================
def build_stgcn_input(seq_kpts):
    T = len(seq_kpts)
    data = np.zeros((1, 3, T, 17, 1), dtype=np.float32)

    for t, k in enumerate(seq_kpts):
        data[0, 0, t, :, 0] = k[:, 0]
        data[0, 1, t, :, 0] = k[:, 1]
        data[0, 2, t, :, 0] = k[:, 2]

    return torch.from_numpy(data).to(DEVICE)

@torch.no_grad()
def classify_action(stgcn_model, seq_kpts):
    x = build_stgcn_input(seq_kpts)
    logits = stgcn_model(x)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    prob = torch.softmax(logits, dim=1)[0]
    pred_id = int(torch.argmax(prob).item())
    pred_score = float(prob[pred_id].item())
    return pred_id, pred_score

# =========================
# 6) 从YOLO+ByteTrack结果中提取多人（按track_id）
#    返回 dict: {track_id: {"box":..., "kpt":(17,3)}}
# =========================
def extract_multi_persons(result, img_w, img_h):
    persons = {}

    # 没有检测到
    if result.boxes is None or len(result.boxes) == 0:
        return persons
    if result.keypoints is None or result.keypoints.data is None:
        return persons

    boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy()          # [N,4]
    boxes_conf = result.boxes.conf.detach().cpu().numpy() if result.boxes.conf is not None else None
    boxes_id = None
    if hasattr(result.boxes, "id") and result.boxes.id is not None:
        boxes_id = result.boxes.id.detach().cpu().numpy().astype(int)   # [N]
    else:
        # 如果没有track id，就退化为按索引（不稳定，不建议）
        boxes_id = np.arange(len(boxes_xyxy), dtype=int)

    kpts_all = result.keypoints.data.detach().cpu().numpy()        # [N,17,3]

    n = min(len(boxes_xyxy), len(kpts_all), len(boxes_id))
    for i in range(n):
        conf_box = float(boxes_conf[i]) if boxes_conf is not None else 1.0
        if conf_box < PERSON_CONF_THRES:
            continue

        tid = int(boxes_id[i])
        box = boxes_xyxy[i]

        # 归一化关键点
        kpt_raw = kpts_all[i]  # (17,3)
        kpt_norm = np.zeros((17, 3), dtype=np.float32)
        valid_count = 0
        for j in range(17):
            x, y, s = kpt_raw[j]
            x = np.clip(x / max(img_w, 1), 0.0, 1.0)
            y = np.clip(y / max(img_h, 1), 0.0, 1.0)
            kpt_norm[j] = [x, y, s]
            if s > KPT_CONF_THRES:
                valid_count += 1

        # 太差的骨架可跳过（你也可以保留）
        if valid_count < MIN_VALID_KPTS:
            continue

        persons[tid] = {
            "box": box,
            "kpt": kpt_norm,
            "det_conf": conf_box
        }

    return persons

# =========================
# 7) 可视化
# =========================
def draw_pose(img, kpt_norm, conf_thres=0.3):
    h, w = img.shape[:2]
    pts = []
    for i in range(17):
        x = int(kpt_norm[i, 0] * w)
        y = int(kpt_norm[i, 1] * h)
        s = float(kpt_norm[i, 2])
        pts.append((x, y, s))

    for a, b in COCO_SKELETON:
        xa, ya, sa = pts[a]
        xb, yb, sb = pts[b]
        if sa > conf_thres and sb > conf_thres:
            cv2.line(img, (xa, ya), (xb, yb), (255, 0, 0), 2)

    for x, y, s in pts:
        if s > conf_thres:
            cv2.circle(img, (x, y), 3, (0, 255, 0), -1)

def draw_person_label(img, box, track_id, action_name, action_score):
    x1, y1, x2, y2 = map(int, box[:4])
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)

    label = f"ID {track_id} | {action_name} {action_score:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y_top = max(0, y1 - th - 8)
    cv2.rectangle(img, (x1, y_top), (x1 + tw + 6, y1), (0, 0, 0), -1)
    cv2.putText(img, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

# =========================
# 8) 主流程（多人）
# =========================
def main():
    print("Loading YOLOPose...")
    yolo_model = YOLO(YOLO_POSE_WEIGHTS)

    print("Loading ST-GCN...")
    stgcn_model = build_stgcn_model()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"视频打开失败: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if SAVE_PATH is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(SAVE_PATH, fourcc, fps, (width, height))

    # 每个track_id的状态
    track_buffers = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))   # {id: deque[(17,3), ...]}
    track_last_seen = {}                                             # {id: frame_idx}
    track_pred = {}                                                  # {id: (name, score)}

    frame_idx = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # --- YOLOPose + ByteTrack ---
        # persist=True 会在连续帧间保持跟踪状态
        results = yolo_model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        result = results[0]

        # 提取本帧所有已跟踪的人
        persons = extract_multi_persons(result, width, height)

        # 更新各track的缓冲区
        current_ids = set()
        for tid, info in persons.items():
            current_ids.add(tid)
            track_buffers[tid].append(info["kpt"])
            track_last_seen[tid] = frame_idx

            # 到达窗口长度后定期分类
            if len(track_buffers[tid]) == WINDOW_SIZE and (frame_idx % STRIDE == 0):
                pred_id, pred_score = classify_action(stgcn_model, list(track_buffers[tid]))
                pred_name = CLASS_NAMES[pred_id] if 0 <= pred_id < len(CLASS_NAMES) else f"class_{pred_id}"
                track_pred[tid] = (pred_name, pred_score)

        # 对“本帧没出现”的已知track做补帧（可选）
        # 这里用“最后一帧骨架”补齐，有利于短时遮挡时不断流
        for tid in list(track_buffers.keys()):
            if tid not in current_ids:
                # 若这个ID之前存在过，用最后一帧骨架补一帧
                if len(track_buffers[tid]) > 0:
                    track_buffers[tid].append(track_buffers[tid][-1])

        # 清理长时间消失的track
        to_delete = []
        for tid, last_seen in track_last_seen.items():
            if frame_idx - last_seen > MAX_MISSING_FRAMES:
                to_delete.append(tid)
        for tid in to_delete:
            track_buffers.pop(tid, None)
            track_last_seen.pop(tid, None)
            track_pred.pop(tid, None)

        # --- 可视化本帧出现的人 ---
        for tid, info in persons.items():
            box = info["box"]
            kpt = info["kpt"]
            draw_pose(frame, kpt, conf_thres=KPT_CONF_THRES)

            if tid in track_pred:
                name, score = track_pred[tid]
            else:
                name, score = "warming...", 0.0
            draw_person_label(frame, box, tid, name, score)

        # 显示整体信息
        elapsed = time.time() - t0
        cur_fps = frame_idx / max(elapsed, 1e-6)
        cv2.putText(frame, f"Frame:{frame_idx}  FPS:{cur_fps:.2f}  Persons:{len(persons)}",
                    (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if writer is not None:
            writer.write(frame)

        if SHOW_WINDOW:
            cv2.imshow("YOLOPose + ByteTrack + ST-GCN (Multi-person)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    print("Done.")
    if SAVE_PATH is not None:
        print(f"结果视频已保存: {SAVE_PATH}")

if __name__ == "__main__":
    main()