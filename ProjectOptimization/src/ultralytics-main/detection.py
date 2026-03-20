import sys
import os
import time
import cv2
import torch
import numpy as np
from collections import deque, defaultdict
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QTableWidget, QTableWidgetItem,
                             QHeaderView, QGroupBox, QGridLayout, QSplitter, QFrame, QStyleFactory,
                             QSlider, QDoubleSpinBox)
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QBrush, QColor, QIcon, QPainter
from PyQt5.QtCore import Qt, QTimer, pyqtSlot, QSize, QRect
from PyQt5 import uic
from ultralytics import YOLO
from net.st_gcn import Model as STGCNModel

# ==================== ST-GCN 配置（和训练保持一致） ====================
STGCN_WEIGHTS = r"work_dir-gcn/recognition/kinetics_skeleton/ST_GCN/epoch50_model.pt"
NUM_CLASS = 4
IN_CHANNELS = 3
WINDOW_SIZE = 30
STRIDE = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["BaseballPitch", "Basketball", "Bowling", "CricketBowling"]

PERSON_CONF_THRES = 0.25
KPT_CONF_THRES = 0.3
MIN_VALID_KPTS = 3
MAX_MISSING_FRAMES = 30

COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]


# ==================== ST-GCN 工具函数 ====================
def build_stgcn_model(weights_path):
    model = STGCNModel(
        in_channels=IN_CHANNELS,
        num_class=NUM_CLASS,
        graph_args={"layout": "yolopose", "strategy": "spatial"},
        edge_importance_weighting=True,
    )

    ckpt = torch.load(weights_path, map_location="cpu")

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    fixed_sd = {}
    for k, v in state_dict.items():
        fixed_sd[k[7:] if k.startswith("module.") else k] = v

    model.load_state_dict(fixed_sd, strict=False)
    model.to(DEVICE)
    model.eval()
    return model


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


def extract_persons(result, img_w, img_h, use_track_id=True):
    """从 YOLO Pose 结果中提取人物框 + 归一化关键点"""
    persons = {}

    if result.boxes is None or len(result.boxes) == 0:
        return persons
    if result.keypoints is None or result.keypoints.data is None:
        return persons

    boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy()
    boxes_conf = result.boxes.conf.detach().cpu().numpy() if result.boxes.conf is not None else None

    if use_track_id and hasattr(result.boxes, "id") and result.boxes.id is not None:
        boxes_id = result.boxes.id.detach().cpu().numpy().astype(int)
    else:
        boxes_id = np.arange(len(boxes_xyxy), dtype=int)

    kpts_all = result.keypoints.data.detach().cpu().numpy()

    n = min(len(boxes_xyxy), len(kpts_all), len(boxes_id))
    for i in range(n):
        conf_box = float(boxes_conf[i]) if boxes_conf is not None else 1.0
        if conf_box < PERSON_CONF_THRES:
            continue

        tid = int(boxes_id[i])
        box = boxes_xyxy[i]

        kpt_raw = kpts_all[i]
        kpt_norm = np.zeros((17, 3), dtype=np.float32)
        valid_count = 0
        for j in range(17):
            x, y, s = kpt_raw[j]
            x = np.clip(x / max(img_w, 1), 0.0, 1.0)
            y = np.clip(y / max(img_h, 1), 0.0, 1.0)
            kpt_norm[j] = [x, y, s]
            if s > KPT_CONF_THRES:
                valid_count += 1

        if valid_count < MIN_VALID_KPTS:
            continue

        persons[tid] = {
            "box": box,
            "kpt": kpt_norm,
            "det_conf": conf_box,
        }

    return persons


# ==================== 主界面 ====================
class DetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        uic.loadUi("ui.ui", self)

        # 表格：序号 / 检测类别 / 检测置信度 / 动作分类 / 分类置信度 / 位置 / 检测类型
        self.result_table.setColumnCount(7)
        self.result_table.setHorizontalHeaderLabels(
            ["序号", "检测类别", "检测置信度", "动作分类", "分类置信度", "位置", "检测类型"]
        )
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.background_image = None
        if os.path.exists("background.jpg"):
            self.background_image = QPixmap("background.jpg")
        self.setAttribute(Qt.WA_StyledBackground, True)

        # ========== 双模型：YOLO Pose（检测）+ ST-GCN（动作分类） ==========
        self.det_model = YOLO("best.pt")
        self.det_weight_path = "best.pt"

        self.stgcn_model = None
        self.stgcn_weight_path = None
        if os.path.exists(STGCN_WEIGHTS):
            try:
                self.stgcn_model = build_stgcn_model(STGCN_WEIGHTS)
                self.stgcn_weight_path = STGCN_WEIGHTS
                print(f"ST-GCN 模型已加载: {STGCN_WEIGHTS}")
            except Exception as e:
                print(f"ST-GCN 模型加载失败: {e}")

        # 多人时序轨迹管理（视频/摄像头用）
        self.track_buffers = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))
        self.track_last_seen = {}
        self.track_pred = {}
        self.frame_idx = 0

        # 通用变量
        self.original_image = None
        self.result_image = None
        self.detection_results = []
        self.camera = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera)
        self.is_detecting_camera = False
        self.camera_detection_timer = QTimer()
        self.camera_detection_timer.timeout.connect(self.detect_camera_frame)

        self.video_capture = None
        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self.update_video_frame)
        self.is_detecting_video = False
        self.video_detection_timer = QTimer()
        self.video_detection_timer.timeout.connect(self.detect_video_frame)
        self.video_path = None
        self.video_frame_count = 0
        self.current_frame = 0

        self.conf_threshold = 0.25
        self.iou_threshold = 0.45

        # 视频录制相关
        self.video_writer = None
        self.save_video_path = None
        self.is_recording = False

        # UI：添加动态按钮
        self.load_weight_btn.setText("加载检测权重")
        btn_style = self.load_weight_btn.styleSheet()

        self.load_cls_weight_btn = QPushButton("加载分类权重")
        self.load_cls_weight_btn.setMinimumHeight(40)
        self.load_cls_weight_btn.setStyleSheet(btn_style)

        self.save_video_btn = QPushButton("保存视频")
        self.save_video_btn.setMinimumHeight(40)
        self.save_video_btn.setStyleSheet(btn_style)
        self.save_video_btn.setEnabled(False)

        grid = self.control_group.layout()
        grid.addWidget(self.load_cls_weight_btn, 4, 0)
        grid.addWidget(self.save_video_btn, 4, 1)

        self.setup_connections()

    # ==================== 信号连接 ====================
    def setup_connections(self):
        self.upload_btn.clicked.connect(self.upload_image)
        self.detect_btn.clicked.connect(self.detect_image)
        self.camera_btn.clicked.connect(self.toggle_camera)
        self.detect_camera_btn.clicked.connect(self.toggle_camera_detection)
        self.video_btn.clicked.connect(self.open_video)
        self.detect_video_btn.clicked.connect(self.toggle_video_detection)
        self.load_weight_btn.clicked.connect(self.load_weight)
        self.load_cls_weight_btn.clicked.connect(self.load_cls_weight)
        self.save_image_btn.clicked.connect(self.save_result_image)
        self.save_video_btn.clicked.connect(self.toggle_save_video)

        self.conf_slider.valueChanged.connect(self.update_conf_threshold)
        self.conf_spinbox.valueChanged.connect(self.update_conf_from_spinbox)
        self.iou_slider.valueChanged.connect(self.update_iou_threshold)
        self.iou_spinbox.valueChanged.connect(self.update_iou_from_spinbox)

    # ==================== 参数调节 ====================
    def update_conf_threshold(self, value):
        self.conf_threshold = value / 100.0
        self.conf_spinbox.blockSignals(True)
        self.conf_spinbox.setValue(self.conf_threshold)
        self.conf_spinbox.blockSignals(False)

    def update_conf_from_spinbox(self, value):
        self.conf_threshold = value
        self.conf_slider.blockSignals(True)
        self.conf_slider.setValue(int(value * 100))
        self.conf_slider.blockSignals(False)

    def update_iou_threshold(self, value):
        self.iou_threshold = value / 100.0
        self.iou_spinbox.blockSignals(True)
        self.iou_spinbox.setValue(self.iou_threshold)
        self.iou_spinbox.blockSignals(False)

    def update_iou_from_spinbox(self, value):
        self.iou_threshold = value
        self.iou_slider.blockSignals(True)
        self.iou_slider.setValue(int(value * 100))
        self.iou_slider.blockSignals(False)

    # ==================== 重置轨迹状态 ====================
    def reset_track_state(self):
        self.track_buffers = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))
        self.track_last_seen = {}
        self.track_pred = {}
        self.frame_idx = 0

    # ==================== 绘制骨架 + 标签 ====================
    def draw_pose(self, img, kpt_norm):
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
            if sa > KPT_CONF_THRES and sb > KPT_CONF_THRES:
                cv2.line(img, (xa, ya), (xb, yb), (255, 0, 0), 2)

        for x, y, s in pts:
            if s > KPT_CONF_THRES:
                cv2.circle(img, (x, y), 3, (0, 255, 0), -1)

    def draw_pose_results(self, image, persons, action_results):
        """绘制所有人的骨架 + 检测框 + 动作分类标签"""
        result_img = image.copy()

        for tid, info in persons.items():
            box = info["box"]
            kpt = info["kpt"]

            self.draw_pose(result_img, kpt)

            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 255), 2)

            if action_results:
                if tid in action_results:
                    name, score = action_results[tid]
                    label = f"ID {tid} | {name} {score:.2f}"
                else:
                    label = f"ID {tid} | warming..."
            else:
                det_conf = info["det_conf"]
                label = f"person {det_conf:.2f}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            y_top = max(0, y1 - th - 8)
            cv2.rectangle(result_img, (x1, y_top), (x1 + tw + 6, y1), (0, 0, 0), -1)
            cv2.putText(result_img, label, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return result_img

    # ==================== 图片检测 ====================
    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图像文件 (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.original_image = cv2.imread(file_path)
            if self.original_image is not None:
                self.display_image(self.original_image, self.original_label)
                self.detect_btn.setEnabled(True)
                self.result_label.clear()
                self.result_image = None
                self.save_image_btn.setEnabled(False)
            else:
                print(f"无法读取图片: {file_path}")

    def detect_image(self):
        """
        图片检测流程（仅检测，不做动作分类）：
        1) YOLO Pose 检测 → 提取人物框 + 关键点
        2) 绘制骨架 + 框（图片无时序信息，不进行 ST-GCN 动作分类）
        """
        if self.original_image is None:
            return

        start_time = time.time()
        try:
            img_h, img_w = self.original_image.shape[:2]

            results = self.det_model(
                self.original_image, conf=self.conf_threshold, iou=self.iou_threshold
            )
            result = results[0]

            persons = extract_persons(result, img_w, img_h, use_track_id=False)

            end_time = time.time()
            detection_time = end_time - start_time

            result_image = self.draw_pose_results(
                self.original_image, persons, {}
            )
            self.result_image = result_image.copy()
            self.display_image(result_image, self.result_label)

            self.count_label.setText(str(len(persons)))
            self.time_label.setText(f"{detection_time:.2f} s")
            fps = 1.0 / detection_time if detection_time > 0 else 0.0
            self.fps_label.setText(f"{fps:.2f} FPS")

            self.update_result_table(persons, {}, "图片检测")
            self.save_image_btn.setEnabled(True)
        except Exception as e:
            print(f"检测过程中出错: {str(e)}")

    # ==================== 摄像头检测 ====================
    def toggle_camera(self):
        if self.timer.isActive():
            self.timer.stop()
            if self.camera is not None:
                self.camera.release()
                self.camera = None
            self.camera_btn.setText("打开摄像头")
            self.detect_camera_btn.setEnabled(False)
            self.save_image_btn.setEnabled(False)
            self.stop_recording()

            if self.is_detecting_camera:
                self.toggle_camera_detection()
        else:
            self.camera = cv2.VideoCapture(0)
            if self.camera.isOpened():
                self.reset_track_state()
                self.timer.start(30)
                self.camera_btn.setText("关闭摄像头")
                self.detect_camera_btn.setEnabled(True)
                self.result_label.clear()
                self.result_image = None
            else:
                print("无法打开摄像头")

    def update_camera(self):
        if self.camera is not None and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                self.original_image = frame.copy()
                self.display_image(frame, self.original_label)

    def toggle_camera_detection(self):
        self.is_detecting_camera = not self.is_detecting_camera

        if self.is_detecting_camera:
            self.reset_track_state()
            self.detect_camera_btn.setText("停止检测")
            self.camera_detection_timer.start(100)
            self.save_video_btn.setEnabled(True)
        else:
            self.detect_camera_btn.setText("摄像头检测")
            self.camera_detection_timer.stop()
            self.result_label.clear()
            self.result_image = None
            self.save_image_btn.setEnabled(False)
            self.stop_recording()

    def detect_camera_frame(self):
        """
        摄像头逐帧检测（和视频检测逻辑完全一致）：
        YOLO Pose + ByteTrack 跟踪 → 累积关键点序列 → ST-GCN 动作分类
        """
        if self.original_image is None:
            return

        try:
            start_time = time.time()
            self.frame_idx += 1
            img_h, img_w = self.original_image.shape[:2]

            results = self.det_model.track(
                self.original_image,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            result = results[0]

            persons = extract_persons(result, img_w, img_h, use_track_id=True)

            current_ids = set()
            for tid, info in persons.items():
                current_ids.add(tid)
                self.track_buffers[tid].append(info["kpt"])
                self.track_last_seen[tid] = self.frame_idx

                if (
                    len(self.track_buffers[tid]) == WINDOW_SIZE
                    and self.frame_idx % STRIDE == 0
                    and self.stgcn_model is not None
                ):
                    pred_id, pred_score = classify_action(
                        self.stgcn_model, list(self.track_buffers[tid])
                    )
                    pred_name = (
                        CLASS_NAMES[pred_id]
                        if 0 <= pred_id < len(CLASS_NAMES)
                        else f"class_{pred_id}"
                    )
                    self.track_pred[tid] = (pred_name, pred_score)

            for tid in list(self.track_buffers.keys()):
                if tid not in current_ids and len(self.track_buffers[tid]) > 0:
                    self.track_buffers[tid].append(self.track_buffers[tid][-1])

            to_delete = [
                tid
                for tid, ls in self.track_last_seen.items()
                if self.frame_idx - ls > MAX_MISSING_FRAMES
            ]
            for tid in to_delete:
                self.track_buffers.pop(tid, None)
                self.track_last_seen.pop(tid, None)
                self.track_pred.pop(tid, None)

            action_results = {}
            for tid in persons:
                action_results[tid] = self.track_pred.get(tid, ("warming...", 0.0))

            end_time = time.time()
            detection_time = end_time - start_time

            result_image = self.draw_pose_results(
                self.original_image, persons, action_results
            )
            self.result_image = result_image.copy()
            self.display_image(result_image, self.result_label)

            self.count_label.setText(str(len(persons)))
            self.time_label.setText(f"{detection_time:.2f} s")
            fps = 1.0 / detection_time if detection_time > 0 else 0.0
            self.fps_label.setText(f"{fps:.2f} FPS")

            self.update_result_table(persons, action_results, "摄像头检测")
            self.save_image_btn.setEnabled(True)

            if self.is_recording and self.video_writer is not None:
                self.video_writer.write(self.result_image)
        except Exception as e:
            print(f"摄像头检测过程中出错: {str(e)}")

    # ==================== 视频检测 ====================
    def open_video(self):
        if self.video_timer.isActive():
            self.stop_video()
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)"
        )
        if file_path:
            try:
                self.video_capture = cv2.VideoCapture(file_path)
                if self.video_capture.isOpened():
                    self.video_path = file_path
                    self.video_frame_count = int(
                        self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
                    )
                    self.current_frame = 0
                    self.reset_track_state()

                    ret, frame = self.video_capture.read()
                    if ret:
                        self.original_image = frame.copy()
                        self.display_image(frame, self.original_label)

                        self.video_timer.start(30)
                        self.video_btn.setText("关闭视频")
                        self.detect_video_btn.setEnabled(True)
                        self.result_label.clear()
                        self.result_image = None
                        self.save_image_btn.setEnabled(False)
                    else:
                        print("无法读取视频帧")
                        self.video_capture.release()
                        self.video_capture = None
                else:
                    print(f"无法打开视频: {file_path}")
            except Exception as e:
                print(f"打开视频时出错: {str(e)}")

    def stop_video(self):
        self.video_timer.stop()
        self.stop_recording()

        if self.is_detecting_video:
            self.toggle_video_detection()

        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None

        self.video_btn.setText("打开视频")
        self.detect_video_btn.setEnabled(False)
        self.save_image_btn.setEnabled(False)
        self.result_label.clear()
        self.result_image = None

    def update_video_frame(self):
        if self.video_capture is not None and self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            if ret:
                self.original_image = frame.copy()
                self.display_image(frame, self.original_label)
                self.current_frame += 1

                if self.current_frame >= self.video_frame_count:
                    self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.current_frame = 0
                    self.reset_track_state()
            else:
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.current_frame = 0
                self.reset_track_state()

    def toggle_video_detection(self):
        self.is_detecting_video = not self.is_detecting_video

        if self.is_detecting_video:
            self.reset_track_state()
            self.detect_video_btn.setText("停止检测")
            self.video_detection_timer.start(100)
            self.save_video_btn.setEnabled(True)
        else:
            self.detect_video_btn.setText("视频检测")
            self.video_detection_timer.stop()
            self.result_label.clear()
            self.result_image = None
            self.save_image_btn.setEnabled(False)
            self.stop_recording()

    def detect_video_frame(self):
        """
        视频逐帧检测（完全复刻 yolopose_bytetrack_stgcn_multi.py 主流程）：
        1) YOLO Pose + ByteTrack → 多人跟踪 + 关键点提取
        2) 按 track_id 累积关键点到时序缓冲区
        3) 缓冲区满 WINDOW_SIZE 帧后，每隔 STRIDE 帧调用 ST-GCN 分类
        4) 短时遮挡补帧 + 长时消失清理
        """
        if self.original_image is None:
            return

        try:
            start_time = time.time()
            self.frame_idx += 1
            img_h, img_w = self.original_image.shape[:2]

            results = self.det_model.track(
                self.original_image,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            result = results[0]

            persons = extract_persons(result, img_w, img_h, use_track_id=True)

            # 更新各 track 的时序缓冲区
            current_ids = set()
            for tid, info in persons.items():
                current_ids.add(tid)
                self.track_buffers[tid].append(info["kpt"])
                self.track_last_seen[tid] = self.frame_idx

                if (
                    len(self.track_buffers[tid]) == WINDOW_SIZE
                    and self.frame_idx % STRIDE == 0
                    and self.stgcn_model is not None
                ):
                    pred_id, pred_score = classify_action(
                        self.stgcn_model, list(self.track_buffers[tid])
                    )
                    pred_name = (
                        CLASS_NAMES[pred_id]
                        if 0 <= pred_id < len(CLASS_NAMES)
                        else f"class_{pred_id}"
                    )
                    self.track_pred[tid] = (pred_name, pred_score)

            # 短时遮挡补帧
            for tid in list(self.track_buffers.keys()):
                if tid not in current_ids and len(self.track_buffers[tid]) > 0:
                    self.track_buffers[tid].append(self.track_buffers[tid][-1])

            # 清理长时间消失的 track
            to_delete = [
                tid
                for tid, ls in self.track_last_seen.items()
                if self.frame_idx - ls > MAX_MISSING_FRAMES
            ]
            for tid in to_delete:
                self.track_buffers.pop(tid, None)
                self.track_last_seen.pop(tid, None)
                self.track_pred.pop(tid, None)

            # 构建本帧的动作分类结果
            action_results = {}
            for tid in persons:
                action_results[tid] = self.track_pred.get(tid, ("warming...", 0.0))

            end_time = time.time()
            detection_time = (end_time - start_time) * 1000

            result_image = self.draw_pose_results(
                self.original_image, persons, action_results
            )
            self.result_image = result_image.copy()
            self.display_image(result_image, self.result_label)

            self.count_label.setText(str(len(persons)))
            self.time_label.setText(f"{detection_time:.2f} ms")
            detection_time_s = detection_time / 1000.0
            fps = 1.0 / detection_time_s if detection_time_s > 0 else 0.0
            self.fps_label.setText(f"{fps:.2f} FPS")

            self.update_result_table(persons, action_results, "视频检测")
            self.save_image_btn.setEnabled(True)

            if self.is_recording and self.video_writer is not None:
                self.video_writer.write(self.result_image)
        except Exception as e:
            print(f"视频检测过程中出错: {str(e)}")

    # ==================== 显示 & 表格 ====================
    def display_image(self, cv_img, label):
        if cv_img is None:
            return

        try:
            h, w, ch = cv_img.shape
            bytes_per_line = ch * w
            qt_img = QImage(
                cv_img.data, w, h, bytes_per_line, QImage.Format_RGB888
            ).rgbSwapped()
            label.setPixmap(QPixmap.fromImage(qt_img))
        except Exception as e:
            print(f"显示图像时出错: {str(e)}")

    def update_result_table(self, persons, action_results, detection_type):
        try:
            self.result_table.setRowCount(0)

            for i, (tid, info) in enumerate(persons.items()):
                row = self.result_table.rowCount()
                self.result_table.insertRow(row)

                # 序号
                item_id = QTableWidgetItem(str(i + 1))
                item_id.setTextAlignment(Qt.AlignCenter)
                item_id.setForeground(QColor(0, 0, 0))
                self.result_table.setItem(row, 0, item_id)

                # 检测类别
                item_det = QTableWidgetItem(f"person (ID {tid})")
                item_det.setTextAlignment(Qt.AlignCenter)
                item_det.setForeground(QColor(0, 0, 0))
                self.result_table.setItem(row, 1, item_det)

                # 检测置信度
                det_conf = info["det_conf"]
                item_det_conf = QTableWidgetItem(f"{det_conf:.4f}")
                item_det_conf.setTextAlignment(Qt.AlignCenter)
                item_det_conf.setForeground(QColor(0, 0, 0))
                self.result_table.setItem(row, 2, item_det_conf)

                # 动作分类（ST-GCN 结果）
                if action_results and tid in action_results:
                    name, score = action_results[tid]
                    cls_text = name
                    cls_conf_text = f"{score:.4f}"
                elif action_results:
                    cls_text = "warming..."
                    cls_conf_text = "-"
                else:
                    cls_text = "-"
                    cls_conf_text = "-"
                item_cls = QTableWidgetItem(cls_text)
                item_cls.setTextAlignment(Qt.AlignCenter)
                item_cls.setForeground(QColor(0, 0, 0))
                self.result_table.setItem(row, 3, item_cls)

                # 分类置信度
                item_cls_conf = QTableWidgetItem(cls_conf_text)
                item_cls_conf.setTextAlignment(Qt.AlignCenter)
                item_cls_conf.setForeground(QColor(0, 0, 0))
                self.result_table.setItem(row, 4, item_cls_conf)

                # 位置
                x1, y1, x2, y2 = map(int, info["box"][:4])
                w = x2 - x1
                h = y2 - y1
                position = f"({x1}, {y1}, {x2}, {y2}) [{w}×{h}]"
                item_pos = QTableWidgetItem(position)
                item_pos.setTextAlignment(Qt.AlignCenter)
                item_pos.setForeground(QColor(0, 0, 0))
                self.result_table.setItem(row, 5, item_pos)

                # 检测类型
                item_type = QTableWidgetItem(detection_type)
                item_type.setTextAlignment(Qt.AlignCenter)
                item_type.setForeground(QColor(0, 0, 0))
                self.result_table.setItem(row, 6, item_type)

            if self.result_table.rowCount() == 0:
                self.result_table.insertRow(0)
                no_result_item = QTableWidgetItem("未检测到目标")
                no_result_item.setTextAlignment(Qt.AlignCenter)
                no_result_item.setForeground(QColor(0, 0, 0))
                self.result_table.setSpan(0, 0, 1, 7)
                self.result_table.setItem(0, 0, no_result_item)

            self.result_table.resizeRowsToContents()

        except Exception as e:
            print(f"更新结果表格时出错: {str(e)}")

    # ==================== 权重加载 ====================
    def load_weight(self):
        """加载 YOLO Pose 检测模型权重"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择检测权重文件", "", "权重文件 (*.pt *.pth)"
        )
        if file_path:
            try:
                self.det_model = YOLO(file_path)
                self.det_weight_path = file_path

                weight_name = os.path.basename(file_path)
                self.setWindowTitle(f"动作识别分类系统 - 检测权重: {weight_name}")

                if self.original_image is not None:
                    self.detect_btn.setEnabled(True)
            except Exception as e:
                print(f"加载检测权重文件时出错: {str(e)}")

    def load_cls_weight(self):
        """加载 ST-GCN 动作分类模型权重"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择ST-GCN分类权重文件", "", "权重文件 (*.pt *.pth)"
        )
        if file_path:
            try:
                self.stgcn_model = build_stgcn_model(file_path)
                self.stgcn_weight_path = file_path

                weight_name = os.path.basename(file_path)
                title = self.windowTitle()
                if "分类权重" not in title:
                    self.setWindowTitle(f"{title} | 分类权重: {weight_name}")
                else:
                    base = title.split(" | 分类权重:")[0]
                    self.setWindowTitle(f"{base} | 分类权重: {weight_name}")

                print(f"ST-GCN 分类模型已加载: {file_path}")
            except Exception as e:
                print(f"加载ST-GCN分类权重文件时出错: {str(e)}")

    # ==================== 视频录制 ====================
    def toggle_save_video(self):
        if not self.is_recording:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存检测视频", "", "视频文件 (*.mp4 *.avi)"
            )
            if not file_path:
                return
            if not any(file_path.lower().endswith(ext) for ext in [".mp4", ".avi"]):
                file_path += ".mp4"

            if self.video_capture is not None and self.video_capture.isOpened():
                fps = self.video_capture.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    fps = 25.0
            else:
                fps = 10.0

            if self.result_image is not None:
                h, w = self.result_image.shape[:2]
            elif self.original_image is not None:
                h, w = self.original_image.shape[:2]
            else:
                print("无法确定视频尺寸")
                return

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(file_path, fourcc, fps, (w, h))
            if self.video_writer.isOpened():
                self.save_video_path = file_path
                self.is_recording = True
                self.save_video_btn.setText("停止录制")
                print(f"开始录制视频: {file_path}")
            else:
                print("无法创建视频文件")
                self.video_writer = None
        else:
            self.stop_recording()

    def stop_recording(self):
        if self.is_recording and self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            self.is_recording = False
            self.save_video_btn.setText("保存视频")
            print(f"视频已保存至: {self.save_video_path}")
            self.save_video_path = None
        if not self.is_detecting_video and not self.is_detecting_camera:
            self.save_video_btn.setEnabled(False)

    # ==================== 保存图片 & 关闭 ====================
    def save_result_image(self):
        if self.result_image is not None:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存检测结果", "", "图像文件 (*.png *.jpg *.jpeg *.bmp)"
            )
            if file_path:
                try:
                    if not any(
                        file_path.lower().endswith(ext)
                        for ext in [".png", ".jpg", ".jpeg", ".bmp"]
                    ):
                        file_path += ".jpg"
                    cv2.imwrite(file_path, self.result_image)
                    print(f"检测结果已保存至: {file_path}")
                except Exception as e:
                    print(f"保存图像时出错: {str(e)}")
        else:
            print("没有可保存的检测结果图像")

    def closeEvent(self, event):
        self.stop_recording()
        if self.camera is not None:
            self.camera.release()
        if self.video_capture is not None:
            self.video_capture.release()
        event.accept()

    def paintEvent(self, event):
        if self.background_image:
            painter = QPainter(self)
            painter.drawPixmap(
                QRect(0, 0, self.width(), self.height()), self.background_image
            )
        super().paintEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DetectionApp()
    window.show()
    sys.exit(app.exec_())
