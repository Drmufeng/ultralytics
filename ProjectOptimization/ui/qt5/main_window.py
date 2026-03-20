import sys
from datetime import datetime
from pathlib import Path
import re
import json

import cv2
import yaml
from PyQt5.QtCore import QProcess, Qt
from PyQt5.QtGui import QColor, QImage, QPixmap, QTextCursor
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

from scripts.infer.run_pose_stgcn import (  # noqa: E402
    finalize_config,
    infer_layout_from_stgcn_weights,
    infer_num_class_from_stgcn_weights,
    load_runtime_config,
)
from ui.qt5.services.data_inspector import inspect_stgcn_dataset  # noqa: E402
from ui.qt5.services.inference_session import InferenceSession  # noqa: E402
from ui.qt5.services.model_registry import ModelRegistry  # noqa: E402


class MainWindow(QMainWindow):


    def __init__(self):
        super().__init__()
        self.root = Path(__file__).resolve().parents[2]
        self.python = (self.root.parent / ".venv310" / "Scripts" / "python.exe").resolve()
        self.default_train_profile = self._pick_default_train_profile()
        self.default_test_video_dir = self.root / "src" / "ultralytics-main" / "dataset" / "GCNData" / "BaseballPitch"
        self.default_infer_output_dir = self.root / "outputs" / "infer"
        self.default_ablation_dir = self.root / "outputs" / "eval" / "ablation"

        self.process = None
        self.current_task = "idle"

        self.model_registry = ModelRegistry(self.root)
        self.infer_session = InferenceSession(
            parent=self,
            on_log=lambda text: self._log(text, "INFO"),
            on_frame=self._display_frame_in_label,
            on_table=self._update_infer_table,
            on_state=self._on_infer_state_changed,
        )

        self.log_file = self.root / "logs" / "run_logs" / "ui_app.log"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.ui_state_file = self.root / "configs" / "app" / "ui_state.json"
        self.ui_state_file.parent.mkdir(parents=True, exist_ok=True)
        self.ui_state = self._load_ui_state()
        self.detail_log_window = None
        self.detail_log_text = None
        self.detail_log_buffer = []

        self.setWindowTitle("训练控制台")
        self.resize(1280, 860)
        self._build_ui()
        self._on_ablation_theme_changed()
        self._refresh_infer_path_labels()
        self._on_train_profile_changed()
        self.refresh_model_lists()
        self._set_status_card(self.env_status_value, "待检测", "#f39c12")

    def _pick_default_train_profile(self) -> str:
        out17_cfg = self.root / "configs" / "train" / "train_stgcn_out17.yaml"
        out17_data = self.root / "data" / "stgcn_dataset_out17" / "stgcn_dataset_out"
        if out17_cfg.exists() and out17_data.exists():
            return "v17"
        return "v18"

    def _load_ui_state(self) -> dict:
        if not self.ui_state_file.exists():
            return {}
        try:
            return json.loads(self.ui_state_file.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_ui_state(self):
        try:
            self.ui_state_file.write_text(json.dumps(self.ui_state, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            self._log(f"保存UI状态失败: {e}", "WARN")

    def _update_recent_list(self, key: str, value: str, limit: int = 12):
        v = (value or "").strip()
        if not v:
            return
        lst = [x for x in self.ui_state.get(key, []) if x != v]
        lst.insert(0, v)
        self.ui_state[key] = lst[:limit]
        self._save_ui_state()

    def _scan_yaml_options(self, base_dir: Path) -> list[str]:
        if not base_dir.exists():
            return []
        out = []
        for p in sorted(base_dir.rglob("*.yaml")) + sorted(base_dir.rglob("*.yml")):
            try:
                out.append(str(p.resolve()))
            except Exception:
                pass
        # 去重保持顺序
        seen = set()
        uniq = []
        for x in out:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        return uniq

    def _classify_model_yaml(self, p: Path) -> str:
        s = p.name.lower()
        if "pose" in s:
            return "POSE"
        if "-cls" in s or "cls" in s:
            return "CLS"
        if "seg" in s:
            return "SEG"
        if "obb" in s:
            return "OBB"
        if "detect" in s or "yolo" in s:
            base = "DETECT"
        else:
            base = "OTHER"

        # 回退：文件名不明显时，尝试按内容判断（便于外部目录自定义命名）
        if p.exists():
            kpt = self._read_kpt_count_from_yaml(p)
            if kpt is not None and kpt > 0:
                return "POSE"
        return base

    def _classify_data_yaml(self, p: Path) -> str:
        s = p.name.lower()
        if "pose" in s or "keypoint" in s:
            return "POSE"
        if "cls" in s or "class" in s:
            return "CLS"
        if "seg" in s:
            return "SEG"
        if "obb" in s:
            return "OBB"

        # 回退：按内容判断，避免外部数据yaml命名不含pose时误判
        if p.exists():
            kpt = self._read_kpt_count_from_yaml(p)
            if kpt is not None and kpt > 0:
                return "POSE"
        return "DETECT"

    def _read_kpt_count_from_yaml(self, p: Path) -> int | None:
        try:
            obj = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            kpt = obj.get("kpt_shape", None)
            if isinstance(kpt, (list, tuple)) and len(kpt) >= 1:
                return int(kpt[0])
        except Exception:
            return None
        return None

    def _scan_model_yaml_grouped(self) -> list[tuple[str, str, str]]:
        base = self.root / "src" / "ultralytics-main" / "ultralytics" / "cfg" / "models"
        paths = [Path(x) for x in self._scan_yaml_options(base)]
        order = {"POSE": 0, "DETECT": 1, "CLS": 2, "SEG": 3, "OBB": 4, "OTHER": 5}
        rows = []
        for p in paths:
            cat = self._classify_model_yaml(p)
            rows.append((cat, str(p.resolve())))
        rows.sort(key=lambda x: (order.get(x[0], 99), x[1].lower()))
        return [(f"[{cat}] {path}", path, cat) for cat, path in rows]

    def _scan_data_yaml_grouped(self) -> list[tuple[str, str, str]]:
        base = self.root / "src" / "ultralytics-main" / "ultralytics" / "cfg" / "datasets"
        paths = [Path(x) for x in self._scan_yaml_options(base)]
        order = {"POSE": 0, "DETECT": 1, "CLS": 2, "SEG": 3, "OBB": 4}
        rows = []
        for p in paths:
            cat = self._classify_data_yaml(p)
            rows.append((cat, str(p.resolve())))
        rows.sort(key=lambda x: (order.get(x[0], 99), x[1].lower()))
        return [(f"[{cat}] {path}", path, cat) for cat, path in rows]

    def _filter_recent_model_paths(self, paths: list[str], allowed_categories: set[str]) -> list[str]:
        out = []
        for p in paths:
            try:
                cat = self._classify_model_yaml(Path(p))
            except Exception:
                continue
            if cat in allowed_categories:
                out.append(p)
        return out

    def _filter_recent_data_paths(self, paths: list[str], allowed_categories: set[str]) -> list[str]:
        out = []
        for p in paths:
            try:
                cat = self._classify_data_yaml(Path(p))
            except Exception:
                continue
            if cat in allowed_categories:
                out.append(p)
        return out

    def _extract_combo_path(self, text: str) -> str:
        t = (text or "").strip()
        while t.startswith("[") and "] " in t:
            t = t.split("] ", 1)[1].strip()
        return t

    def _fill_editable_combo(self, combo: QComboBox, values: list[str], current: str = ""):
        combo.clear()
        for v in values:
            combo.addItem(v)
        if current:
            combo.setCurrentText(current)

    def _fill_grouped_editable_combo(
        self,
        combo: QComboBox,
        grouped_rows: list[tuple[str, str, str]],
        recent_paths: list[str],
        current_path: str = "",
    ):
        combo.clear()
        path_to_display = {}
        for display, path, _ in grouped_rows:
            path_to_display[path] = display

        # 最近使用放最前，但保持分组标签显示
        for p in recent_paths:
            if p in path_to_display:
                combo.addItem("[RECENT] " + path_to_display[p])

        for display, path, _ in grouped_rows:
            combo.addItem(display)

        if current_path:
            target = path_to_display.get(current_path)
            if target:
                idx = combo.findText(target)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
                    return
            combo.setCurrentText(current_path)

    def _train_profile_settings(self, key: str) -> dict:
        if key == "v17":
            return {
                "config": "configs/train/train_stgcn_out17.yaml",
                "layout": "yolopose",
                "data_rel": "data/stgcn_dataset_out17/stgcn_dataset_out",
            }
        if key == "v18":
            return {
                "config": "configs/train/train_stgcn_out3.yaml",
                "layout": "openpose",
                "data_rel": "data/stgcn_dataset_out3",
            }

        preferred = self.default_train_profile
        return self._train_profile_settings(preferred)

    def _on_train_profile_changed(self):
        setting = self._train_profile_settings(self.train_profile_combo.currentData())
        self.train_config_line.setText(setting["config"])
        self.train_layout_value.setText(setting["layout"])
        self.data_root_label.setText(setting["data_rel"])
        self._sync_train_init_weight_by_profile(setting["layout"])
        self.refresh_data_status()
        self._refresh_train_init_policy_hint()

    def _current_train_target_num_class(self) -> int | None:
        try:
            config_path = self._resolve_user_path(self.train_config_line.text().strip()).resolve()
            if not config_path.exists():
                return None
            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            val = cfg.get("model_args", {}).get("num_class", None)
            return int(val) if val is not None else None
        except Exception:
            return None

    def _set_train_hint(self, text: str, color: str):
        self.train_init_policy_label.setText(text)
        self.train_init_policy_label.setStyleSheet(
            "padding:2px 8px;border-radius:8px;font-weight:600;"
            f"color:{color};background:#f8fafc;border:1px solid #e5e7eb;"
        )

    def _refresh_train_init_policy_hint(self):
        mode = self.train_mode_combo.currentText().strip()
        layout = self.train_layout_value.text().strip()
        init_weight = self.train_init_weight_combo.currentData()
        target_num_class = self._current_train_target_num_class()

        if not init_weight:
            if mode == "预训练初始化":
                self._set_train_hint("从头训练：不加载初始化权重。", "#0f766e")
            else:
                self._set_train_hint("微调训练需要选择已有权重。", "#b45309")
            return

        inferred_layout = infer_layout_from_stgcn_weights(str(init_weight))
        inferred_num_class = infer_num_class_from_stgcn_weights(str(init_weight))
        if inferred_layout in {"yolopose", "openpose"} and inferred_layout != layout:
            self._set_train_hint(
                f"布局不匹配：weights={inferred_layout}, target={layout}，无法启动。",
                "#b91c1c",
            )
            return

        if mode == "预训练初始化":
            if (
                target_num_class is not None
                and inferred_num_class is not None
                and int(inferred_num_class) == int(target_num_class)
            ):
                self._set_train_hint("将保留分类头，继续预训练。", "#0f766e")
            else:
                self._set_train_hint("将自动忽略fcn分类头（迁移初始化模式）。", "#92400e")
            return

        if target_num_class is not None and inferred_num_class is not None and int(inferred_num_class) != int(target_num_class):
            self._set_train_hint(
                f"类别数不一致：weights={inferred_num_class}, target={target_num_class}，可能报错。",
                "#b91c1c",
            )
            return
        self._set_train_hint("将加载完整权重，继续微调训练。", "#0f766e")

    def _sync_train_init_weight_by_profile(self, target_layout: str):
        if self.train_init_weight_combo.count() <= 0:
            return

        compatible_idx = -1
        fallback_idx = -1
        for i in range(self.train_init_weight_combo.count()):
            path = self.train_init_weight_combo.itemData(i)
            if not path:
                continue
            inferred = infer_layout_from_stgcn_weights(str(path))
            if inferred == target_layout:
                compatible_idx = i
                break
            if fallback_idx < 0:
                fallback_idx = i

        idx = compatible_idx if compatible_idx >= 0 else (fallback_idx if fallback_idx >= 0 else 0)
        self.train_init_weight_combo.setCurrentIndex(idx)

    def _resolve_user_path(self, value: str) -> Path:
        """将用户输入路径解析为绝对路径：支持相对路径与绝对路径。"""
        p = Path(value)
        return p if p.is_absolute() else (self.root / p)

    # ----------------------------- UI 构建 -----------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(10)

        title = QLabel("项目控制台（训练 / 测试 / 推理 / 消融）")
        title.setStyleSheet("font-size:22px;font-weight:700;color:#1f2937;padding:4px 0;")
        root.addWidget(title)

        status_box = self._build_status_cards()
        root.addWidget(status_box)

        main_split = QHBoxLayout()
        main_split.setSpacing(10)
        root.addLayout(main_split, 1)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(self._build_tab_env_data(), "环境与数据")
        self.tabs.addTab(self._build_tab_train(), "训练")
        self.tabs.addTab(self._build_tab_test(), "测试")
        self.tabs.addTab(self._build_tab_infer(), "推理")
        self.tabs.addTab(self._build_tab_ablation(), "消融")
        main_split.addWidget(self.tabs, 3)

        log_panel = self._build_log_panel()
        main_split.addWidget(log_panel, 2)

        self._connect_signals()

    def _build_status_cards(self):
        box = QGroupBox("状态概览")
        grid = QGridLayout(box)
        grid.setHorizontalSpacing(18)

        self.env_status_value = QLabel()
        self.data_status_value = QLabel()
        self.model_status_value = QLabel()
        self.task_status_value = QLabel()
        self.show_all_model_versions_ck = QCheckBox("显示全部模型版本")
        self.show_all_model_versions_ck.setChecked(False)

        self._set_status_card(self.data_status_value, "待检测", "#f39c12")
        self._set_status_card(self.model_status_value, "待刷新", "#f39c12")
        self._set_status_card(self.task_status_value, "空闲", "#6b7280")

        grid.addWidget(QLabel("环境状态"), 0, 0)
        grid.addWidget(self.env_status_value, 0, 1)
        grid.addWidget(QLabel("数据状态"), 0, 2)
        grid.addWidget(self.data_status_value, 0, 3)
        grid.addWidget(QLabel("模型状态"), 1, 0)
        grid.addWidget(self.model_status_value, 1, 1)
        grid.addWidget(QLabel("任务状态"), 1, 2)
        grid.addWidget(self.task_status_value, 1, 3)
        grid.addWidget(self.show_all_model_versions_ck, 2, 0, 1, 2)
        return box

    def _build_tab_env_data(self):
        page = QWidget()
        v = QVBoxLayout(page)

        env_box = QGroupBox("环境管理")
        env_form = QFormLayout(env_box)
        self.python_path_label = QLabel(str(self.python))
        self.python_path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.btn_env_check = QPushButton("环境检测（仅检查）")
        self.btn_env_fix = QPushButton("环境检测并自动修复")
        env_form.addRow("Python解释器", self.python_path_label)
        env_form.addRow(self.btn_env_check, self.btn_env_fix)

        data_box = QGroupBox("数据状态")
        data_form = QFormLayout(data_box)
        self.data_root_label = QLabel("data/stgcn_dataset_out3")
        self.data_root_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.data_detail_label = QLabel("未检查")
        self.btn_refresh_data = QPushButton("刷新数据状态")
        data_form.addRow("数据目录", self.data_root_label)
        data_form.addRow("检查结果", self.data_detail_label)
        data_form.addRow(self.btn_refresh_data)

        v.addWidget(env_box)
        v.addWidget(data_box)
        v.addStretch(1)
        return page

    def _build_tab_train(self):
        page = QWidget()
        v = QVBoxLayout(page)

        box = QGroupBox("训练控制")
        form = QFormLayout(box)

        self.train_mode_combo = QComboBox()
        self.train_mode_combo.addItems(["预训练初始化", "微调训练"])
        self.train_init_weight_combo = QComboBox()

        self.train_profile_combo = QComboBox()
        self.train_profile_combo.addItem("17关键点（yolopose）", "v17")
        self.train_profile_combo.addItem("18关键点（openpose）", "v18")
        profile_idx = self.train_profile_combo.findData(self.default_train_profile)
        if profile_idx >= 0:
            self.train_profile_combo.setCurrentIndex(profile_idx)

        self.train_layout_value = QLabel("-")
        self.train_init_policy_label = QLabel("-")
        self.train_config_line = QLineEdit()
        self.train_config_line.setReadOnly(True)
        self.btn_train_start = QPushButton("开始训练")
        self.btn_train_stop = QPushButton("停止训练")
        self.btn_train_stop.setEnabled(False)

        action_row = QWidget()
        h = QHBoxLayout(action_row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(self.btn_train_start)
        h.addWidget(self.btn_train_stop)

        form.addRow("训练模式", self.train_mode_combo)
        form.addRow("初始化模型（默认best）", self.train_init_weight_combo)
        form.addRow("训练方案", self.train_profile_combo)
        form.addRow("关键点布局", self.train_layout_value)
        form.addRow("初始化策略", self.train_init_policy_label)
        form.addRow("训练配置", self.train_config_line)
        form.addRow(action_row)

        v.addWidget(box)
        v.addStretch(1)
        return page

    def _build_tab_test(self):
        page = QWidget()
        v = QVBoxLayout(page)

        box = QGroupBox("模型测试（快速健康检查）")
        form = QFormLayout(box)
        self.test_yolo_combo = QComboBox()
        self.test_stgcn_combo = QComboBox()

        self.test_layout_combo = QComboBox()
        self.test_layout_combo.addItem("自动（按模型）", "auto")
        self.test_layout_combo.addItem("17关键点（yolopose）", "yolopose")
        self.test_layout_combo.addItem("18关键点（openpose）", "openpose")

        self.btn_test_run = QPushButton("开始测试")
        form.addRow("YOLO模型", self.test_yolo_combo)
        form.addRow("ST-GCN模型", self.test_stgcn_combo)
        form.addRow("关键点布局", self.test_layout_combo)
        form.addRow(self.btn_test_run)

        v.addWidget(box)
        v.addStretch(1)
        return page

    def _build_tab_infer(self):
        page = QWidget()
        v = QVBoxLayout(page)

        box = QGroupBox("推理控制")
        form = QFormLayout(box)
        self.infer_yolo_combo = QComboBox()
        self.infer_stgcn_combo = QComboBox()

        self.infer_layout_combo = QComboBox()
        self.infer_layout_combo.addItem("自动（按模型）", "auto")
        self.infer_layout_combo.addItem("17关键点（yolopose）", "yolopose")
        self.infer_layout_combo.addItem("18关键点（openpose）", "openpose")

        self.infer_video_line = QLineEdit()
        self.infer_video_line.setPlaceholderText("不填则使用配置文件默认视频")
        self.btn_choose_video = QPushButton("选择视频")
        self.btn_open_test_video_dir = QPushButton("打开测试视频目录")
        self.btn_open_infer_output_dir = QPushButton("打开推理输出目录")
        self.infer_show_ck = QCheckBox("显示推理窗口")
        self.infer_show_ck.setChecked(True)
        self.btn_infer_start = QPushButton("开始推理")
        self.btn_infer_stop = QPushButton("停止推理")
        self.btn_infer_stop.setEnabled(False)
        self.infer_test_video_dir_label = QLabel("-")
        self.infer_test_video_dir_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.infer_output_dir_label = QLabel("-")
        self.infer_output_dir_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        video_row = QWidget()
        hv = QHBoxLayout(video_row)
        hv.setContentsMargins(0, 0, 0, 0)
        hv.addWidget(self.infer_video_line, 1)
        hv.addWidget(self.btn_choose_video)

        action_row = QWidget()
        ha = QHBoxLayout(action_row)
        ha.setContentsMargins(0, 0, 0, 0)
        ha.addWidget(self.btn_infer_start)
        ha.addWidget(self.btn_infer_stop)

        form.addRow("YOLO模型（默认best）", self.infer_yolo_combo)
        form.addRow("ST-GCN模型（默认best）", self.infer_stgcn_combo)
        form.addRow("关键点布局", self.infer_layout_combo)
        form.addRow("输入视频", video_row)
        form.addRow("测试视频目录", self.infer_test_video_dir_label)
        form.addRow(self.btn_open_test_video_dir)
        form.addRow("推理输出目录", self.infer_output_dir_label)
        form.addRow(self.btn_open_infer_output_dir)
        form.addRow("显示设置", self.infer_show_ck)
        form.addRow(action_row)

        self.infer_frame_label = QLabel("等待推理启动...")
        self.infer_frame_label.setMinimumSize(640, 360)
        self.infer_frame_label.setStyleSheet("background:#111827;color:#e5e7eb;border:1px solid #374151;")
        self.infer_frame_label.setAlignment(Qt.AlignCenter)

        self.infer_table = QTableWidget()
        self.infer_table.setColumnCount(5)
        self.infer_table.setHorizontalHeaderLabels(["ID", "检测置信度", "动作标签", "动作置信度", "位置"])
        self.infer_table.setMinimumWidth(520)
        header = self.infer_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        self.infer_table.verticalHeader().setVisible(False)

        content = QHBoxLayout()
        content.addWidget(self.infer_frame_label, 2)
        content.addWidget(self.infer_table, 1)

        v.addWidget(box)
        v.addLayout(content, 1)
        v.addStretch(1)
        return page

    def _build_tab_ablation(self):
        page = QWidget()
        v = QVBoxLayout(page)

        box = QGroupBox("消融实验（规范模板）")
        form = QFormLayout(box)

        self.ablation_theme_combo = QComboBox()
        self.ablation_theme_combo.addItem("YOLO模块消融（ADown/DyRFAUp/SCSA）", "yolo_modules")

        self.ablation_profile_combo = QComboBox()
        self.ablation_profile_combo.addItem("17关键点（yolopose）", "v17")
        self.ablation_profile_combo.addItem("18关键点（openpose）", "v18")
        idx = self.ablation_profile_combo.findData(self.default_train_profile)
        if idx >= 0:
            self.ablation_profile_combo.setCurrentIndex(idx)

        self.ablation_out_combo = QComboBox()
        self.ablation_out_combo.setEditable(True)
        out_recent = self.ui_state.get("recent_ablation_dirs", [])
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_out = str((self.default_ablation_dir / ts).resolve())
        self._fill_editable_combo(self.ablation_out_combo, out_recent, default_out)

        self.btn_ablation_choose_dir = QPushButton("选择输出目录")
        self.btn_ablation_generate = QPushButton("生成规范模板")
        self.btn_ablation_run_batch = QPushButton("开始批跑消融")
        self.btn_ablation_stop = QPushButton("停止批跑/任务")
        self.btn_ablation_stop.setEnabled(False)
        self.ablation_continue_on_error_ck = QCheckBox("失败后继续后续实验")
        self.ablation_continue_on_error_ck.setChecked(True)
        self.ablation_dry_run_ck = QCheckBox("仅演练（不实际执行）")
        self.ablation_dry_run_ck.setChecked(False)
        self.btn_ablation_open_dir = QPushButton("打开消融目录")
        self.ablation_hint_label = QLabel("将生成: ablation_plan.md / ablation_plan.csv / run_commands.txt")
        self.ablation_compat_label = QLabel("-")

        self.ablation_model_yaml_combo = QComboBox()
        self.ablation_model_yaml_combo.setEditable(True)
        model_scan = self._scan_model_yaml_grouped()
        model_recent = self.ui_state.get("recent_ablation_model_yaml", [])
        self._fill_grouped_editable_combo(self.ablation_model_yaml_combo, model_scan, model_recent, "")
        self.btn_ablation_choose_model_yaml = QPushButton("选择模型YAML")

        self.ablation_data_yaml_combo = QComboBox()
        self.ablation_data_yaml_combo.setEditable(True)
        data_scan = self._scan_data_yaml_grouped()
        data_recent = self.ui_state.get("recent_ablation_data_yaml", [])
        self._fill_grouped_editable_combo(self.ablation_data_yaml_combo, data_scan, data_recent, "")
        self.btn_ablation_choose_data_yaml = QPushButton("选择数据YAML")

        self.ablation_epochs_spin = QSpinBox()
        self.ablation_epochs_spin.setRange(0, 10000)
        self.ablation_epochs_spin.setValue(100)
        self.ablation_epochs_spin.setToolTip("0 表示生成占位 <EPOCHS>")
        self.ablation_device_line = QLineEdit()
        self.ablation_device_line.setPlaceholderText("0 / 0,1 / cpu")
        self.ablation_device_line.setText("0")
        self.ablation_device_line.setToolTip("留空表示不写入 device 参数；默认建议 0")

        row1 = QWidget()
        h1 = QHBoxLayout(row1)
        h1.setContentsMargins(0, 0, 0, 0)
        h1.addWidget(self.ablation_out_combo, 1)
        h1.addWidget(self.btn_ablation_choose_dir)

        row2 = QWidget()
        h2 = QHBoxLayout(row2)
        h2.setContentsMargins(0, 0, 0, 0)
        h2.addWidget(self.btn_ablation_generate)
        h2.addWidget(self.btn_ablation_run_batch)
        h2.addWidget(self.btn_ablation_stop)
        h2.addWidget(self.btn_ablation_open_dir)

        row3 = QWidget()
        h3 = QHBoxLayout(row3)
        h3.setContentsMargins(0, 0, 0, 0)
        h3.addWidget(self.ablation_continue_on_error_ck)
        h3.addWidget(self.ablation_dry_run_ck)

        row_model = QWidget()
        hm = QHBoxLayout(row_model)
        hm.setContentsMargins(0, 0, 0, 0)
        hm.addWidget(self.ablation_model_yaml_combo, 1)
        hm.addWidget(self.btn_ablation_choose_model_yaml)

        row_data = QWidget()
        hd = QHBoxLayout(row_data)
        hd.setContentsMargins(0, 0, 0, 0)
        hd.addWidget(self.ablation_data_yaml_combo, 1)
        hd.addWidget(self.btn_ablation_choose_data_yaml)

        form.addRow("实验主题", self.ablation_theme_combo)
        form.addRow("数据方案", self.ablation_profile_combo)
        form.addRow("模型YAML", row_model)
        form.addRow("数据YAML", row_data)
        form.addRow("训练轮次(epochs)", self.ablation_epochs_spin)
        form.addRow("训练设备(device)", self.ablation_device_line)
        form.addRow("输出目录", row1)
        form.addRow("说明", self.ablation_hint_label)
        form.addRow("兼容性", self.ablation_compat_label)
        form.addRow("执行选项", row3)
        form.addRow(row2)

        v.addWidget(box)
        v.addStretch(1)
        return page

    def _build_log_panel(self):
        box = QGroupBox("日志观测")
        v = QVBoxLayout(box)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setStyleSheet("font-family:Consolas,Monaco,monospace;font-size:12px;")
        self.log_view.document().setMaximumBlockCount(600)
        self.btn_clear_log = QPushButton("清空日志窗口")
        self.btn_open_log = QPushButton("打开日志文件夹")
        self.btn_open_detail_log = QPushButton("详细日志窗口")

        row = QHBoxLayout()
        row.addWidget(self.btn_clear_log)
        row.addWidget(self.btn_open_detail_log)
        row.addWidget(self.btn_open_log)

        v.addWidget(self.log_view, 1)
        v.addLayout(row)
        return box

    def _connect_signals(self):
        self.btn_env_check.clicked.connect(lambda: self.run_env_check(False))
        self.btn_env_fix.clicked.connect(lambda: self.run_env_check(True))
        self.btn_refresh_data.clicked.connect(self.refresh_data_status)
        self.btn_clear_log.clicked.connect(self.log_view.clear)
        self.btn_open_detail_log.clicked.connect(self.open_detail_log_window)
        self.btn_open_log.clicked.connect(self.open_log_folder)
        self.show_all_model_versions_ck.stateChanged.connect(lambda _: self.refresh_model_lists())

        self.btn_train_start.clicked.connect(self.run_train)
        self.btn_train_stop.clicked.connect(self.stop_process)
        self.train_profile_combo.currentIndexChanged.connect(lambda _: self._on_train_profile_changed())
        self.train_mode_combo.currentIndexChanged.connect(lambda _: self._refresh_train_init_policy_hint())
        self.train_init_weight_combo.currentIndexChanged.connect(lambda _: self._refresh_train_init_policy_hint())
        self.btn_test_run.clicked.connect(self.run_model_test)
        self.btn_infer_start.clicked.connect(self.run_infer)
        self.btn_infer_stop.clicked.connect(self.stop_process)
        self.btn_choose_video.clicked.connect(self.choose_video)
        self.btn_open_test_video_dir.clicked.connect(self.open_test_video_dir)
        self.btn_open_infer_output_dir.clicked.connect(self.open_infer_output_dir)
        self.btn_ablation_choose_dir.clicked.connect(self.choose_ablation_dir)
        self.ablation_theme_combo.currentIndexChanged.connect(lambda _: self._on_ablation_theme_changed())
        self.btn_ablation_choose_model_yaml.clicked.connect(self.choose_ablation_model_yaml)
        self.btn_ablation_choose_data_yaml.clicked.connect(self.choose_ablation_data_yaml)
        self.ablation_model_yaml_combo.currentIndexChanged.connect(lambda _: self._refresh_ablation_compatibility_status())
        self.ablation_data_yaml_combo.currentIndexChanged.connect(lambda _: self._refresh_ablation_compatibility_status())
        self.ablation_out_combo.currentIndexChanged.connect(lambda _: self._refresh_ablation_compatibility_status())
        self.ablation_theme_combo.currentIndexChanged.connect(lambda _: self._refresh_ablation_compatibility_status())
        self.btn_ablation_generate.clicked.connect(self.run_ablation_plan)
        self.btn_ablation_run_batch.clicked.connect(self.run_ablation_batch)
        self.btn_ablation_stop.clicked.connect(self.stop_process)
        self.btn_ablation_open_dir.clicked.connect(self.open_ablation_dir)

    def _set_ablation_compat(self, ok: bool, text: str):
        color = "#0f766e" if ok else "#b91c1c"
        self.ablation_compat_label.setText(text)
        self.ablation_compat_label.setStyleSheet(
            "padding:2px 8px;border-radius:8px;font-weight:600;"
            f"color:{color};background:#f8fafc;border:1px solid #e5e7eb;"
        )
        self.btn_ablation_generate.setEnabled(ok)
        # 批跑还依赖计划文件存在，先按兼容性控制基本可用性
        self.btn_ablation_run_batch.setEnabled(ok)

    def _refresh_ablation_compatibility_status(self):
        theme = self.ablation_theme_combo.currentData() or "yolo_modules"
        out_dir = self.ablation_out_combo.currentText().strip()
        model_yaml = self._extract_combo_path(self.ablation_model_yaml_combo.currentText().strip())
        data_yaml = self._extract_combo_path(self.ablation_data_yaml_combo.currentText().strip())

        if not out_dir:
            self._set_ablation_compat(False, "输出目录为空，请先设置。")
            return False

        # 允许先不填 model/data 生成占位模板
        if not model_yaml or not data_yaml:
            self._set_ablation_compat(True, "可生成占位模板；建议选择 Pose 模型与 Pose 数据。")
            return True

        if not Path(model_yaml).exists():
            self._set_ablation_compat(False, f"模型YAML不存在: {model_yaml}")
            return False
        if not Path(data_yaml).exists():
            self._set_ablation_compat(False, f"数据YAML不存在: {data_yaml}")
            return False

        if theme == "yolo_modules":
            mcat = self._classify_model_yaml(Path(model_yaml))
            dcat = self._classify_data_yaml(Path(data_yaml))
            if mcat != dcat:
                self._set_ablation_compat(False, f"不兼容：模型类型={mcat}，数据类型={dcat}，需一致。")
                return False

            if mcat == "POSE":
                mk = self._read_kpt_count_from_yaml(Path(model_yaml))
                dk = self._read_kpt_count_from_yaml(Path(data_yaml))
                if mk is not None and dk is not None and mk != dk:
                    self._set_ablation_compat(False, f"关键点数不一致：模型={mk}，数据={dk}。")
                    return False

                profile = self.ablation_profile_combo.currentData() or "v17"
                expected = 17 if profile == "v17" else 18
                if dk is not None and dk != expected:
                    self._set_ablation_compat(False, f"数据方案为{profile}，但数据kpt_shape={dk}，应为{expected}。")
                    return False

        self._set_ablation_compat(True, "组合兼容，可生成模板并批跑。")
        return True

    def choose_ablation_dir(self):
        fp = QFileDialog.getExistingDirectory(self, "选择消融输出目录", str(self.default_ablation_dir))
        if fp:
            self.ablation_out_combo.setCurrentText(str(Path(fp).resolve()))

    def choose_ablation_model_yaml(self):
        fp, _ = QFileDialog.getOpenFileName(self, "选择YOLO模型YAML", str(self.root), "YAML Files (*.yaml *.yml)")
        if fp:
            self.ablation_model_yaml_combo.setCurrentText(str(Path(fp).resolve()))

    def _on_ablation_theme_changed(self):
        theme = self.ablation_theme_combo.currentData()
        if theme == "yolo_modules":
            model_scan = self._scan_model_yaml_grouped()
            recent = self.ui_state.get("recent_ablation_model_yaml", [])
            current = self._extract_combo_path(self.ablation_model_yaml_combo.currentText().strip())
            self._fill_grouped_editable_combo(
                self.ablation_model_yaml_combo,
                model_scan,
                recent,
                current,
            )

            data_scan = self._scan_data_yaml_grouped()
            d_recent = self.ui_state.get("recent_ablation_data_yaml", [])
            d_current = self._extract_combo_path(self.ablation_data_yaml_combo.currentText().strip())
            self._fill_grouped_editable_combo(
                self.ablation_data_yaml_combo,
                data_scan,
                d_recent,
                d_current,
            )
        self._refresh_ablation_compatibility_status()

    def choose_ablation_data_yaml(self):
        fp, _ = QFileDialog.getOpenFileName(self, "选择数据YAML", str(self.root), "YAML Files (*.yaml *.yml)")
        if fp:
            self.ablation_data_yaml_combo.setCurrentText(str(Path(fp).resolve()))

    def open_ablation_dir(self):
        path_text = self.ablation_out_combo.currentText().strip()
        path = Path(path_text).resolve() if path_text else self.default_ablation_dir
        path.mkdir(parents=True, exist_ok=True)
        try:
            import os

            os.startfile(str(path))
        except Exception as e:
            self._log(f"打开消融目录失败: {e}", "ERROR")

    def _refresh_infer_path_labels(self, output_dir: Path | None = None):
        self.infer_test_video_dir_label.setText(str(self.default_test_video_dir))
        out_dir = output_dir if output_dir is not None else self.default_infer_output_dir
        self.infer_output_dir_label.setText(str(out_dir))

    def open_test_video_dir(self):
        path = self.default_test_video_dir
        path.mkdir(parents=True, exist_ok=True)
        try:
            import os

            os.startfile(str(path))
        except Exception as e:
            self._log(f"打开测试视频目录失败: {e}", "ERROR")

    def open_infer_output_dir(self):
        path = Path(self.infer_output_dir_label.text().strip()) if self.infer_output_dir_label.text().strip() else self.default_infer_output_dir
        path.mkdir(parents=True, exist_ok=True)
        try:
            import os

            os.startfile(str(path))
        except Exception as e:
            self._log(f"打开推理输出目录失败: {e}", "ERROR")

    # ----------------------------- 状态与日志 -----------------------------
    def _set_status_card(self, widget: QLabel, text: str, color: str):
        widget.setText(text)
        widget.setStyleSheet(
            f"background:{color};color:white;padding:4px 10px;border-radius:10px;font-weight:600;"
        )

    def _is_compact_log_line(self, text: str, level: str) -> bool:
        if level in {"WARN", "ERROR"}:
            return True
        keywords = [
            "启动命令",
            "任务结束",
            "Training epoch",
            "Iter",
            "Environment check passed",
            "训练layout",
            "推理布局自动推断",
            "推理类别数自动推断",
            "内嵌推理启动",
            "用户手动停止",
            "模型列表刷新完成",
            "配置检查失败",
        ]
        return any(k in text for k in keywords)

    def open_detail_log_window(self):
        if self.detail_log_window is None:
            w = QWidget(None)
            w.setWindowFlag(Qt.Window, True)
            w.setAttribute(Qt.WA_DeleteOnClose, True)
            w.setWindowTitle("详细日志")
            w.resize(980, 700)
            layout = QVBoxLayout(w)
            text = QTextEdit()
            text.setReadOnly(True)
            text.setStyleSheet("font-family:Consolas,Monaco,monospace;font-size:12px;")
            layout.addWidget(text)
            self.detail_log_window = w
            self.detail_log_text = text
            w.destroyed.connect(self._on_detail_log_window_closed)
        if self.detail_log_text is not None:
            self.detail_log_text.setPlainText("\n".join(self.detail_log_buffer))
            self.detail_log_text.moveCursor(QTextCursor.End)
        if self.detail_log_window is not None:
            self.detail_log_window.show()
            self.detail_log_window.raise_()
            self.detail_log_window.activateWindow()

    def _on_detail_log_window_closed(self):
        self.detail_log_window = None
        self.detail_log_text = None

    def _log(self, text: str, level: str = "INFO"):
        text = ANSI_ESCAPE_RE.sub("", text)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] [{level}] {text}"

        self.detail_log_buffer.append(line)
        if len(self.detail_log_buffer) > 10000:
            self.detail_log_buffer = self.detail_log_buffer[-8000:]
        if self.detail_log_text is not None:
            self.detail_log_text.append(line)

        if not self._is_compact_log_line(text, level):
            with self.log_file.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            return

        color = {
            "INFO": QColor("#1f2937"),
            "WARN": QColor("#b45309"),
            "ERROR": QColor("#b91c1c"),
        }.get(level, QColor("#1f2937"))

        cursor = self.log_view.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_view.setTextCursor(cursor)
        self.log_view.setTextColor(color)
        self.log_view.insertPlainText(line + "\n")
        self.log_view.ensureCursorVisible()

        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def refresh_data_status(self):
        setting = self._train_profile_settings(self.train_profile_combo.currentData())
        self.data_root_label.setText(setting["data_rel"])
        data_dir = self.root / Path(setting["data_rel"])
        status = inspect_stgcn_dataset(data_dir)
        self.data_detail_label.setText(status.message)
        if status.ok:
            self._set_status_card(self.data_status_value, "正常", "#16a34a")
        else:
            self._set_status_card(self.data_status_value, "异常", "#dc2626")
            self._log(status.message, "ERROR")

    def open_log_folder(self):
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            import os

            os.startfile(str(self.log_file.parent))
        except Exception as e:
            self._log(f"打开日志目录失败: {e}", "ERROR")

    # ----------------------------- 模型列表 -----------------------------
    def _fill_combo(self, combo: QComboBox, files):
        combo.clear()
        name_count = {}
        for p in files:
            name_count[p.name] = name_count.get(p.name, 0) + 1
        for p in files:
            if name_count.get(p.name, 0) > 1:
                try:
                    label = str(p.resolve().relative_to(self.root))
                except Exception:
                    label = str(p.resolve())
            else:
                label = p.name
            combo.addItem(label, str(p.resolve()))

    def refresh_model_lists(self):
        show_all = self.show_all_model_versions_ck.isChecked()
        yolo_files = self.model_registry.yolo_files(show_all)
        stgcn_files = self.model_registry.stgcn_files(show_all)
        pretrain_files = self.model_registry.stgcn_pretrain_files(show_all)

        for c in [self.test_yolo_combo, self.infer_yolo_combo]:
            self._fill_combo(c, yolo_files)

        for c in [self.test_stgcn_combo, self.infer_stgcn_combo]:
            self._fill_combo(c, stgcn_files)

        self._fill_combo(self.train_init_weight_combo, pretrain_files)
        self.train_init_weight_combo.insertItem(0, "从头训练（不加载权重）", "")
        self._sync_train_init_weight_by_profile(self.train_layout_value.text().strip())
        self._refresh_train_init_policy_hint()

        preferred_yolo = str(self.model_registry.preferred_yolo())
        for c in [self.test_yolo_combo, self.infer_yolo_combo]:
            idx = c.findData(preferred_yolo)
            if idx >= 0:
                c.setCurrentIndex(idx)

        preferred_stgcn = str(self.model_registry.preferred_stgcn())
        for c in [self.test_stgcn_combo, self.infer_stgcn_combo]:
            idx = c.findData(preferred_stgcn)
            if idx >= 0:
                c.setCurrentIndex(idx)
            elif c.count() > 0:
                epoch_idx = -1
                epoch_max = -1
                for i in range(c.count()):
                    m = re.match(r"epoch(\d+)_model\.pt$", c.itemText(i).lower())
                    if m:
                        ep = int(m.group(1))
                        if ep > epoch_max:
                            epoch_max = ep
                            epoch_idx = i
                c.setCurrentIndex(epoch_idx if epoch_idx >= 0 else 0)
            else:
                c.addItem("无可用模型", "")
                c.setCurrentIndex(0)

        if yolo_files:
            self._set_status_card(self.model_status_value, "就绪", "#16a34a")
        else:
            self._set_status_card(self.model_status_value, "待补充", "#f39c12")
        mode = "完整" if show_all else "精简"
        self._log(f"模型列表刷新完成（{mode}模式，默认best策略）")

    # ----------------------------- 推理画面 -----------------------------
    def _display_frame_in_label(self, frame_bgr):
        if frame_bgr is None:
            return
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = rgb.copy(order="C")
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        pix = pix.scaled(self.infer_frame_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.infer_frame_label.setPixmap(pix)

    def _update_infer_table(self, rows):
        self.infer_table.setRowCount(len(rows))
        for r, item in enumerate(rows):
            self.infer_table.setItem(r, 0, QTableWidgetItem(str(item["id"])))
            self.infer_table.setItem(r, 1, QTableWidgetItem(f"{item['det_conf']:.3f}"))
            self.infer_table.setItem(r, 2, QTableWidgetItem(str(item["action"])))
            self.infer_table.setItem(r, 3, QTableWidgetItem(f"{item['action_conf']:.3f}"))
            self.infer_table.setItem(r, 4, QTableWidgetItem(item["bbox"]))

    def _on_infer_state_changed(self, state: str):
        if state == "running":
            self.current_task = "infer_local"
            self.btn_infer_start.setEnabled(False)
            self.btn_infer_stop.setEnabled(True)
            self._set_status_card(self.task_status_value, "运行中:infer", "#2563eb")
            return

        self.current_task = "idle"
        self.btn_infer_start.setEnabled(True)
        self.btn_infer_stop.setEnabled(False)
        self._set_status_card(self.task_status_value, "空闲", "#6b7280")
        self.infer_frame_label.setText("等待推理启动...")
        self.infer_frame_label.setPixmap(QPixmap())

    # ----------------------------- 子进程任务 -----------------------------
    def _start_process(self, cmd, task_name: str):
        if self.process is not None:
            QMessageBox.warning(self, "提示", "已有任务在运行，请先停止当前任务。")
            return
        if self.infer_session.is_running:
            QMessageBox.warning(self, "提示", "推理会话运行中，请先停止后再执行该任务。")
            return
        if not self.python.exists():
            QMessageBox.critical(self, "错误", f"Python解释器不存在: {self.python}")
            return

        self.process = QProcess(self)
        self.current_task = task_name
        self.process.setProgram(cmd[0])
        self.process.setArguments(cmd[1:])
        self.process.setWorkingDirectory(str(self.root))
        self.process.readyReadStandardOutput.connect(self._on_stdout)
        self.process.readyReadStandardError.connect(self._on_stderr)
        self.process.finished.connect(self._on_finished)
        self.process.start()

        self.btn_train_start.setEnabled(False)
        self.btn_infer_start.setEnabled(False)
        self.btn_train_stop.setEnabled(task_name == "train")
        self.btn_ablation_stop.setEnabled(task_name in {"ablation", "ablation_run"})
        self.btn_infer_stop.setEnabled(False)
        self._set_status_card(self.task_status_value, f"运行中:{task_name}", "#2563eb")
        self._log("启动命令: " + " ".join(cmd))

    def _on_stdout(self):
        if self.process is None:
            return
        data = bytes(self.process.readAllStandardOutput()).decode(errors="ignore")
        if data.strip():
            for ln in data.rstrip().splitlines():
                self._log(ln, "INFO")

    def _on_stderr(self):
        if self.process is None:
            return
        data = bytes(self.process.readAllStandardError()).decode(errors="ignore")
        if data.strip():
            for ln in data.rstrip().splitlines():
                self._log(ln, "ERROR")

    def _on_finished(self, exit_code=0, exit_status=None):
        finished_task = self.current_task
        self._log(f"任务结束: {finished_task}")

        if finished_task == "env":
            if int(exit_code) == 0:
                self._set_status_card(self.env_status_value, "正常", "#16a34a")
            else:
                self._set_status_card(self.env_status_value, "异常", "#dc2626")

        self.process = None
        self.current_task = "idle"
        self.btn_train_start.setEnabled(True)
        self.btn_infer_start.setEnabled(True)
        self.btn_train_stop.setEnabled(False)
        self.btn_ablation_stop.setEnabled(False)
        self.btn_infer_stop.setEnabled(False)
        self._set_status_card(self.task_status_value, "空闲", "#6b7280")

    def stop_process(self):
        if self.infer_session.is_running:
            self.infer_session.stop("用户手动停止推理。")
            return
        if self.process is not None:
            self.process.kill()
            self._log("任务已手动停止", "WARN")

    # ----------------------------- 操作事件 -----------------------------
    def run_env_check(self, auto_fix: bool):
        cmd = [str(self.python), str(self.root / "scripts" / "env" / "check_env.py")]
        if auto_fix:
            cmd += ["--fix", "--prefer-gpu"]
        self._start_process(cmd, "env")
        self._set_status_card(self.env_status_value, "检测中", "#2563eb")

    def run_train(self):
        mode = self.train_mode_combo.currentText()
        setting = self._train_profile_settings(self.train_profile_combo.currentData())
        config_path = setting["config"]
        init_weight = self.train_init_weight_combo.currentData()
        layout = setting["layout"]

        if not config_path:
            QMessageBox.warning(self, "提示", "训练配置不能为空。")
            return

        config_file = self._resolve_user_path(config_path).resolve()
        if not config_file.exists():
            QMessageBox.warning(self, "提示", f"训练配置不存在: {config_file}")
            return

        if init_weight:
            inferred = infer_layout_from_stgcn_weights(str(init_weight))
            if inferred in {"yolopose", "openpose"} and inferred != layout:
                QMessageBox.warning(
                    self,
                    "提示",
                    f"初始化权重布局与训练方案不一致: weights={inferred}, target={layout}。请更换初始化模型。",
                )
                return

        if mode == "预训练初始化":
            cmd = [
                str(self.python),
                str(self.root / "scripts" / "train" / "train_stgcn_pretrain.py"),
                "--config",
                str(config_file),
            ]
            if init_weight:
                cmd += ["--weights", init_weight]
        else:
            if not init_weight:
                QMessageBox.warning(self, "提示", "微调训练需要选择已有初始化模型。")
                return
            cmd = [
                str(self.python),
                str(self.root / "scripts" / "train" / "train_stgcn_finetune.py"),
                "--config",
                str(config_file),
            ]
            cmd += ["--weights", init_weight]

        if layout:
            cmd += ["--layout", layout]

        self._start_process(cmd, "train")

    def run_model_test(self):
        yolo = self.test_yolo_combo.currentData()
        stgcn = self.test_stgcn_combo.currentData()
        layout = self.test_layout_combo.currentData()
        if not yolo or not stgcn:
            QMessageBox.warning(self, "提示", "请先选择YOLO和ST-GCN模型。")
            return

        cmd = [
            str(self.python),
            str(self.root / "scripts" / "tools" / "model_smoke_test.py"),
            "--yolo",
            yolo,
            "--stgcn",
            stgcn,
            "--layout",
            layout if layout else "auto",
        ]
        self._start_process(cmd, "test")

    def run_ablation_plan(self):
        if not self._refresh_ablation_compatibility_status():
            return

        out_dir = self.ablation_out_combo.currentText().strip()
        if not out_dir:
            QMessageBox.warning(self, "提示", "请先设置消融输出目录。")
            return
        theme = self.ablation_theme_combo.currentData() or "yolo_modules"
        profile = self.ablation_profile_combo.currentData() or self.default_train_profile
        model_yaml = self._extract_combo_path(self.ablation_model_yaml_combo.currentText().strip())
        data_yaml = self._extract_combo_path(self.ablation_data_yaml_combo.currentText().strip())
        epochs = int(self.ablation_epochs_spin.value())
        device = self.ablation_device_line.text().strip()

        if model_yaml and not Path(model_yaml).exists():
            QMessageBox.warning(self, "提示", f"模型YAML不存在: {model_yaml}")
            return
        if data_yaml and not Path(data_yaml).exists():
            QMessageBox.warning(self, "提示", f"数据YAML不存在: {data_yaml}")
            return

        if theme == "yolo_modules" and model_yaml:
            mcat = self._classify_model_yaml(Path(model_yaml))
            dcat = self._classify_data_yaml(Path(data_yaml)) if data_yaml else None
            if dcat and mcat != dcat:
                QMessageBox.warning(self, "提示", f"模型与数据任务类型不一致：model={mcat}, data={dcat}。")
                return
        if theme == "yolo_modules" and data_yaml:
            dcat = self._classify_data_yaml(Path(data_yaml))
            mcat = self._classify_model_yaml(Path(model_yaml)) if model_yaml else None
            if mcat and mcat != dcat:
                QMessageBox.warning(self, "提示", f"模型与数据任务类型不一致：model={mcat}, data={dcat}。")
                return

            if dcat == "POSE":
                mk = self._read_kpt_count_from_yaml(Path(model_yaml)) if model_yaml else None
                dk = self._read_kpt_count_from_yaml(Path(data_yaml))
                if mk is not None and dk is not None and mk != dk:
                    QMessageBox.warning(self, "提示", f"关键点数不一致：模型={mk}，数据={dk}。")
                    return

                profile = self.ablation_profile_combo.currentData() or "v17"
                expected = 17 if profile == "v17" else 18
                if dk is not None and dk != expected:
                    QMessageBox.warning(self, "提示", f"数据方案为{profile}，但数据kpt_shape={dk}，应为{expected}。")
                    return

        cmd = [
            str(self.python),
            str(self.root / "scripts" / "tools" / "generate_ablation_plan.py"),
            "--output-dir",
            str(Path(out_dir).resolve()),
            "--theme",
            str(theme),
            "--data-profile",
            str(profile),
        ]
        if model_yaml:
            cmd += ["--model-yaml", str(Path(model_yaml).resolve())]
        if data_yaml:
            cmd += ["--data-yaml", str(Path(data_yaml).resolve())]
        if epochs > 0:
            cmd += ["--epochs", str(epochs)]
        if device:
            cmd += ["--device", device]

        self._update_recent_list("recent_ablation_dirs", str(Path(out_dir).resolve()))
        if model_yaml:
            self._update_recent_list("recent_ablation_model_yaml", str(Path(model_yaml).resolve()))
        if data_yaml:
            self._update_recent_list("recent_ablation_data_yaml", str(Path(data_yaml).resolve()))
        self._start_process(cmd, "ablation")

    def run_ablation_batch(self):
        if not self._refresh_ablation_compatibility_status():
            return

        out_dir = self.ablation_out_combo.currentText().strip()
        if not out_dir:
            QMessageBox.warning(self, "提示", "请先设置消融输出目录。")
            return
        out_path = Path(out_dir).resolve()
        if not (out_path / "run_commands.txt").exists() or not (out_path / "ablation_plan.csv").exists():
            QMessageBox.warning(self, "提示", "未找到 run_commands.txt 或 ablation_plan.csv，请先生成规范模板。")
            return

        cmd = [
            str(self.python),
            str(self.root / "scripts" / "tools" / "run_ablation_batch.py"),
            "--plan-dir",
            str(out_path),
        ]
        if self.ablation_continue_on_error_ck.isChecked():
            cmd.append("--continue-on-error")
        if self.ablation_dry_run_ck.isChecked():
            cmd.append("--dry-run")
        self._start_process(cmd, "ablation_run")

    def choose_video(self):
        fp, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频",
            str(self.root),
            "Video Files (*.mp4 *.avi *.mov *.mkv)",
        )
        if fp:
            self.infer_video_line.setText(fp)

    def run_infer(self):
        yolo = self.infer_yolo_combo.currentData()
        stgcn = self.infer_stgcn_combo.currentData()
        layout = self.infer_layout_combo.currentData()
        if not yolo or not stgcn:
            QMessageBox.warning(self, "提示", "请先选择推理模型。")
            return

        cfg_file = (self.root / "configs" / "infer" / "runtime_out3.json").resolve()
        try:
            cfg = finalize_config(load_runtime_config(cfg_file))
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载推理配置失败: {e}")
            return

        cfg.yolo_pose_weights = str(Path(yolo).resolve())
        cfg.stgcn_weights = str(Path(stgcn).resolve())
        cfg.show_window = self.infer_show_ck.isChecked()

        if layout and layout != "auto":
            cfg.graph_layout = layout
        else:
            inferred = infer_layout_from_stgcn_weights(cfg.stgcn_weights)
            if inferred in {"yolopose", "openpose"}:
                cfg.graph_layout = inferred
                self._log(f"推理布局自动推断为: {cfg.graph_layout}")

        inferred_num_class = infer_num_class_from_stgcn_weights(cfg.stgcn_weights)
        if inferred_num_class and inferred_num_class > 0:
            if cfg.num_class != inferred_num_class:
                cfg.num_class = inferred_num_class
                self._log(f"推理类别数自动推断为: {cfg.num_class}")
            if len(cfg.class_names) != cfg.num_class:
                if len(cfg.class_names) > cfg.num_class:
                    cfg.class_names = cfg.class_names[: cfg.num_class]
                else:
                    cfg.class_names = [f"class_{i}" for i in range(cfg.num_class)]

        video_path = self.infer_video_line.text().strip()
        if video_path:
            cfg.video_path = str(self._resolve_user_path(video_path).resolve())

        if cfg.save_path:
            self._refresh_infer_path_labels(Path(cfg.save_path).resolve().parent)
        else:
            self._refresh_infer_path_labels(self.default_infer_output_dir)

        if not Path(cfg.video_path).exists():
            QMessageBox.warning(self, "提示", f"视频文件不存在: {cfg.video_path}")
            return
        if not Path(cfg.yolo_pose_weights).exists():
            QMessageBox.warning(self, "提示", f"YOLO模型不存在: {cfg.yolo_pose_weights}")
            return
        if not Path(cfg.stgcn_weights).exists():
            QMessageBox.warning(self, "提示", f"ST-GCN模型不存在: {cfg.stgcn_weights}")
            return

        self.infer_session.start(cfg)


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
