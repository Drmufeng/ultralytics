项目完整说明
本文档基于对 全目录的遍历检索整理，目标是让第一次接手项目的人也能独立完成：环境检查、训练、测试、推理、结果查看与排错。

---

## 0. 先看结论
- 你真正工作的目录是：`B:\ultralytics\ProjectOptimization`
- 客户原始目录是：`B:\ultralytics\customer`（只读，别改）
- Python 环境固定用：`B:\ultralytics\.venv310`
- 训练数据目录：
  - 17点：`data/stgcn_dataset_out17/stgcn_dataset_out`
  - 18点：`data/stgcn_dataset_out3`
- 训练模型输出目录：`models/stgcn/out17` 或 `models/stgcn/out3`
- 推理输出目录：`outputs/infer`

---

## 1. 全目录遍历结果（根目录）

`\ultralytics` 下主要目录：

- `.venv310/`：统一 Python 运行环境（依赖包都在这里）
- `customer/`：客户原始基线（数据、代码、压缩包）
- `ProjectOptimization/`：优化后的正式工作区（开发与运行都在这里）
- `env/`：外层环境补充文件
- `.idea/`：IDE 配置

遍历统计（用于评估体量）：

- `ProjectOptimization`：约 264 个目录 / 1339 个文件
- `customer`：约 224 个目录 / 1223 个文件
- `.venv310`：约 2808 个目录 / 33299 个文件（环境本身）

---

## 2. 工作区总览（`ProjectOptimization` 每个文件夹做什么）

工作区根目录：`\ultralytics\ProjectOptimization`

- `archives/`：历史归档（备份、阶段性产物）
- `configs/`：训练/推理配置
  - `configs/train/`：ST-GCN 训练配置（17点/18点）
  - `configs/infer/`：推理运行参数（如 `runtime_out3.json`）
  - `configs/app/`：应用级配置
- `data/`：ST-GCN 数据输入目录
- `docs/`：中文文档、迁移日志、使用指南
- `env/`：项目依赖清单（如 `requirements-py310.txt`）
- `logs/`：运行日志与临时配置
  - `logs/run_logs/ui_app.log`：UI 主日志
  - `logs/run_logs/_tmp_train_*.yaml`：训练临时配置快照
- `models/`：模型权重目录（YOLO / ST-GCN）
- `outputs/`：推理和评估输出目录
- `scripts/`：直接运行的入口脚本（环境/训练/推理/工具）
- `src/`：核心源码（含 `ultralytics-main`）
- `ui/`：Qt5 图形界面源码

---

## 3. 输入目录与输出目录（你最常找的路径）

### 3.1 数据输入目录

- 17关键点数据：
  - `\ultralytics\ProjectOptimization\data\stgcn_dataset_out17\stgcn_dataset_out`
  - 关键文件：`train_data.npy`、`val_data.npy`、`train_label.pkl`、`val_label.pkl`
- 18关键点数据：
  - `\ultralytics\ProjectOptimization\data\stgcn_dataset_out3`
  - 关键文件：`train_data_c3v18.npy`、`val_data_c3v18.npy`、`train_label.pkl`、`val_label.pkl`

### 3.2 测试视频输入目录

- 默认测试视频目录：
  - `\ultralytics\ProjectOptimization\src\ultralytics-main\dataset\GCNData\BaseballPitch`

### 3.3 推理输出目录

- 默认目录：`\ultralytics\ProjectOptimization\outputs\infer`
- 默认输出文件：`output_multi_action_out3.mp4`

### 3.4 日志输出目录

- UI 日志：`\ultralytics\ProjectOptimization\logs\run_logs\ui_app.log`
- 训练日志：各训练目录内 `log.txt`（如 `models/stgcn/out17/log.txt`）

---

## 4. 模型目录说明（训练后放哪里）

### 4.1 YOLO 模型（关键点提取）

- 客户/自定义：`models/yolo/custom`
  - 例如：`yolo_pose_best.pt`、`yolo_pose_last.pt`
- 官方预训练：`models/yolo/pretrained`
  - 例如：`yolo11n-pose.pt`、`yolo11n.pt`
- 消融实验产物（批跑后）：`outputs/eval/ablation/*/weights`
  - 常见文件：`best.pt`、`last.pt`

### 4.2 ST-GCN 模型（动作分类）

- 17点训练输出：`models/stgcn/out17`
- 17点微调输出：`models/stgcn/out17_finetune`
- 18点训练输出：`models/stgcn/out3`
- 18点微调输出：`models/stgcn/out3_finetune`
- 历史权重：`models/stgcn/legacy`
- 预训练底座：`models/stgcn/pretrained`

常见保存规则（默认）：

- 每 10 epoch 会保存 `epochXX_model.pt`
- 验证效果更好时更新 `best.pt`

---

## 5. 配置文件怎么选（17点 / 18点）

训练配置在：`configs/train`

- `train_stgcn_out17.yaml`：17点训练
- `train_stgcn_out17_finetune.yaml`：17点微调
- `train_stgcn_out3.yaml`：18点训练
- `train_stgcn_out3_finetune.yaml`：18点微调

匹配规则（必须遵守）：

- 17点数据 <-> `yolopose` <-> 17点权重
- 18点数据 <-> `openpose` <-> 18点权重

---

## 6. 核心脚本入口（命令行）

入口目录：`\ultralytics\ProjectOptimization\scripts`

- 环境检查：`scripts/env/check_env.py`
- 预训练初始化：`scripts/train/train_stgcn_pretrain.py`
- 微调训练：`scripts/train/train_stgcn_finetune.py`
- 训练菜单：`scripts/train/train_stgcn_menu.py`
- 推理：`scripts/infer/run_pose_stgcn.py`
- 模型快速检查：`scripts/tools/model_smoke_test.py`
- 消融模板生成：`scripts/tools/generate_ablation_plan.py`
- 消融批跑执行：`scripts/tools/run_ablation_batch.py`
- Qt5 UI 启动器：`scripts/tools/run_qt5_app.py`

---

## 7. UI 使用指南（小白流程）

### 7.1 启动 UI

在 `\ultralytics\ProjectOptimization` 下执行：

```bash
\ultralytics\.venv310\Scripts\python.exe scripts/tools/run_qt5_app.py
```

### 7.2 四个页签做什么

- 环境与数据：检查 Python/Torch/CUDA 与数据完整性
- 训练：选择训练方案和模式，启动训练
- 测试：模型连通性与基本可用性检查
- 推理：选择模型与视频，输出带动作结果视频
- 消融：按规范生成消融实验矩阵（计划表 + 命令模板）

### 7.3 训练页关键点

- 训练方案（17点/18点）会联动：
  - 关键点布局
  - 训练配置
  - 初始化模型匹配
- 初始化策略会显示提示：
  - 继续预训练（保留分类头）
  - 迁移初始化（自动忽略 `fcn`）
  - 不兼容警告（布局/类别不匹配）

### 7.4 推理页关键点

- 会显示并可打开：
  - 测试视频目录
  - 推理输出目录
- 支持内嵌画面 + 结果表格显示

### 7.5 消融页关键点（外部目录也可用）

- 模型YAML和数据YAML支持两种方式：
  1) 下拉自动扫描项目内候选
  2) 手动选择任意目录文件（包括项目外）
- 无论是否在项目内，都会做兼容性检测（任务类型/关键点数/17-18方案）。
- 消融主题支持 `POSE/DETECT/CLS/SEG/OBB`，但要求模型YAML与数据YAML任务类型一致。
- 若是 Pose 消融，还会校验 kpt_shape 与 17/18 方案是否一致。
- 若希望“新增文件后自动出现在下拉中”，请放在：
  - `src/ultralytics-main/ultralytics/cfg/models/`
  - `src/ultralytics-main/ultralytics/cfg/datasets/`
- 放在外部目录也没问题：可手动选择，之后会进入“最近使用”。
- 消融训练完成后的模型默认保存在 `outputs/eval/ablation/*/weights`，可直接复用于测试/推理或下一轮消融。

### 7.5 日志查看

- 右侧日志：简易关键信息
- 详细日志窗口：完整日志（独立窗口）

---

## 8. 常见任务怎么做（一步一步）

### 8.1 从头训练 17点 ST-GCN

1) 训练方案选 17点（yolopose）

2) 训练模式选 预训练初始化

3) 初始化模型选“从头训练（不加载权重）”

4) 开始训练，产物在：`models/stgcn/out17`

### 8.2 继续训练（预训练不够）

1) 训练模式保持 预训练初始化

2) 初始化模型选上次产物（如 `best.pt` 或 `epochXX_model.pt`）

3) 看“初始化策略”提示是否合理，再开始

### 8.3 微调训练

1) 训练模式选 微调训练

2) 必须选择初始化权重（不能空）

3) 训练完成后看 `models/stgcn/out17_finetune` 或对应目录

### 8.4 跑推理并找输出

1) 进入推理页，选 YOLO + ST-GCN 模型

2) 选视频，开始推理

3) 输出在：`outputs/infer`

---

## 9. 常见问题（FAQ）

### Q1：训练了但说“没保存”？

先看 `models/stgcn/out17`（或 out3）是否有 `best.pt` / `epochXX_model.pt` / `log.txt`。

### Q2：新训练会污染客户原始目录吗？

不会。只要在 `ProjectOptimization` 运行，产物都写在 `ProjectOptimization/models`。

### Q3：为什么会报 17/18 不匹配？

因为数据、layout、初始化权重三者不一致。请统一关键点方案后再训。

### Q4：右侧日志为什么不全？

右侧是简易日志，完整日志请看“详细日志窗口”或 `logs/run_logs/ui_app.log`。

---

## 10. 不能踩的红线

- 不要在 `\ultralytics\customer` 中开发和训练
- 不要混用 17点数据与 18点权重（反之亦然）
- 不要随意改 `work_dir` 到项目外路径

---

## 11. 本次目录遍历参考快照

为保证文档准确性，已生成遍历快照：

- `\ultralytics\ProjectOptimization\logs\run_logs\_project_tree_depth4.txt`
- `\ultralytics\ProjectOptimization\logs\run_logs\_customer_tree_depth4.txt`

如后续目录发生变化，建议重新生成快照并同步更新本文件。
