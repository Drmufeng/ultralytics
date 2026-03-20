# 运行手册（ProjectOptimization）

## 1) 环境检测

使用复用环境：`<workspace_root>/.venv310`

```bash
python scripts/env/check_env.py --fix --prefer-gpu
```

## 2) 训练

### 训练前先检查关键点维度（强烈建议）

```bash
python scripts/tools/check_keypoint_layout.py --data data/stgcn_dataset_out3/train_data_c3v18.npy --config configs/train/train_stgcn_out3.yaml
```

说明：

- `--layout auto` 时按数据自动判定（V=17->yolopose，V=18->openpose）
- `yolopose` 必须对应 `V=17`
- `openpose` 必须对应 `V=18`
- 不一致时请先修正数据或配置，不要直接训练。

### 预训练初始化训练

```bash
python scripts/train/train_stgcn_pretrain.py --layout auto
```

### 微调训练

```bash
python scripts/train/train_stgcn_finetune.py --layout auto
```

可手动指定布局：

```bash
python scripts/train/train_stgcn_pretrain.py --layout yolopose   # 17关键点
python scripts/train/train_stgcn_pretrain.py --layout openpose   # 18关键点
```

### 菜单入口

```bash
python scripts/train/train_stgcn_menu.py
```

## 3) 推理

```bash
python scripts/infer/run_pose_stgcn.py --config configs/infer/runtime_out3.json
```

可手动指定推理布局：

```bash
python scripts/infer/run_pose_stgcn.py --config configs/infer/runtime_out3.json --layout auto
python scripts/infer/run_pose_stgcn.py --config configs/infer/runtime_out3.json --layout yolopose
python scripts/infer/run_pose_stgcn.py --config configs/infer/runtime_out3.json --layout openpose
```

可指定 ST-GCN 权重：

```bash
python scripts/infer/run_pose_stgcn.py --config configs/infer/runtime_out3.json --stgcn-weights models/stgcn/out3/epoch50_model.pt
```

## 4) Qt5 可视化控制台

```bash
python scripts/tools/run_qt5_app.py
```

界面包含：

- 预训练模型下拉框（YOLO，默认优先 best）
- 模型下拉框（ST-GCN，默认优先 best）
- 环境检测、模型测试、推理启动、日志观测

详细设计说明见：`docs/ui_design_spec_zh.md`
使用步骤见：`docs/ui_user_guide_zh.md`
