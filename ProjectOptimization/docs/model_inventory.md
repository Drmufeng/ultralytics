# 模型清单

## YOLO

- `models/yolo/pretrained/yolo11n-pose.pt`
- `models/yolo/pretrained/yolo11n.pt`
- `models/yolo/custom/yolo_pose_best.pt`
- `models/yolo/custom/yolo_pose_last.pt`

## ST-GCN

- 预训练：`models/stgcn/pretrained/st_gcn.kinetics.pt`
- 历史：`models/stgcn/legacy/epoch10_model.pt` ... `epoch50_model.pt`
- OUT3：`models/stgcn/out3/`（训练后生成 `best.pt` 和 `epoch*_model.pt`）

## 使用建议

- 推理默认优先使用 `models/yolo/custom/yolo_pose_best.pt` 与 `models/stgcn/out3/best.pt`。
- 需要回溯实验时，再指定 `epoch*_model.pt` 进行对比。
- 所有模型命名与路径请保持英文，避免跨平台路径兼容问题。
