# ProjectOptimization

该目录是优化项目的唯一工作区，用于代码开发、模型训练、推理、文档、日志与后续可视化扩展。

## 约束规则

- 客户原始目录 `<workspace_root>/customer` 仅作为基线，保持只读不修改。
- 复用已有环境 `.venv310`（优先 `<workspace_root>/.venv310`），不单独新建虚拟环境。
- 代码与配置禁止使用绝对硬编码路径，统一走相对路径。
- Qt5 可视化已实现企业版界面（`ui/qt5`），支持环境检测、训练/测试、推理可视化与日志观测。

## 主要入口

- 训练菜单入口：`scripts/train/train_stgcn_menu.py`
- 预训练初始化训练：`scripts/train/train_stgcn_pretrain.py`
- 微调训练：`scripts/train/train_stgcn_finetune.py`
- 推理入口：`scripts/infer/run_pose_stgcn.py`
- 环境检测入口：`scripts/env/check_env.py`

## 训练方案说明

- 17关键点：`configs/train/train_stgcn_out17.yaml`
- 18关键点：`configs/train/train_stgcn_out3.yaml`
- Qt5 训练页已将“训练方案/关键点布局/训练配置”联动，减少17/18错配。
