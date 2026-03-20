# 路径可迁移性检查报告

## 目标

确保 `ProjectOptimization` 在迁移到客户机器后不因硬编码路径失效。

## 检查范围

- `scripts/`
- `configs/`
- `ui/`
- `env/`
- `docs/`

## 检查规则

- 禁止固定盘符路径（如 `B:\...`、`D:\...`）
- 禁止固定工程根路径（如 `B:/ultralytics/...`）
- 配置与脚本统一采用相对路径 + 动态解析项目根

## 修复项

1. `scripts/tools/_copy_models.ps1`
   - 从固定绝对路径改为基于 `$PSScriptRoot` 动态解析。
2. `scripts/env/check_env_fix.bat`
   - 解释器选择改为动态优先级：
     - `PYTHON_EXE` 环境变量
     - 外层 `.venv310`
     - 项目内 `.venv310`
     - `PATH` 中的 `python`
3. `src/ultralytics-main/tools/my_gendat.py`
   - 移除固定盘符根目录，改为基于脚本位置动态推导。
4. `src/ultralytics-main/runs/pose/train/args.yaml`
   - 将历史绝对 `save_dir` 改为相对路径。

## 结果

- 关键运行目录（`scripts/configs/ui/env/docs`）已无固定绝对路径。
- 当前项目可整体迁移到新路径后继续运行（前提：依赖与数据/模型同步迁移）。
