# 代码修改说明（改了什么、为什么改）

## 1. 模块接入相关

- `src/ultralytics-main/ultralytics/nn/modules/custom_blocks.py`
  - 改动：整理并接入 `ADown`、`DyRFAUp`、`SCSA`，清理重复定义。
  - 原因：避免同名模块重复导致行为不确定，确保三模块可维护、可复用。
  - 效果：模块实现单一、稳定，可直接被解析器调用。

- `src/ultralytics-main/ultralytics/nn/modules/__init__.py`
  - 改动：修复 `__all__` 导出项（补逗号），补齐 `RFAConv`、`DyRFAUp` 导出。
  - 原因：原导出存在拼接风险，可能导致导入异常。
  - 效果：模块可稳定 `import`。

- `src/ultralytics-main/ultralytics/nn/tasks.py`
  - 改动：新增 `DyRFAUp`、`SCSA` 的导入与 `parse_model` 参数组装分支。
  - 原因：仅实现模块不够，YOLO 构图阶段必须识别并正确注入通道参数。
  - 效果：YAML 中可直接使用新模块。

## 2. 训练链路相关

- `src/ultralytics-main/processor/io.py`
  - 改动：设备选择改为“有GPU走GPU，无GPU自动回退CPU”。
  - 原因：避免 CPU 版 torch 时训练直接报错退出。
  - 效果：同一套流程可跨设备运行。

- `src/ultralytics-main/processor/recognition.py`
  - 改动：记录评估 `Top-K` 到 `latest_topk`。
  - 原因：为 best 模型保存提供实时指标依据。
  - 效果：上层可依据 Top1 自动保存最佳权重。

- `src/ultralytics-main/processor/processor.py`
  - 改动：新增 best 追踪逻辑，验证 Top1 刷新时保存 `best.pt`。
  - 原因：仅保存 epoch 轮次文件不利于快速部署。
  - 效果：训练产物包含 `best.pt` 与 `epoch*_model.pt`。

## 3. 推理链路相关

- `scripts/infer/run_pose_stgcn.py`
  - 改动：配置驱动推理、ByteTrack 接入、关键点图结构可选（auto/yolopose/openpose）、权重/视频路径覆盖参数。
  - 原因：去硬编码，兼容新数据结构，并支持 UI 动态传参。
  - 效果：默认可跑，且可通过 CLI 覆盖 `yolo/stgcn/video/save/show`。

## 4. 训练入口分层

- `scripts/train/train_stgcn_pretrain.py`
  - 改动：预训练初始化训练入口（支持忽略分类头 + 17/18布局可选）。
  - 原因：将迁移学习流程与微调流程解耦。

- `scripts/train/train_stgcn_finetune.py`
  - 改动：微调训练入口（支持17/18布局可选）。
  - 原因：便于在 IDE 直接区分训练阶段。

- `scripts/train/train_stgcn_menu.py`
  - 改动：中文菜单入口（1预训练/2微调）。
  - 原因：降低操作门槛，避免参数误传。

## 5. 环境与可维护性

- `scripts/env/check_env.py`
  - 改动：依赖检查、关键文件检查、可选自动安装、GPU优先安装torch。
  - 原因：环境问题是最常见失败点，需要一键自检。

- `scripts/env/check_env_fix.bat`
  - 改动：双击触发 `--fix --prefer-gpu`。
  - 原因：便于非命令行用户使用。

## 6. Qt5 UI 接入（本期）

- `ui/qt5/main_window.py`
  - 改动：接入标准化控制台UI，包含：
    - 预训练模型下拉框（YOLO，默认优先 best）
    - 模型下拉框（ST-GCN，默认优先 best）
    - 环境检测按钮
    - 模型测试按钮
    - 视频选择与推理启动按钮
    - 推理页内嵌画面区与目标结果表格（ID/检测置信度/动作标签/动作置信度/位置）
    - 日志观测窗口（并落盘到 `logs/run_logs/ui_app.log`）
  - 原因：让客户可视化完成“选模型-测模型-跑推理-看日志”的闭环。

- `ui/qt5/services/model_registry.py`
  - 改动：新增模型扫描与默认 best 选择策略，支持“精简/完整”两种列表模式。
  - 原因：把模型发现逻辑从窗口层剥离，降低主窗口复杂度。

- `ui/qt5/services/data_inspector.py`
  - 改动：新增数据集完整性与关键点数（17/18）检测服务。
  - 原因：将数据校验从 UI 事件中抽离，便于复用并减少误配风险。

- `ui/qt5/services/inference_session.py`
  - 改动：新增内嵌推理会话服务，统一管理模型加载、视频循环、帧回调、表格回调与停止清理。
  - 原因：避免 `main_window.py` 承担推理主循环，实现 UI 与业务逻辑解耦。

- `ui/qt5/app_stub.py`
  - 改动：由占位文件升级为 Qt5 真正启动入口（创建 `QApplication` 并启动 `MainWindow`）。
  - 原因：统一 UI bootstrap，便于脚本和文档指向单一入口。

- `scripts/tools/model_smoke_test.py`
  - 改动：新增模型快速检查脚本。
  - 原因：在正式推理前快速验证权重可加载，减少运行期报错。

- `scripts/tools/run_qt5_app.py`
  - 改动：统一通过 `python -m ui.qt5.app_stub` 启动 UI。
  - 原因：避免重复维护入口逻辑，降低启动路径差异带来的问题。

## 7. 文档与结构

- `docs/runbook.md`、`docs/structure.md`、`docs/model_inventory.md`、`docs/migration_log.md`、`README.md`
  - 改动：统一中文文档，补全运行与结构说明。
  - 原因：降低交接成本，保证可审计、可复现。

## 8. 去硬编码（迁移稳定性）

- `scripts/env/check_env_fix.bat`
  - 改动：解释器路径改为动态解析（`PYTHON_EXE` > 外层`.venv310` > 项目内`.venv310` > PATH）。
  - 原因：避免换目录、换机器后路径失效。

- `src/ultralytics-main/tools/my_gendat.py`
  - 改动：移除固定盘符根路径，改为基于脚本位置动态推导项目根。
  - 原因：避免迁移后仍指向旧机器目录。
