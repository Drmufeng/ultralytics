# Qt5 界面设计说明（V1）

## 目标

- 面向低门槛用户，提供“选模型-测模型-跑推理-看日志”的完整闭环。
- 不使用硬编码绝对路径，全部从项目根目录动态解析。

## 入口

- 启动脚本：`scripts/tools/run_qt5_app.py`
- 主窗口代码：`ui/qt5/main_window.py`

## 功能模块

1. 训练页
   - 预训练模型下拉框（YOLO）
     - 数据源：`models/yolo/custom` + `models/yolo/pretrained`
     - 默认：`yolo_pose_best.pt`
   - 关键点布局下拉框
     - 自动（按数据）
     - 17关键点（yolopose）
     - 18关键点（openpose）

2. 测试页
   - 模型下拉框（YOLO + ST-GCN）
   - 一键执行 `model_smoke_test.py`

3. 推理页
    - 预训练模型下拉框（YOLO）
      - 数据源：`models/yolo/custom` + `models/yolo/pretrained`
      - 默认：`yolo_pose_best.pt`
    - 模型下拉框（ST-GCN）
      - 数据源：`models/stgcn/out3` + `models/stgcn/legacy` + `models/stgcn/pretrained`
      - 默认：`models/stgcn/out3/best.pt`（未训练时显示“best.pt (待训练)”）
   - 关键点布局下拉框
     - 自动（按模型）
     - 17关键点（yolopose）
     - 18关键点（openpose）

4. 推理参数区
    - 视频路径选择（可选，空则使用配置默认视频）
    - 显示窗口开关

5. 推理可视化区（新增）
   - 内嵌视频画面区：实时显示推理帧。
   - 目标结果表格：展示 ID、检测置信度、动作标签、动作置信度、目标位置。

6. 操作区
    - 环境检测：调用 `scripts/env/check_env.py --fix --prefer-gpu`
    - 测试模型：调用 `scripts/tools/model_smoke_test.py`
    - 开始推理：调用 `scripts/infer/run_pose_stgcn.py`
    - 停止：中断当前任务进程

7. 日志观测区
    - 实时显示标准输出/错误输出
    - 同步落盘：`logs/run_logs/ui_app.log`

8. 模型列表展示策略
   - 默认精简模式：仅展示 best、预训练、最新 epoch，减少误选。
   - 可切换完整模式：显示全部历史 epoch。

## 防呆策略

- 任务运行中禁用重复启动，避免并发误操作。
- 模型未选择时禁止执行并提示。
- 模型默认优先选择 best 路径。
- 关键点布局支持 17/18 手动选择，训练前自动做维度一致性校验。
