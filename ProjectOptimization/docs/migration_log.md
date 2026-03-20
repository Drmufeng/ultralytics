# 迁移与重构日志

## 基本原则

- 客户原始目录：`<workspace_root>/customer` 未修改。
- 优化工作目录：`<workspace_root>/ProjectOptimization`。
- 环境复用：`<workspace_root>/.venv310`。

## 已完成事项

1. 建立工程化目录结构（src/data/models/configs/scripts/outputs/logs/ui/docs/archives）。
2. 复制客户代码与数据到本工程内部（不引用外部共享目录）。
3. 重新接入并清理三模块：`ADown`、`DyRFAUp`、`SCSA`。
4. 修复模块注册链路（`modules/__init__.py`、`tasks.py`）。
5. 训练端加入：
   - GPU可用自动使用GPU，无GPU自动回退CPU。
   - 验证Top1最优自动保存`best.pt`。
6. 新增工程级训练脚本：预训练初始化、微调、菜单入口。
7. 新增工程级推理脚本（配置驱动、ByteTrack接入、可指定ST-GCN权重）。
8. 新增Qt5可视化接口占位层（event bus + hooks）。
9. 新增环境检测与自动修复脚本（支持GPU优先安装）。
10. 模型已在本工程内归类复制。
11. 新增Qt5企业版控制台UI：
    - 四Tab：环境与数据、训练、测试、推理。
    - 双模型下拉框（默认best策略）。
    - 环境检测、模型测试、推理启动、任务中断、日志观测落盘。
12. 完成路径去硬编码：
    - 移除脚本中的绝对盘符路径；
    - 环境脚本改为动态解析解释器（支持 `PYTHON_EXE` 覆盖）。
13. 新增路径可迁移性报告：`docs/path_portability_report_zh.md`。
14. 新增17/18关键点布局可选机制：
    - 训练脚本支持 `--layout auto/yolopose/openpose`；
    - 推理脚本支持 `--layout auto/yolopose/openpose`；
    - UI提供布局下拉框，默认自动模式。
15. UI新增模型版本展示开关：
    - 默认精简模式（best/预训练/最新epoch）；
    - 可切换完整模式查看全部epoch。
16. UI推理页增强：
    - 内嵌视频画面区实时显示；
    - 目标表格实时展示检测与动作结果；
    - 支持17/18布局下拉选择与自动推断。
