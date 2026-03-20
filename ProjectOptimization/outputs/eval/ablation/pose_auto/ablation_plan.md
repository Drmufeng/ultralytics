# 消融实验计划（规范版）

- 生成时间：2026-03-17 09:44:59
- 主题：yolo_modules
- 数据方案：v17
- 关键点布局：yolopose
- 训练配置：`configs/train/train_stgcn_out17.yaml`
- 模型来源：`generated`

## 固定控制变量

- 数据集、训练轮次、优化器、学习率策略、随机种子保持一致。
- 每次只改变模块组合（单变量原则）。

## 实验矩阵

| 实验ID | 变体 | 启用模块 | 说明 |
|---|---|---|---|
| A0 | baseline | - | 基线模型，不启用新增模块 |
| A1 | adown | ADown | 仅启用 ADown |
| A2 | dyrfaup | DyRFAUp | 仅启用 DyRFAUp |
| A3 | scsa | SCSA | 仅启用 SCSA |
| A4 | adown+dyrfaup | ADown,DyRFAUp | 启用 ADown + DyRFAUp |
| A5 | adown+scsa | ADown,SCSA | 启用 ADown + SCSA |
| A6 | dyrfaup+scsa | DyRFAUp,SCSA | 启用 DyRFAUp + SCSA |
| A7 | all3 | ADown,DyRFAUp,SCSA | 三模块全启用 |

## 结果记录模板

请记录以下指标：Top1、Top4、参数量、GFLOPs、推理FPS。

| 实验ID | Top1 | Top4 | Params(M) | GFLOPs | FPS | 备注 |
|---|---:|---:|---:|---:|---:|---|
| A0 |  |  |  |  |  |  |
| A1 |  |  |  |  |  |  |
| A2 |  |  |  |  |  |  |
| A3 |  |  |  |  |  |  |
| A4 |  |  |  |  |  |  |
| A5 |  |  |  |  |  |  |
| A6 |  |  |  |  |  |  |
| A7 |  |  |  |  |  |  |
