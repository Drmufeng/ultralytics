import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


def build_yolo_module_matrix() -> list[dict]:
    items = [
        ("A0", "baseline", "-", "基线模型，不启用新增模块", set()),
        ("A1", "adown", "ADown", "仅启用 ADown", {"ADown"}),
        ("A2", "dyrfaup", "DyRFAUp", "仅启用 DyRFAUp", {"DyRFAUp"}),
        ("A3", "scsa", "SCSA", "仅启用 SCSA", {"SCSA"}),
        ("A4", "adown+dyrfaup", "ADown,DyRFAUp", "启用 ADown + DyRFAUp", {"ADown", "DyRFAUp"}),
        ("A5", "adown+scsa", "ADown,SCSA", "启用 ADown + SCSA", {"ADown", "SCSA"}),
        ("A6", "dyrfaup+scsa", "DyRFAUp,SCSA", "启用 DyRFAUp + SCSA", {"DyRFAUp", "SCSA"}),
        ("A7", "all3", "ADown,DyRFAUp,SCSA", "三模块全启用", {"ADown", "DyRFAUp", "SCSA"}),
    ]
    rows = []
    for exp_id, tag, modules, note, mod_set in items:
        rows.append(
            {
                "exp_id": exp_id,
                "variant": tag,
                "modules_on": modules,
                "note": note,
                "modules_set": mod_set,
            }
        )
    return rows


def _is_pose_yaml(text: str) -> bool:
    return "kpt_shape" in text and "Pose" in text


def _classify_model_yaml(path: Path, text: str) -> str:
    s = path.name.lower()
    if "pose" in s or _is_pose_yaml(text):
        return "POSE"
    if "-cls" in s or "cls" in s:
        return "CLS"
    if "seg" in s:
        return "SEG"
    if "obb" in s:
        return "OBB"
    if "detect" in s or "yolo" in s:
        return "DETECT"
    return "OTHER"


def _classify_data_yaml(path: Path) -> str:
    s = path.name.lower()
    if "pose" in s or "keypoint" in s:
        return "POSE"
    if "cls" in s or "class" in s:
        return "CLS"
    if "seg" in s:
        return "SEG"
    if "obb" in s:
        return "OBB"
    return "DETECT"


def _task_from_model_category(cat: str) -> str:
    return {
        "POSE": "pose",
        "DETECT": "detect",
        "CLS": "classify",
        "SEG": "segment",
        "OBB": "obb",
    }.get(cat, "detect")


def _read_kpt_count(path: Path) -> int | None:
    try:
        import yaml

        obj = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    kpt = obj.get("kpt_shape", None)
    if isinstance(kpt, (list, tuple)) and len(kpt) >= 1:
        try:
            return int(kpt[0])
        except Exception:
            return None
    return None


def _replace_dyrfaup(text: str, enabled: bool) -> str:
    if enabled:
        text = text.replace("nn.Upsample", "DyRFAUp")
    else:
        text = text.replace("DyRFAUp", "nn.Upsample")
    return text


def _replace_adown(text: str, enabled: bool) -> str:
    if enabled:
        text = text.replace("- [-1, 1, Conv, [256, 3, 2]]", "- [-1, 1, ADown, [256]]")
        text = text.replace("- [-1, 1, Conv, [512, 3, 2]]", "- [-1, 1, ADown, [512]]")
    else:
        text = text.replace("- [-1, 1, ADown, [256]]", "- [-1, 1, Conv, [256, 3, 2]]")
        text = text.replace("- [-1, 1, ADown, [512]]", "- [-1, 1, Conv, [512, 3, 2]]")
    return text


def _replace_scsa(text: str, enabled: bool) -> str:
    if enabled:
        text = text.replace("- [-1, 2, C2PSA, [1024]]", "- [-1, 1, SCSA, []]")
    else:
        text = text.replace("- [-1, 1, SCSA, []]", "- [-1, 2, C2PSA, [1024]]")
    return text


def _apply_modules(base_text: str, modules: set[str]) -> str:
    out = base_text
    out = _replace_dyrfaup(out, "DyRFAUp" in modules)
    out = _replace_adown(out, "ADown" in modules)
    out = _replace_scsa(out, "SCSA" in modules)
    return out


def _q(s: str) -> str:
    if s.startswith("<") and s.endswith(">"):
        return s
    return f'"{s}"'


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate standardized ablation experiment plan")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--theme", default="yolo_modules", choices=["yolo_modules"])
    parser.add_argument("--data-profile", default="v17", choices=["v17", "v18"])
    parser.add_argument("--model-yaml", default="")
    parser.add_argument("--data-yaml", default="")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.data_profile == "v17":
        layout = "yolopose"
        train_config = "configs/train/train_stgcn_out17.yaml"
    else:
        layout = "openpose"
        train_config = "configs/train/train_stgcn_out3.yaml"

    rows = build_yolo_module_matrix()
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 写计划表
    csv_path = out_dir / "ablation_plan.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["exp_id", "variant", "modules_on", "note", "data_profile", "layout", "train_config"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "exp_id": r["exp_id"],
                    "variant": r["variant"],
                    "modules_on": r["modules_on"],
                    "note": r["note"],
                    "data_profile": args.data_profile,
                    "layout": layout,
                    "train_config": train_config,
                }
            )

    # 生成可执行命令：若给了 model/data/epochs 则直接生成可跑版
    model_yaml = args.model_yaml.strip()
    data_yaml = args.data_yaml.strip()
    epochs = str(args.epochs) if args.epochs and args.epochs > 0 else "<EPOCHS>"
    device = (args.device or "").strip()

    generated_models_dir = out_dir / "generated_models"
    generated_models_dir.mkdir(parents=True, exist_ok=True)

    model_mode = "placeholder"
    model_ref_by_exp: dict[str, str] = {}
    warning_lines: list[str] = []
    model_category = "UNKNOWN"
    data_category = "UNKNOWN"
    task_name = "detect"

    model_kpt = None
    if model_yaml:
        model_path = Path(model_yaml).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"模型YAML不存在: {model_path}")
        base_text = model_path.read_text(encoding="utf-8")
        model_category = _classify_model_yaml(model_path, base_text)
        task_name = _task_from_model_category(model_category)
        if model_category == "OTHER":
            warning_lines.append("模型类型无法明确识别，默认按 detect 任务运行。")

        model_kpt = _read_kpt_count(model_path)

        model_mode = "generated"
        for r in rows:
            exp_id = r["exp_id"]
            variant = r["variant"]
            transformed = _apply_modules(base_text, r["modules_set"])
            out_model = generated_models_dir / f"{exp_id}_{variant}.yaml"
            out_model.write_text(transformed, encoding="utf-8")
            model_ref_by_exp[exp_id] = str(out_model)
    else:
        for r in rows:
            model_ref_by_exp[r["exp_id"]] = "<YOUR_MODEL_YAML>"
        warning_lines.append("未提供模型YAML，run_commands.txt 使用占位符。")

    if not data_yaml:
        warning_lines.append("未提供数据YAML，run_commands.txt 使用占位符。")
        data_yaml = "<YOUR_DATA_YAML>"
    else:
        data_path = Path(data_yaml).resolve()
        if not data_path.exists():
            raise FileNotFoundError(f"数据YAML不存在: {data_path}")
        data_category = _classify_data_yaml(data_path)
        data_kpt = _read_kpt_count(data_path)
        if model_yaml and model_category != "UNKNOWN" and data_category != "UNKNOWN" and model_category != data_category:
            raise ValueError(f"模型与数据任务类型不一致：model={model_category}, data={data_category}。")
        if model_category == "POSE":
            if data_kpt is None:
                raise ValueError("Pose 消融要求数据YAML包含 kpt_shape。")
            expected = 17 if args.data_profile == "v17" else 18
            if data_kpt != expected:
                raise ValueError(f"数据方案为 {args.data_profile}，但数据 kpt_shape={data_kpt}，应为 {expected}。")
            if model_yaml and model_kpt is not None and model_kpt != data_kpt:
                raise ValueError(f"关键点数不一致：模型 kpt_shape={model_kpt}，数据 kpt_shape={data_kpt}。")
        data_yaml = str(data_path)

    # 命令模板
    cmd_path = out_dir / "run_commands.txt"
    cmd_lines = [
        "# 命令模板（支持批跑）",
        "# 说明：本主题固定使用 yolo pose train。",
    ]
    if warning_lines:
        cmd_lines.append("# 注意：" + "；".join(warning_lines))

    for r in rows:
        exp_id = r["exp_id"]
        variant = r["variant"]
        model_ref = model_ref_by_exp[exp_id]
        cmd_lines.append(
            f"# {exp_id} {variant} modules={r['modules_on']}\n"
            f"yolo {task_name} train model={_q(model_ref)} data={_q(data_yaml)} epochs={epochs} "
            f"{('device=' + device + ' ') if device else ''}"
            f"project=outputs/eval/ablation name={exp_id}_{variant}"
        )
    cmd_path.write_text("\n\n".join(cmd_lines) + "\n", encoding="utf-8")

    md_path = out_dir / "ablation_plan.md"
    lines = [
        "# 消融实验计划（规范版）",
        "",
        f"- 生成时间：{generated_at}",
        f"- 主题：{args.theme}",
        f"- 数据方案：{args.data_profile}",
        f"- 关键点布局：{layout}",
        f"- 训练配置：`{train_config}`",
        f"- 模型来源：`{model_mode}`",
        f"- 模型任务类型：`{model_category}`",
        f"- 数据任务类型：`{data_category}`",
        f"- 训练命令任务：`{task_name}`",
        "",
        "## 固定控制变量",
        "",
        "- 数据集、训练轮次、优化器、学习率策略、随机种子保持一致。",
        "- 每次只改变模块组合（单变量原则）。",
        "",
        "## 实验矩阵",
        "",
        "| 实验ID | 变体 | 启用模块 | 说明 |",
        "|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['exp_id']} | {r['variant']} | {r['modules_on']} | {r['note']} |")

    lines += [
        "",
        "## 结果记录模板",
        "",
        "请记录以下指标：Top1、Top4、参数量、GFLOPs、推理FPS。",
        "",
        "| 实验ID | Top1 | Top4 | Params(M) | GFLOPs | FPS | 备注 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(f"| {r['exp_id']} |  |  |  |  |  |  |")

    if warning_lines:
        lines += ["", "## 提示", ""] + [f"- {w}" for w in warning_lines]

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    meta = {
        "generated_at": generated_at,
        "theme": args.theme,
        "data_profile": args.data_profile,
        "layout": layout,
        "train_config": train_config,
        "model_mode": model_mode,
        "model_category": model_category,
        "data_category": data_category,
        "task_name": task_name,
        "model_yaml_input": model_yaml,
        "data_yaml_input": data_yaml,
        "files": [str(csv_path), str(md_path), str(cmd_path)],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] 消融模板已生成: {out_dir}")
    print(f"[OK] 计划文件: {md_path}")
    print(f"[OK] 表格文件: {csv_path}")
    print(f"[OK] 命令模板: {cmd_path}")
    if model_mode == "generated":
        print(f"[OK] 变体模型YAML目录: {generated_models_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
