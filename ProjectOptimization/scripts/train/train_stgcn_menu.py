import subprocess
import sys
from pathlib import Path


def run(script_name: str) -> int:
    # 菜单只负责分发，不直接承载训练逻辑。
    root = Path(__file__).resolve().parent
    cmd = [sys.executable, str(root / script_name)]
    print("[INFO]", " ".join(cmd))
    return subprocess.call(cmd)


def main() -> int:
    # IDE双击运行时，直接通过中文菜单选择训练阶段。
    print("请选择训练模式：")
    print("1) 预训练初始化训练")
    print("2) 微调训练")
    print("0) 退出")
    choice = input("请输入编号：").strip()
    if choice == "1":
        return run("train_stgcn_pretrain.py")
    if choice == "2":
        return run("train_stgcn_finetune.py")
    if choice == "0":
        return 0
    print("输入无效。")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
