import argparse
import csv
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


def _load_commands(path: Path) -> list[str]:
    cmds = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        cmds.append(line)
    return cmds


def _load_exp_ids(path: Path) -> list[str]:
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    out = []
    for i, r in enumerate(rows):
        out.append(r.get("exp_id") or f"EXP_{i+1}")
    return out


def _normalize_command(raw: str) -> list[str]:
    cmd = raw.strip()
    if not cmd:
        return []

    tokens = shlex.split(cmd, posix=False)
    if not tokens:
        return []

    def clean_token(tok: str) -> str:
        t = tok.replace('\\"', '"').strip()
        if "=" in t:
            k, v = t.split("=", 1)
            v = v.strip()
            if len(v) >= 2 and v[0] == '"' and v[-1] == '"':
                v = v[1:-1]
            return f"{k}={v}"
        if len(t) >= 2 and t[0] == '"' and t[-1] == '"':
            t = t[1:-1]
        return t

    tokens = [clean_token(t) for t in tokens]

    if tokens[0].lower() == "yolo":
        yolo_exe = Path(sys.executable).with_name("yolo.exe")
        if yolo_exe.exists():
            return [str(yolo_exe), *tokens[1:]]
        return [
            sys.executable,
            "-c",
            "from ultralytics.cfg import entrypoint; entrypoint()",
            *tokens[1:],
        ]
    return tokens


def _stream_process_to_log(exec_args: list[str], env: dict[str, str], cwd: str, log_path: Path, exp_id: str) -> tuple[int, float]:
    t0 = time.time()
    heartbeat_sec = 15
    last_heartbeat = t0

    with log_path.open("w", encoding="utf-8", newline="\n") as logf:
        logf.write(f"[CMD] {subprocess.list2cmdline(exec_args)}\n")
        logf.write(f"[START] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        logf.write("[STREAM]\n")
        logf.flush()

        try:
            proc = subprocess.Popen(
                exec_args,
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=cwd,
                env=env,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except Exception as e:
            err = f"[ERROR] [{exp_id}] 启动失败: {e}"
            print(err, flush=True)
            logf.write(err + "\n")
            logf.write("[RETURN_CODE] 127\n")
            logf.write(f"[END] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            duration = round(time.time() - t0, 2)
            logf.write(f"[DURATION_SEC] {duration}\n")
            logf.flush()
            return 127, duration

        line_queue: queue.Queue[str] = queue.Queue()

        def _reader() -> None:
            if proc.stdout is None:
                return
            for raw_line in proc.stdout:
                line_queue.put(raw_line.rstrip("\r\n"))

        th = threading.Thread(target=_reader, daemon=True)
        th.start()

        while True:
            try:
                line = line_queue.get(timeout=1.0)
                msg = f"[{exp_id}] {line}"
                print(msg, flush=True)
                logf.write(line + "\n")
                logf.flush()
                last_heartbeat = time.time()
            except queue.Empty:
                now = time.time()
                if now - last_heartbeat >= heartbeat_sec:
                    elapsed = int(now - t0)
                    hb = f"[INFO] [{exp_id}] 运行中... elapsed={elapsed}s"
                    print(hb, flush=True)
                    logf.write(hb + "\n")
                    logf.flush()
                    last_heartbeat = now

            if proc.poll() is not None and line_queue.empty():
                break

        th.join(timeout=1.0)
        rc = int(proc.returncode or 0)

        ended = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        duration = round(time.time() - t0, 2)
        logf.write(f"[RETURN_CODE] {rc}\n")
        logf.write(f"[END] {ended}\n")
        logf.write(f"[DURATION_SEC] {duration}\n")
        logf.flush()

    return rc, round(time.time() - t0, 2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ablation commands in batch and summarize results")
    parser.add_argument("--plan-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    plan_dir = Path(args.plan_dir).resolve()
    project_root = Path(__file__).resolve().parents[2]
    local_ultralytics_src = project_root / "src" / "ultralytics-main"
    cmd_file = plan_dir / "run_commands.txt"
    plan_csv = plan_dir / "ablation_plan.csv"
    if not cmd_file.exists():
        raise FileNotFoundError(f"未找到命令模板: {cmd_file}")
    if not plan_csv.exists():
        raise FileNotFoundError(f"未找到实验计划: {plan_csv}")

    commands = _load_commands(cmd_file)
    exp_ids = _load_exp_ids(plan_csv)
    if not commands:
        raise RuntimeError("run_commands.txt 没有可执行命令（去掉注释后为空）")

    logs_dir = plan_dir / "run_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    results = []
    print(f"[INFO] 批跑目录: {plan_dir}")
    print(f"[INFO] 命令数: {len(commands)}")

    for i, cmd in enumerate(commands):
        exp_id = exp_ids[i] if i < len(exp_ids) else f"EXP_{i+1}"
        started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_path = logs_dir / f"{exp_id}.log"

        print(f"[INFO] [{exp_id}] 开始: {started_at}")
        exec_args = _normalize_command(cmd)
        if not exec_args:
            print(f"[WARN] [{exp_id}] 空命令，已跳过")
            continue
        exec_cmd = subprocess.list2cmdline(exec_args)
        print(f"[INFO] [{exp_id}] CMD: {exec_cmd}")

        if args.dry_run:
            ended_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            results.append(
                {
                    "exp_id": exp_id,
                    "status": "DRY_RUN",
                    "return_code": 0,
                    "duration_sec": 0.0,
                    "started_at": started_at,
                    "ended_at": ended_at,
                    "log_file": str(log_path),
                    "command": cmd,
                }
            )
            log_path.write_text("[DRY_RUN] " + exec_cmd + "\n", encoding="utf-8")
            continue

        env = os.environ.copy()
        py_path_parts = [str(local_ultralytics_src), str(project_root)]
        if env.get("PYTHONPATH"):
            py_path_parts.append(env.get("PYTHONPATH", ""))
        env["PYTHONPATH"] = os.pathsep.join(py_path_parts)

        return_code, duration = _stream_process_to_log(
            exec_args=exec_args,
            env=env,
            cwd=str(project_root),
            log_path=log_path,
            exp_id=exp_id,
        )
        ended_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        status = "OK" if return_code == 0 else "FAILED"
        print(f"[INFO] [{exp_id}] 结束: {ended_at} | status={status} | cost={duration}s")
        results.append(
            {
                "exp_id": exp_id,
                "status": status,
                "return_code": return_code,
                "duration_sec": duration,
                "started_at": started_at,
                "ended_at": ended_at,
                "log_file": str(log_path),
                "command": exec_cmd,
            }
        )

        if return_code != 0 and not args.continue_on_error:
            print(f"[ERROR] [{exp_id}] 失败，已停止后续实验。")
            break

    result_csv = plan_dir / "ablation_results.csv"
    with result_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "exp_id",
                "status",
                "return_code",
                "duration_sec",
                "started_at",
                "ended_at",
                "log_file",
                "command",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    result_md = plan_dir / "ablation_results.md"
    ok_count = sum(1 for r in results if r["status"] in {"OK", "DRY_RUN"})
    fail_count = sum(1 for r in results if r["status"] == "FAILED")
    lines = [
        "# 消融批跑结果",
        "",
        f"- 总数: {len(results)}",
        f"- 成功: {ok_count}",
        f"- 失败: {fail_count}",
        f"- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| 实验ID | 状态 | 返回码 | 耗时(s) | 日志文件 |",
        "|---|---|---:|---:|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['exp_id']} | {r['status']} | {r['return_code']} | {r['duration_sec']} | `{Path(r['log_file']).name}` |"
        )
    result_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[OK] 结果CSV: {result_csv}")
    print(f"[OK] 结果MD: {result_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
