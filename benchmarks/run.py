#!/usr/bin/env python3
"""
Benchmark runner for java-pico Agent.
Usage:
    python3 benchmark/run.py                          # 列出所有任务
    python3 benchmark/run.py config_value_modify      # 跑单个任务（手动模式）
    python3 benchmark/run.py --all                    # 跑全部任务
"""

import json
import shutil
import subprocess
import sys
import os
import tempfile
import datetime
from pathlib import Path

BENCHMARK_FILE = Path(__file__).parent / "coding_tasks.json"
ROOT_DIR = Path(__file__).parent.parent
WORKSPACE_DIR = ROOT_DIR / "benchmarks" / "workspace"
SESSION_DIR = None  # --all 时设置为单次会话目录


def load_tasks():
    with open(BENCHMARK_FILE) as f:
        return json.load(f)


def find_task(task_id):
    for t in load_tasks():
        if t["id"] == task_id:
            return t
    return None


def list_tasks():
    tasks = load_tasks()
    print(f"\n{'ID':<35} {'Category':<20} {'Budget':<8}  Description")
    print("-" * 90)
    for t in tasks:
        desc = t["prompt"][:60] + "..."
        print(f"{t['id']:<35} {t['category']:<20} {t['step_budget']:<8}  {desc}")
    print(f"\nTotal: {len(tasks)} tasks\n")


def setup_fixture(task):
    """复制 fixture 到会话目录下的独立任务目录"""
    src = ROOT_DIR / task["fixture_repo"]
    if not src.exists():
        print(f"  ❌ Fixture not found: {src}")
        return None

    dst = SESSION_DIR / task["id"]
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.is_dir():
            shutil.copytree(item, dst / item.name, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst / item.name)
    return dst


def run_verifier(workspace, task):
    """在 workspace 里执行验证脚本"""
    try:
        result = subprocess.run(
            task["verifier"],
            shell=True,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "", "verifier timed out"
    except Exception as e:
        return False, "", str(e)


def run_task(task, auto=False):
    print(f"\n{'='*60}")
    print(f"  Task: {task['id']}")
    print(f"  Category: {task['category']}")
    print(f"  Budget: {task['step_budget']} steps")
    print(f"{'='*60}")
    print(f"\n  Prompt: {task['prompt']}")
    print(f"\n  Allowed tools: {', '.join(task['allowed_tools'])}")

    workspace = setup_fixture(task)
    if not workspace:
        return {"id": task["id"], "status": "error", "reason": "fixture_not_found"}

    print(f"\n  Fixture copied to: {workspace}")

    if auto:
        # 自动模式：管道传 prompt，非交互执行
        jar_path = ROOT_DIR / "target" / "ai-helper-0.0.1-SNAPSHOT.jar"
        if not jar_path.exists():
            print(f"  ❌ Jar not found at {jar_path}")
            return {"id": task["id"], "status": "error", "reason": "jar_not_found"}

        prompt_escaped = task['prompt'].replace('"', '\\"')
        cmd = f"java -jar {jar_path} --spring.profiles.active=cli --code.workspace={workspace} --benchmark-workspace \"{workspace}\" --benchmark \"{prompt_escaped}\" 2>&1"
        print(f"  Running agent...")
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        print(f"  Agent exited (stdout: {len(proc.stdout)} chars)")

        # 读取 agent 自己的 report.json
        runs_dir = workspace / ".pico" / "runs"
        task_result = {"id": task["id"], "status": "pending", "category": task["category"], "agent_output_size": len(proc.stdout)}
        if runs_dir.exists():
            run_dirs = sorted(runs_dir.iterdir())
            if run_dirs:
                report_file = run_dirs[-1] / "report.json"
                if report_file.exists():
                    with open(report_file) as rf:
                        report = json.load(rf)
                        task_result["agent_report"] = report
                        duration = report.get("run_duration_ms", 0)
                        print(f"     report: {report.get('status','?')} / "
                              f"{report.get('tool_steps',0)} tools / "
                              f"attempts={report.get('attempts',0)} / "
                              f"{duration}ms")
                else:
                    print(f"     (no report.json at {report_file})")
            else:
                print(f"     (no run dirs in {runs_dir})")
        else:
            print(f"     (no .pico/runs at {runs_dir})")
    else:
        # 手动模式
        print(f"\n  {'─'*50}")
        print(f"  STEP: Run the agent manually in this directory,")
        print(f"  then type 'done' to verify, or 'skip' to skip.")
        print(f"  {'─'*50}")
        while True:
            cmd = input("\n  > ").strip()
            if cmd == "done":
                break
            elif cmd == "skip":
                return {"id": task["id"], "status": "skipped"}
            elif cmd == "dir":
                print(f"  {workspace}")
            elif cmd == "ls":
                for f in sorted(workspace.iterdir()):
                    print(f"  {'📁' if f.is_dir() else '📄'} {f.name}")
            else:
                print(f"  Unknown command. Try: done / skip / dir / ls")

    passed, stdout, stderr = run_verifier(workspace, task)

    if auto:
        result = task_result
        result["status"] = "passed" if passed else "failed"
        result["stdout"] = stdout
        result["stderr"] = stderr
        result["prompt"] = task["prompt"]
    else:
        result = {
            "id": task["id"],
            "status": "passed" if passed else "failed",
            "category": task["category"],
            "stdout": stdout,
            "stderr": stderr,
            "prompt": task["prompt"],
        }

    if passed:
        print(f"\n  ✅ PASSED: {stdout}")
    else:
        print(f"\n  ❌ FAILED")
        if stdout: print(f"     stdout: {stdout}")
        if stderr: print(f"     stderr: {stderr}")

    return result


def run_all(auto=False):
    print(f"\n📁 Session: {SESSION_DIR}\n")
    results = []
    for t in load_tasks():
        results.append(run_task(t, auto=auto))
    print_summary(results)
    return results


def print_summary(results):
    passed = sum(1 for r in results if r["status"] == "passed")
    failed = sum(1 for r in results if r["status"] == "failed")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errors = sum(1 for r in results if r["status"] == "error")

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Total:  {len(results)}")
    print(f"  Passed: {passed} ✅")
    print(f"  Failed: {failed} ❌")
    print(f"  Skipped: {skipped} ⏭")
    print(f"  Errors: {errors} ⚠️")
    print(f"  Pass rate: {passed}/{len(results) - skipped} ({passed*100//max(1, len(results)-skipped)}%)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        list_tasks()
        sys.exit(0)

    args = sys.argv[1:]
    auto = "--auto" in args
    output_file = None
    filtered = []
    skip_next = False
    for i, a in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if a == "--auto":
            continue
        if a == "--output" and i + 1 < len(args):
            output_file = args[i + 1]
            skip_next = True
            continue
        filtered.append(a)
    args = filtered

    results = None

    # 统一会话目录：一次 run.py 调用的所有任务都在同一个目录下
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    SESSION_DIR = WORKSPACE_DIR / f"run_{timestamp}"
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Session: {SESSION_DIR}\n")

    if not args:
        list_tasks()
    elif args[0] == "--all":
        results = run_all(auto=auto)
    else:
        task = find_task(args[0])
        if task:
            r = run_task(task, auto=auto)
            results = [r]
        else:
            print(f"Task '{args[0]}' not found.")
            list_tasks()

    if results and output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")
