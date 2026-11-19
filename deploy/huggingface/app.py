"""Hugging Face Gradio entrypoint for the PC28 DeepFusion Telegram bot."""
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path
import gradio as gr

ENGINE_URL = "https://github.com/xianhettkx/PC28-DeepFusion-Engine.git"
ENGINE_DIR = Path("/tmp/pc28-deepfusion-engine")
LOG_PATH = Path("/tmp/pc28-deepfusion-bot.log")
_bot_process: subprocess.Popen | None = None
_log_handle = None
_start_error: str | None = None

def prepare_engine() -> None:
    if ENGINE_DIR.exists():
        result = subprocess.run(["git", "-C", str(ENGINE_DIR), "pull", "--ff-only", "origin", "main"], capture_output=True, text=True, timeout=90, check=False)
    else:
        result = subprocess.run(["git", "clone", "--depth", "1", ENGINE_URL, str(ENGINE_DIR)], capture_output=True, text=True, timeout=180, check=False)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "Git operation failed")[-1200:])

def start_bot() -> None:
    global _bot_process, _log_handle, _start_error
    if _bot_process is not None and _bot_process.poll() is None:
        return
    if not (os.getenv("TG_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")):
        _start_error = "缺少 Space Secret：TG_TOKEN。请在 Space Settings → Variables and secrets 中添加。"
        return
    try:
        prepare_engine()
        env = os.environ.copy()
        env.setdefault("BOT_AUTO_GIT_PULL", "true")
        env.setdefault("BOT_POLL_SECONDS", "60")
        _log_handle = LOG_PATH.open("a", encoding="utf-8")
        _bot_process = subprocess.Popen([sys.executable, "-m", "bot.main"], cwd=str(ENGINE_DIR), env=env, stdout=_log_handle, stderr=subprocess.STDOUT, text=True)
        _start_error = None
    except Exception as exc:
        _start_error = f"{type(exc).__name__}: {exc}"

def status_text() -> str:
    if _start_error:
        return f"### 尚未运行\n\n{_start_error}"
    if _bot_process is None:
        return "### 尚未启动\n\n应用入口已加载，但机器人进程尚未启动。"
    code = _bot_process.poll()
    if code is None:
        return "### 运行中\n\nGradio 管理页已启动，Telegram Bot 子进程正在运行。"
    tail = ""
    try:
        if LOG_PATH.exists():
            tail = LOG_PATH.read_text(encoding="utf-8", errors="replace")[-1800:]
    except OSError:
        pass
    return f"### 进程已退出（代码：{code}）\n\n最近日志：\n\n```text\n{tail}\n```"

start_bot()
with gr.Blocks(title="PC28 DeepFusion Bot") as demo:
    gr.Markdown("# PC28 DeepFusion Bot\nTelegram 机器人运行状态面板。预测与验证数据来自 GitHub 引擎仓库中的冻结文件。\n\n**注意：** Hugging Face 免费 Space 可能休眠或重启；不保证 24/7 在线。")
    state = gr.Markdown(status_text())
    gr.Button("刷新运行状态").click(fn=status_text, outputs=state)
    gr.Markdown("Telegram 命令：`/start`、`/predict`、`/history`、`/verify`、`/status`、`/subscribe`、`/unsubscribe`、`/id`、`/admin`。")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
