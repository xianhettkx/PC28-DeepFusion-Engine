"""Telegram interface for the PC28 DeepFusion engine.

Run from the repository root with: python -m bot.main
Required environment: TG_TOKEN (or TELEGRAM_BOT_TOKEN).
Optional: BOT_ADMIN_IDS, BOT_POLL_SECONDS.
"""
from __future__ import annotations

import asyncio
import csv
import html
import json
import logging
import os
import socket
import ssl
import subprocess
import time
from pathlib import Path
from typing import Any

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application, ApplicationBuilder, CommandHandler, ContextTypes,
)
from telegram.request import HTTPXRequest
import httpx

ROOT = Path(__file__).resolve().parents[1]
PLAN_DIR = ROOT / "prediction" / "plans"
VALIDATION_DIR = ROOT / "prediction" / "validation"
DATA_FILE = ROOT / "data" / "rolling_10000.csv"
TRAINING_REPORT = ROOT / "logs" / "training_latest.json"
BACKTEST_REPORT = ROOT / "evaluation" / "reports" / "backtest_600.json"
BOT_DIR = ROOT / "bot"
SUBSCRIBERS_FILE = BOT_DIR / "subscribers.json"
STATE_FILE = BOT_DIR / "runtime_state.json"
CLASSES = ("大单", "小单", "大双", "小双")
LOG = logging.getLogger("pc28.bot")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        LOG.exception("Unable to read JSON file: %s", path)
        return default


def latest_plan() -> dict[str, Any] | None:
    plans = []
    for path in PLAN_DIR.glob("*.json"):
        record = read_json(path)
        if isinstance(record, dict) and record.get("frozen") is True:
            plans.append(record)
    return max(plans, key=lambda p: str(p.get("generated_at", "")), default=None)


def latest_issue() -> int | None:
    if not DATA_FILE.exists():
        return None
    latest: int | None = None
    with DATA_FILE.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                issue = int(row["draw_nbr"])
            except (KeyError, TypeError, ValueError):
                continue
            latest = issue if latest is None else max(latest, issue)
    return latest


def latest_draw() -> dict[str, str] | None:
    if not DATA_FILE.exists():
        return None
    result = None
    with DATA_FILE.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if result is None or int(row["draw_nbr"]) > int(result["draw_nbr"]):
                result = row
    return result


def validation_for(plan_id: str) -> dict[str, Any] | None:
    return read_json(VALIDATION_DIR / f"{plan_id}.json")


def active_predictions(plan: dict[str, Any], current_issue: int | None) -> list[dict[str, Any]]:
    items = plan.get("predictions", [])
    if not isinstance(items, list):
        return []
    if current_issue is None:
        return items
    return [item for item in items if int(item.get("issue", 0)) > current_issue]


def fmt_prediction(item: dict[str, Any]) -> str:
    probs = item.get("probabilities", {})
    prob_text = " / ".join(
        f"{name} {float(probs.get(name, 0)) * 100:.1f}%"
        for name in CLASSES
    )
    return (
        f"<b>{int(item['issue'])}期</b>  <b>{html.escape(str(item['prediction']))}</b>\n"
        f"<code>{html.escape(prob_text)}</code>"
    )


def fmt_plan(plan: dict[str, Any], current_issue: int | None, limit: int = 20) -> str:
    items = active_predictions(plan, current_issue)[:limit]
    if not items:
        return "当前冻结计划的预测期号已全部开奖，等待下一份计划。"
    lines = [
        "<b>PC28 · DeepFusion 冻结预测</b>",
        f"模型：<code>{html.escape(str(plan.get('model_version', 'unknown')))}</code>",
        f"计划：<code>{html.escape(str(plan.get('plan_id', 'unknown')))}</code>",
        f"最近历史期号：<code>{plan.get('last_historical_issue', '—')}</code>",
        "",
    ]
    lines.extend(fmt_prediction(item) for item in items)
    lines.extend(["", "<i>这是模型输出，不代表未来开奖保证；请勿据此进行超出承受能力的投注。</i>"])
    return "\n".join(lines)


def load_subscribers() -> set[int]:
    raw = read_json(SUBSCRIBERS_FILE, [])
    if not isinstance(raw, list):
        return set()
    result: set[int] = set()
    for value in raw:
        try:
            result.add(int(value))
        except (TypeError, ValueError):
            continue
    return result


def save_subscribers(ids: set[int]) -> None:
    BOT_DIR.mkdir(parents=True, exist_ok=True)
    temp = SUBSCRIBERS_FILE.with_suffix(".tmp")
    temp.write_text(json.dumps(sorted(ids), indent=2) + "\n", encoding="utf-8")
    temp.replace(SUBSCRIBERS_FILE)


def load_state() -> dict[str, Any]:
    state = read_json(STATE_FILE, {})
    return state if isinstance(state, dict) else {}


def save_state(state: dict[str, Any]) -> None:
    BOT_DIR.mkdir(parents=True, exist_ok=True)
    temp = STATE_FILE.with_suffix(".tmp")
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(STATE_FILE)


def admin_ids() -> set[int]:
    raw = os.getenv("BOT_ADMIN_IDS", "")
    ids: set[int] = set()
    for token in raw.replace(";", ",").split(","):
        token = token.strip()
        if token:
            try:
                ids.add(int(token))
            except ValueError:
                LOG.warning("Ignoring invalid BOT_ADMIN_IDS entry")
    return ids


def is_admin(user_id: int | None) -> bool:
    return user_id is not None and user_id in admin_ids()


async def reply(update: Update, text: str) -> None:
    if update.effective_message:
        await update.effective_message.reply_text(
            text, parse_mode=ParseMode.HTML, disable_web_page_preview=True
        )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await reply(
        update,
        "<b>PC28 DeepFusion Bot</b>\n"
        "这里展示仓库中真实冻结的模型预测和开奖验证。\n\n"
        "/predict - 当前20期预测计划\n"
        "/history - 最近开奖与验证结果\n"
        "/verify - 当前计划验证进度\n"
        "/status - 模型训练与回测状态\n"
        "/subscribe - 订阅新计划和开奖播报\n"
        "/unsubscribe - 取消订阅\n"
        "/id - 查看自己的 Telegram ID\n"
        "/help - 帮助"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await start(update, context)


async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    plan = latest_plan()
    if not plan:
        await reply(update, "仓库中还没有可读取的冻结预测计划。")
        return
    await reply(update, fmt_plan(plan, latest_issue()))


async def history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    draw = latest_draw()
    if not draw:
        await reply(update, "暂时无法读取开奖数据文件。")
        return
    issue = html.escape(str(draw.get("draw_nbr", "—")))
    number = html.escape(str(draw.get("draw_number", "—")))
    total = html.escape(str(draw.get("draw_num", "—")))
    kind = html.escape(str(draw.get("combination_type", "—")))
    lines = [f"<b>最近开奖：{issue}期</b>", f"号码：<code>{number}</code>", f"和值：<b>{total}</b>", f"分类：<b>{kind}</b>"]
    plan = latest_plan()
    if plan:
        report = validation_for(str(plan.get("plan_id", "")))
        if report:
            matches = [x for x in report.get("results", []) if int(x.get("issue", -1)) == int(draw["draw_nbr"])]
            if matches:
                item = matches[-1]
                lines.append(f"本模型预测：{html.escape(str(item['prediction']))} · {'命中' if item['hit'] else '未命中'}")
    await reply(update, "\n".join(lines))


async def verify(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    plan = latest_plan()
    if not plan:
        await reply(update, "没有冻结计划。")
        return
    report = validation_for(str(plan.get("plan_id", "")))
    if not report:
        await reply(update, "这份计划还没有可核实的开奖记录；未开奖期不会计入命中率。")
        return
    await reply(
        update,
        "<b>冻结计划验证</b>\n"
        f"计划：<code>{html.escape(str(report.get('plan_id')))}</code>\n"
        f"已验证：{int(report.get('validated_periods', 0))}/20 期\n"
        f"命中：{int(report.get('hits', 0))}\n"
        f"实际命中率：{float(report.get('hit_rate', 0)) * 100:.2f}%\n"
        f"状态：{'已完成' if report.get('completed') else '进行中'}"
    )


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    train = read_json(TRAINING_REPORT, {})
    backtest = read_json(BACKTEST_REPORT, {})
    if not isinstance(train, dict):
        train = {}
    if not isinstance(backtest, dict):
        backtest = {}
    size = int(train.get("model_size_bytes", 0) or 0)
    accuracy = backtest.get("accuracy")
    baseline = (backtest.get("baselines") or {}).get("initial_majority_accuracy")
    lines = [
        "<b>PC28 DeepFusion · 系统状态</b>",
        f"最新开奖期号：<code>{latest_issue() or '未知'}</code>",
        f"模型版本：<code>{html.escape(str(train.get('model_file', '未知')))}</code>",
        f"训练截止期号：<code>{train.get('trained_through_issue', '未知')}</code>",
        f"训练样本：<code>{train.get('train_samples', '未知')}</code>",
        f"验证样本：<code>{train.get('validation_samples', '未知')}</code>",
        f"权重大小：<code>{size:,} bytes</code>",
        f"训练状态：<code>{html.escape(str(train.get('status', '未知')))}</code>",
    ]
    if accuracy is not None:
        lines.append(f"600期回测：<b>{float(accuracy) * 100:.2f}%</b>")
    if baseline is not None:
        lines.append(f"多数类基线：<b>{float(baseline) * 100:.2f}%</b>")
    if accuracy is not None and baseline is not None:
        lines.append("评估：低于多数类基线" if float(accuracy) < float(baseline) else "评估：达到或超过多数类基线")
    lines.append("\n<i>历史回测不保证未来结果。</i>")
    await reply(update, "\n".join(lines))


async def subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    if not chat:
        return
    ids = load_subscribers()
    ids.add(chat.id)
    save_subscribers(ids)
    await reply(update, "已订阅。Bot 在线运行且数据仓库更新后，会推送新预测计划和开奖验证。")


async def unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    if not chat:
        return
    ids = load_subscribers()
    ids.discard(chat.id)
    save_subscribers(ids)
    await reply(update, "已取消订阅。")


async def id_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    await reply(update, f"你的 Telegram ID：<code>{user.id if user else '未知'}</code>")


async def admin_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not user or not is_admin(user.id):
        await reply(update, "此命令仅管理员可用。")
        return
    subscribers = len(load_subscribers())
    await reply(update, f"管理员权限正常。当前订阅人数：{subscribers}")


async def broadcast_update(app: Application, text: str) -> None:
    for chat_id in sorted(load_subscribers()):
        try:
            await app.bot.send_message(
                chat_id=chat_id, text=text, parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
            )
        except Exception:
            LOG.exception("Failed to send scheduled update to chat_id=%s", chat_id)


async def poll_repository(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Refresh the checkout, then notify subscribers about new plans and draws."""
    if os.getenv("BOT_AUTO_GIT_PULL", "true").strip().lower() in {"1", "true", "yes", "on"}:
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                ["git", "pull", "--ff-only", "origin", "main"],
                cwd=ROOT, capture_output=True, text=True, timeout=45, check=False,
            )
            if result.returncode != 0:
                LOG.warning("Repository refresh failed: %s", (result.stderr or result.stdout)[-500:])
        except (OSError, subprocess.SubprocessError):
            LOG.exception("Repository refresh failed; continuing with local data")
    state = load_state()
    plan = latest_plan()
    issue = latest_issue()
    if plan:
        plan_id = str(plan.get("plan_id", ""))
        if plan_id and state.get("last_plan_id") != plan_id:
            # Avoid broadcasting a stale historical plan on first boot.
            if "last_plan_id" in state:
                await broadcast_update(context.application, fmt_plan(plan, issue))
            state["last_plan_id"] = plan_id

    if issue is not None and issue > int(state.get("last_notified_issue", 0) or 0):
        if "last_notified_issue" in state:
            draw = latest_draw()
            if draw and int(draw.get("draw_nbr", 0)) == issue:
                kind = html.escape(str(draw.get("combination_type", "未知")))
                total = html.escape(str(draw.get("draw_num", "未知")))
                plan = latest_plan()
                verdict = "本期尚无对应预测验证记录"
                if plan:
                    report = validation_for(str(plan.get("plan_id", "")))
                    if report:
                        found = [r for r in report.get("results", []) if int(r.get("issue", -1)) == issue]
                        if found:
                            verdict = "本模型命中" if found[-1].get("hit") else "本模型未命中"
                await broadcast_update(
                    context.application,
                    f"<b>PC28 开奖更新 · {issue}期</b>\n和值：<b>{total}</b>\n分类：<b>{kind}</b>\n{verdict}",
                )
        state["last_notified_issue"] = issue
    save_state(state)


async def post_init(app: Application) -> None:
    seconds = max(30, int(os.getenv("BOT_POLL_SECONDS", "60")))
    if app.job_queue:
        app.job_queue.run_repeating(poll_repository, interval=seconds, first=10, name="repository-poll")
    else:
        LOG.warning("JobQueue unavailable; install python-telegram-bot[job-queue] to enable repository polling.")



def probe_telegram_ipv4() -> bool:
    """Check DNS, TCP and TLS independently before selecting the HTTP transport."""
    host = "api.telegram.org"
    try:
        addresses = socket.getaddrinfo(host, 443, family=socket.AF_INET, type=socket.SOCK_STREAM)
    except OSError as exc:
        LOG.warning("Telegram IPv4 DNS lookup failed: %s: %s", type(exc).__name__, exc)
        return False

    tried: set[str] = set()
    for family, socktype, proto, _, sockaddr in addresses:
        ip = sockaddr[0]
        if ip in tried:
            continue
        tried.add(ip)
        try:
            raw = socket.create_connection((ip, 443), timeout=6)
            with raw:
                raw.settimeout(8)
                with ssl.create_default_context().wrap_socket(raw, server_hostname=host):
                    LOG.info("Telegram IPv4 TCP+TLS probe succeeded (%s).", ip)
                    return True
        except (OSError, ssl.SSLError) as exc:
            LOG.warning("Telegram IPv4 probe failed (%s): %s: %s", ip, type(exc).__name__, exc)
    return False


def build_application(token: str, force_ipv4: bool = False) -> Application:
    # Use separate HTTPX clients for Bot API calls and long polling.
    request_kwargs: dict[str, Any] = {
        "connection_pool_size": 8,
        "read_timeout": 30,
        "write_timeout": 30,
        "connect_timeout": 30,
        "pool_timeout": 15,
    }

    def make_request() -> HTTPXRequest:
        kwargs = dict(request_kwargs)
        if force_ipv4:
            # PTB 21.6 supports httpx_kwargs; local_address pins the socket to IPv4.
            kwargs["httpx_kwargs"] = {
                "trust_env": True,
                "transport": httpx.AsyncHTTPTransport(local_address="0.0.0.0"),
            }
        return HTTPXRequest(**kwargs)

    if force_ipv4:
        LOG.info("Telegram HTTP transport: IPv4 selected after successful TLS probe.")
    else:
        LOG.info("Telegram HTTP transport: default HTTPX routing (IPv4 probe did not pass or proxy is configured).")

    app = (
        ApplicationBuilder()
        .token(token)
        .request(make_request())
        .get_updates_request(make_request())
        .concurrent_updates(16)
        .post_init(post_init)
        .build()
    )
    for command, callback in (
        ("start", start), ("help", help_command), ("predict", predict),
        ("history", history), ("verify", verify), ("status", status),
        ("subscribe", subscribe), ("unsubscribe", unsubscribe),
        ("id", id_command), ("admin", admin_command),
    ):
        app.add_handler(CommandHandler(command, callback))
    return app


def polling_error_callback(exc: Exception) -> None:
    LOG.warning("Telegram polling error: %s: %s", type(exc).__name__, exc)


async def telegram_main(token: str) -> None:
    proxy_configured = any(
        os.getenv(name)
        for name in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy")
    )
    force_ipv4 = False if proxy_configured else await asyncio.to_thread(probe_telegram_ipv4)
    app = build_application(token, force_ipv4=force_ipv4)
    initialized = False
    started = False
    polling = False
    try:
        # Explicit lifecycle, matching the working my-tg-zdtz project.
        await app.initialize()
        initialized = True

        # run_polling() invokes post_init internally; manual lifecycle must do it.
        await post_init(app)

        await app.start()
        started = True

        if app.updater is None:
            raise RuntimeError("Telegram Updater is unavailable.")

        await app.updater.start_polling(
            drop_pending_updates=False,
            allowed_updates=Update.ALL_TYPES,
            error_callback=polling_error_callback,
        )
        polling = True
        LOG.info("Telegram Bot connected; long polling is active.")

        await asyncio.Event().wait()
    finally:
        if polling and app.updater is not None:
            await app.updater.stop()
        if started:
            await app.stop()
        if initialized:
            await app.shutdown()


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    token = os.getenv("TG_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("Missing TG_TOKEN (or TELEGRAM_BOT_TOKEN) environment variable.")
    if not admin_ids():
        LOG.warning("BOT_ADMIN_IDS is empty; admin-only commands are disabled.")

    LOG.info("Starting PC28 DeepFusion Telegram bot (manual long-polling lifecycle).")
    attempt = 0
    while True:
        try:
            asyncio.run(telegram_main(token))
            break
        except Exception as exc:
            from telegram.error import InvalidToken
            if isinstance(exc, InvalidToken):
                LOG.exception("Telegram rejected the bot token; check TG_TOKEN.")
                raise
            attempt += 1
            delay = min(60, 5 * attempt)
            LOG.exception("Telegram startup/runtime failed; retrying in %s seconds.", delay)
            time.sleep(delay)


if __name__ == "__main__":
    main()
