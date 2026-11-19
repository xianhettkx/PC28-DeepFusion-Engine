# Telegram Bot

## What it does

The bot reads the repository's existing frozen prediction plans, training report,
600-draw backtest report, validation reports, and synchronized CSV data. It does
not train a second model and never invents predictions.

## Commands

- `/start`, `/help`: command guide
- `/predict`: remaining entries in the latest frozen 20-draw plan
- `/history`: latest actual draw and matching validation result, if available
- `/verify`: verified count, hits, and hit rate for the latest plan
- `/status`: model version, sample counts, checkpoint size, and backtest vs baseline
- `/subscribe`, `/unsubscribe`: opt in/out of update broadcasts
- `/id`: show your Telegram numeric ID
- `/admin`: verify configured administrator access

## Configuration

Set environment variables on the host; never commit secrets:

- `TG_TOKEN` (or `TELEGRAM_BOT_TOKEN`): token from BotFather
- `BOT_ADMIN_IDS`: comma-separated numeric Telegram user IDs
- `BOT_POLL_SECONDS`: repository polling interval; minimum 30, default 60
- `BOT_AUTO_GIT_PULL`: automatically run `git pull --ff-only origin main` before each poll; default `true`

## Run

From the repository root:

```bash
python -m pip install -r requirements.txt
python -m bot.main
```

This is a long-polling process and must run continuously on an always-on host.
GitHub Actions is not a suitable long-running bot host. The working directory
must be a fresh or periodically updated checkout of this repository so the bot
can read the newest CSV, plan, and validation files. Subscriber and runtime state
are stored under `bot/`; use persistent storage or back them up when redeploying.

## Safety and reliability

- Only frozen plans on disk are displayed.
- Only actual synchronized draw records are used for history/verification.
- No credentials or chat IDs are logged.
- Subscriber delivery failures are logged without stopping the polling loop.
- PC28 is random; backtest metrics are descriptive, not guarantees.
