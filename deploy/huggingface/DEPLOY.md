# Hugging Face Space deployment

1. Create or rename a Gradio Space named `pc28-deepfusion-bot`.
2. Upload the contents of this directory to the Space repository root (`app.py`, `requirements.txt`, `README.md`).
3. In Space Settings → Variables and secrets, add Secret `TG_TOKEN` and Variable `BOT_ADMIN_IDS` (comma-separated Telegram numeric user IDs).
4. Restart the Space and open its app page. The status panel should report the bot process as running.
5. Open the Telegram bot and send `/start`, then `/status`.

Never put the Telegram token in a source file or GitHub commit. If the old token was ever committed publicly, revoke it with BotFather before using the replacement.

The Space uses the public GitHub engine repository and pulls current `main` at startup. The bot also refreshes it periodically. Free Spaces may sleep/restart; subscribers and runtime notification state currently live in the ephemeral checkout and can be lost on restart.
