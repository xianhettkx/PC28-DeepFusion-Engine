---
title: PC28 DeepFusion Bot
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# PC28 DeepFusion Bot

Gradio entrypoint for the Telegram long-polling bot. Configure `TG_TOKEN` and `BOT_ADMIN_IDS` as Space Secrets/Variables. The app clones the public engine repository at startup and runs `python -m bot.main` as a child process.

Free Spaces may sleep or restart. Local subscriber state is ephemeral unless persistent storage is configured.
