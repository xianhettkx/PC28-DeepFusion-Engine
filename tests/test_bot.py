import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bot import main as bot


class BotFormattingTests(unittest.TestCase):
    def test_prediction_format_escapes_html_and_shows_probabilities(self):
        item = {
            "issue": 123,
            "prediction": "<script>",
            "probabilities": {"大单": 0.25, "小单": 0.25, "大双": 0.25, "小双": 0.25},
        }
        text = bot.fmt_prediction(item)
        self.assertIn("&lt;script&gt;", text)
        self.assertNotIn("<script>", text)
        self.assertIn("大单 25.0%", text)

    def test_active_predictions_only_returns_future_issues(self):
        plan = {"predictions": [{"issue": 10}, {"issue": 11}, {"issue": 12}]}
        self.assertEqual([x["issue"] for x in bot.active_predictions(plan, 10)], [11, 12])

    def test_subscriber_file_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "subscribers.json"
            with patch.object(bot, "SUBSCRIBERS_FILE", path):
                bot.save_subscribers({123, 456})
                self.assertEqual(bot.load_subscribers(), {123, 456})


if __name__ == "__main__":
    unittest.main()
