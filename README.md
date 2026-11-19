# PC28 DeepFusion Engine

独立的 PC28 数据与模型工程。当前阶段先建立可审计、可重复运行的数据循环；模型训练和机器人会在数据底座稳定后接入。

## 数据循环

1. 将初始历史 CSV 放入 `data/bootstrap.csv`（列名必须为 `draw_nbr, draw_date, draw_time, draw_number, draw_num, size_type, parity_type, combination_type`）。
2. GitHub Actions 每 5 分钟尝试同步 API（GitHub schedule 并非准点保证）。
3. 每期以 `draw_nbr` 为唯一键写入 SQLite，重复数据自动忽略。
4. 导出完整归档 `data/raw_history.csv`、最近 10,000 期 `data/rolling_10000.csv` 和 `data/dataset_manifest.json`。
5. 记录同步日志，检测历史期号缺口；缺口会在 manifest 中公开，不会静默伪造数据。

## 本地运行

```bash
python -m unittest discover -s engine -p 'test_*.py' -v
python engine/sync.py
```

仅使用 Python 3.12 标准库。可通过仓库变量 `PC28_API_URL` 覆盖 API 地址。

## 当前限制 / 安全边界

- 初始 CSV 尚需放入 `data/bootstrap.csv` 后才会执行首次全量导入。
- API 字段结构尚未被本工程运行环境直接验证；解析器支持常见字段别名，若 API 返回结构不同，Actions 会明确失败而不是写入猜测数据。
- 每 5 分钟是 GitHub Actions 的请求频率，不是开奖触发器或准点承诺。
- 本项目不承诺随机开奖可预测。模型必须通过按时间顺序的样本外验证，并与随机基线比较后才允许发布。
