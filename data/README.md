# 数据目录

首次启动前，请将已校验的 10,000 期原始 CSV 上传到本目录，文件名必须为 `bootstrap.csv`。列名必须包含：`draw_nbr, draw_date, draw_time, draw_number, draw_num, size_type, parity_type, combination_type`。

运行同步后会自动生成 `pc28.sqlite3`（完整结构化档案）、`raw_history.csv`（全量导出）、`rolling_10000.csv`（最近 10,000 期）和 `dataset_manifest.json`（数据范围与缺口清单）。不要手动编辑生成文件。