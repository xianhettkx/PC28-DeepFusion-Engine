# Xhe PC28 DeepFusion AI Training Engine

自主研发的 PC28 神经网络训练系统。从随机初始化开始，使用真实历史开奖数据训练，严格时间顺序验证。

**重要声明**：PC28 开奖具有强随机性。历史统计规律不代表未来必然延续。本项目不承诺任何稳定预测能力，所有结果均如实报告，并与随机/频率基线对比。

## 项目结构

```
.
├── data/                  # 数据（由 sync 维护）
├── models/                # 模型权重与配置
├── training/              # 训练脚本
├── features/              # 特征工程
├── evaluation/            # 回测与评估
├── prediction/            # 20期冻结预测
├── storage/               # 模型版本管理
├── bot/                   # Telegram Bot
├── scripts/               # 工具脚本
├── .github/workflows/     # CI / 数据同步 / 训练触发
└── tests/
```

## 快速开始

```bash
pip install -r requirements.txt
python -m training.train_deepfusion          # 训练核心模型
python -m evaluation.backtest_600            # 600期严格回测
python -m prediction.generate_plan           # 生成20期预测计划
```

## 核心特性

- 自主设计 DeepFusion 混合架构（LSTM 序列编码 + 统计特征融合）
- 严格时间顺序：禁止未来数据泄漏、禁止随机打乱
- 600 期滚动样本外回测
- 权重文件 >1 MB（真实训练产生）
- 每新增 20 期触发训练检查，仅验证通过才升级正式模型
- 20 期预测计划冻结后不可修改
- 支持 Hugging Face 存储与 Telegram Bot

## 当前模型状态

见 `logs/` 与 `evaluation/reports/`。

## 数据同步

已有 GitHub Actions 每 5 分钟同步 https://pc28.help/api/kj.json?nbr=500，维护完整归档与最近 10000 期训练窗口。
