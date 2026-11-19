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
- 模型训练、预测和回测核心已在本轮补齐；Hugging Face 发布与 Telegram Bot 尚未接入，不能视为已部署

## 当前模型状态

见 `logs/` 与 `evaluation/reports/`。

## 数据同步

GitHub Actions 每 5 分钟同步 https://pc28.help/api/kj.json?nbr=100（该接口文档限制 nbr 为 1–100），维护完整归档与最近 10000 期训练窗口。达到新增 20 期后才触发训练。


## 部署状态（如实记录）

- GitHub：源码与开奖同步工作流。
- 自主模型：从随机初始化训练；权重文件生成后由工作流提交，并校验大于 1,000,000 字节。
- Hugging Face：尚未创建 Space 或完成模型仓库上传；需要在模型与回测工作流通过后再接入。
- Telegram Bot：尚未实现。
- 开奖接口：/api/kj.json 的 nbr 按接口文档限制为 1–100；同步器使用 100 条并进行期号、和值和组合校验。
