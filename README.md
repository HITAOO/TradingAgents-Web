# TradingAgents-Web

> **基于 [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) 的扩展版本**，在原框架之上新增了 Web 前端界面、Alpha Vantage 数据源备用方案以及本地 OHLCV 数据缓存。

---

## 本版本的主要改动

### 1. Gradio Web 前端（`web/`）

原项目仅提供命令行界面（CLI）。本版本新增了一套完整的 Web 界面，无需命令行即可使用。

- 实时流式输出：分析过程中每个 Agent 的输出逐步显示，无需等待全部完成
- 分析进度面板：按「分析师 → 研究员 → 交易员 → 风控 → 组合管理」五个阶段展示进展
- 报告导出：支持浏览器下载 `.md` 文件，也可指定本地目录自动保存
- 简洁字体：统一使用系统无衬线字体（`-apple-system`、`PingFang SC`、`Segoe UI` 等），去除装饰性字体

**启动方式：**
```bash
python -m web.run
# 浏览器访问 http://localhost:7860
```

---

### 2. Alpha Vantage 数据源备用（yfinance 限流时自动切换）

原项目仅使用 yfinance 获取市场数据。Yahoo Finance 频繁限流，导致分析中断。

本版本的改动：
- `yf_retry()`：对 yfinance 的限流错误（HTTP 429）、网络超时（curl error 28）、静默返回 `None` 均实现指数退避重试
- 重试耗尽后统一抛出 `YFRateLimitError`，触发 `interface.py` 中的 Vendor 切换逻辑
- 自动 fallback 到 **Alpha Vantage**，覆盖股价、技术指标、基本面、财报、新闻、内幕交易等全部数据接口

**配置 Alpha Vantage API Key（`.env` 文件）：**
```bash
ALPHA_VANTAGE_API_KEY=your_key_here
```
免费 Key 可在 [alphavantage.co](https://www.alphavantage.co/support/#api-key) 申请。

---

### 3. 本地 OHLCV 数据缓存

原项目每次请求股价数据都直接调用 API，频繁触发限流。

本版本在 yfinance 和 Alpha Vantage 两个数据源均实现了持久化缓存：

- **缓存路径**：`~/.tradingagents/cache/{SYMBOL}-YFin-data.csv` / `{SYMBOL}-AV-data.csv`
- **缓存逻辑**：请求前检查本地是否已有覆盖目标日期的数据；有则直接读取，不调 API；无则拉取完整历史并写入缓存
- **回测友好**：同一股票的历史数据只需下载一次，后续所有历史日期均复用本地文件
- **按需刷新**：当请求日期超出缓存范围时，自动重新拉取并更新缓存

---

## 安装与使用

### 克隆并安装

```bash
git clone https://github.com/HITAOO/TradingAgents-Web.git
cd TradingAgents-Web

conda create -n tradingagents python=3.13
conda activate tradingagents

pip install .
```

### 配置 API Key

复制示例文件并填入你的 Key：
```bash
cp .env.example .env
```

```bash
# .env
OPENAI_API_KEY=...           # OpenAI (GPT)
GOOGLE_API_KEY=...           # Google (Gemini)
ANTHROPIC_API_KEY=...        # Anthropic (Claude)
DEEPSEEK_API_KEY=...         # DeepSeek
ALPHA_VANTAGE_API_KEY=...    # Alpha Vantage（数据源备用，强烈建议配置）
# 更多 Provider 见 .env.example
```

### 启动 Web 界面

```bash
python -m web.run
```

浏览器访问 `http://localhost:7860`

### 使用原版 CLI

```bash
tradingagents
# 或
python -m cli.main
```

### Python API 调用

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "openai"
config["deep_think_llm"] = "gpt-4o"
config["quick_think_llm"] = "gpt-4o-mini"

ta = TradingAgentsGraph(debug=True, config=config)
_, decision = ta.propagate("AAPL", "2025-01-15")
print(decision)
```

---

## 项目结构

```
TradingAgents-Web/
├── web/                        # Web 前端（本版本新增）
│   ├── gradio_app.py           # Gradio UI 定义
│   ├── stream_handler.py       # 流式输出处理
│   ├── app.py                  # 分析任务调度
│   ├── config_builder.py       # 配置构建
│   └── run.py                  # 入口
├── tradingagents/
│   └── dataflows/
│       ├── stockstats_utils.py # yf_retry + OHLCV 缓存（本版本修改）
│       ├── y_finance.py        # 复用缓存 + YFRateLimitError 传播（本版本修改）
│       ├── yfinance_news.py    # YFRateLimitError 传播（本版本修改）
│       ├── alpha_vantage_stock.py  # OHLCV 缓存（本版本修改）
│       └── interface.py        # Vendor 切换逻辑（本版本修改）
└── ...
```

---

## 原项目

本项目基于 **TradingAgents** 开发：

- 论文：[TradingAgents: Multi-Agents LLM Financial Trading Framework](https://arxiv.org/abs/2412.20138)
- 原始仓库：[TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents)
- 作者：Yijia Xiao, Edward Sun, Di Luo, Wei Wang

```
@misc{xiao2025tradingagentsmultiagentsllmfinancial,
      title={TradingAgents: Multi-Agents LLM Financial Trading Framework},
      author={Yijia Xiao and Edward Sun and Di Luo and Wei Wang},
      year={2025},
      eprint={2412.20138},
      archivePrefix={arXiv},
      primaryClass={q-fin.TR},
      url={https://arxiv.org/abs/2412.20138},
}
```

> 免责声明：本项目仅供研究和学习使用，不构成任何投资建议。
