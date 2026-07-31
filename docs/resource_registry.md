# Resource and Repository Registry

This document records all external repositories, frameworks, and resource dependencies audited for the **Agentic Causal Digital Twin for EURUSD** project.

| Index | Name | Repository URL | Purpose | License | Usage Status | Compatibility & Decision |
|---|---|---|---|---|---|---|
| 1 | MetaTrader 5 | PyPI / MQL5 | Official connection between Python and MT5 terminal | Proprietary | Core Dependency | Adopted: Installed via `MetaTrader5` package. Executable verified at `C:\Program Files\XM Global MT5\terminal64.exe`. |
| 2 | PyTorch | https://github.com/pytorch/pytorch | Core deep-learning framework | BSD-3-Clause | Core Dependency | Adopted: Neural network training, GRU, tensor ops, GPU/CPU execution. |
| 3 | PyTorch Geometric | https://github.com/pyg-team/pytorch_geometric | Graph neural network framework | MIT | Core Dependency | Adopted: GATv2Conv message passing, graph datasets, node/edge features. |
| 4 | Tigramite | https://github.com/jakobrunge/tigramite | Time-series causal-discovery library | GPL-3.0 | Core Dependency | Adopted: PCMCI, PCMCI+, conditional independence tests, dynamic graphs. Note: GPL-3.0 compliance required. |
| 5 | hmmlearn | https://github.com/hmmlearn/hmmlearn | Hidden Markov Models | BSD-3-Clause | Core Dependency | Adopted: Gaussian HMM for EURUSD market regime detection (forward-filtered online inference). |
| 6 | Gymnasium | https://github.com/Farama-Foundation/Gymnasium | Standard RL environment API | MIT | Core Dependency | Adopted: Observation/action spaces for EURUSD single-objective scalar reward environment. |
| 7 | Stable-Baselines3 | https://github.com/DLR-RM/stable-baselines3 | Single-objective RL baselines | MIT | Core Dependency | Adopted: PPO baseline implementation and evaluation callbacks. |
| 8 | MO-Gymnasium | https://github.com/Farama-Foundation/MO-Gymnasium | Vector-valued RL environment API | MIT | Core Dependency | Adopted: Multi-objective EURUSD environment with vector-valued rewards. |
| 9 | MORL-Baselines | https://github.com/LucasAlegre/morl-baselines | Multi-objective DRL algorithms | MIT | Core Dependency | Adopted: PGMORL / CAPQL Pareto frontier algorithms for multi-objective optimization. |
| 10 | XGBoost | https://github.com/dmlc/xgboost | Gradient boosted decision trees & SCM | Apache-2.0 | Core Dependency | Adopted: Predictive baseline, nonlinear SCM structural functions, counterfactual transition modeling. |
| 11 | NetworkX | https://github.com/networkx/networkx | Dynamic graph analysis & manipulation | BSD-3-Clause | Core Dependency | Adopted: Directed graph storage, path analysis, centrality metrics, export utilities. |
| 12 | MLflow | https://github.com/mlflow/mlflow | Experiment tracking & audit metrics | Apache-2.0 | Core Dependency | Adopted: Parameter/metric/artifact tracking, fold/seed logging, data hashes. |
| 13 | Streamlit | https://github.com/streamlit/streamlit | Local decision support UI | Apache-2.0 | Core Dependency | Adopted: Interactive dashboard for regime, graph visualization, trade proposals, counterfactual stress testing. |
| 14 | PyTorch Geometric Temporal | https://github.com/benedekrozemberczki/pytorch_geometric_temporal | Temporal graph dynamic extensions | MIT | Conditional | Conditional: Evaluated against custom PyG + GRU pipeline; used if simpler and compatible. |
| 15 | PyTorch Geometric Signed Directed | https://github.com/SherylHYX/pytorch_geometric_signed_directed | Signed directed GNN extensions | MIT | Conditional | Conditional: Optional research extension for signed causal edges after baseline standard GATv2-GRU works. |
| 16 | vectorbt | https://github.com/polakowo/vectorbt | Fast vectorized backtesting validation | GPL-3.0 | Conditional | Conditional: Fast signal validation and hyperparameter sweeps alongside authoritative custom backtester. |
| 17 | RL Baselines3 Zoo | https://github.com/DLR-RM/rl-baselines3-zoo | RL experiment management & tuning | MIT | Conditional | Reference/Tool: Hyperparameter tuning patterns and training script benchmarks. |
| 18 | FinRL-X / FinRL-Trading | https://github.com/AI4Finance-Foundation/FinRL-Trading | Modular quant trading architecture reference | Apache-2.0 | Reference | Reference: Modular separation of data, strategy, risk engine, and broker execution interface. |
| 19 | FinRL | https://github.com/AI4Finance-Foundation/FinRL | Financial DRL research framework | Apache-2.0 | Reference | Reference: Baseline DRL environment structure and training loops. |
| 20 | FinRL-Meta | https://github.com/AI4Finance-Foundation/FinRL-Meta | Financial data layer & market environments | Apache-2.0 | Reference | Reference: Market friction modeling and data abstraction standards. |
| 21 | RLPortfolio | https://github.com/CaioSBC/RLPortfolio | DRL portfolio management environment | MIT | Reference | Reference: Adapted for single-asset EURUSD target exposure and position accounting. |
| 22 | FinRobot | https://github.com/AI4Finance-Foundation/FinRobot | Financial AI-agent architecture | Apache-2.0 | Future Extension | Future: Optional financial analyst LLM workflow reference (read-only advisory layer). |
| 23 | LangGraph | https://github.com/langchain-ai/langgraph | Stateful agent workflow orchestration | MIT | Future Extension | Future: Multi-agent orchestration layer (Experimenter, Analyst, Risk-Reviewer, Critic). |
| 24 | CrewAI | https://github.com/crewAIInc/crewAI | Multi-agent collaboration framework | MIT | Future Extension | Future: Alternative multi-agent framework to be evaluated against LangGraph. |
| 25 | ABIDES | https://github.com/jpmorganchase/abides-jpmc-public | Market simulation framework | BSD-3-Clause | Future Extension | Future: Order-book simulation and high-frequency market impact research. |

---

## Reference Repository Local Storage

External reference repositories used for architecture design are kept in `_references/` (excluded from git tracking):

```text
_references/
├── FinRL-Trading/
├── FinRL/
├── FinRL-Meta/
├── RLPortfolio/
├── tigramite/
├── pytorch_geometric/
├── pytorch_geometric_temporal/
├── Gymnasium/
├── MO-Gymnasium/
├── morl-baselines/
├── stable-baselines3/
├── rl-baselines3-zoo/
├── hmmlearn/
├── xgboost/
└── vectorbt/
```
