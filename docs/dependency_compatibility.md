# Dependency Stack Compatibility Report

**Timestamp UTC**: 2026-07-31T23:17:21.969862+00:00
**Python Runtime**: 3.13.4 (tags/v3.13.4:8a526ec, Jun  3 2025, 17:46:04) [MSC v.1943 64 bit (AMD64)]
**Overall Status**: `PASS`

## Package Version Inventory

| Package | Installed | Version |
|---|---|---|
| `MetaTrader5` | Yes | `5.0.6070` |
| `numpy` | Yes | `2.5.1` |
| `pandas` | Yes | `2.3.3` |
| `pyarrow` | Yes | `24.0.0` |
| `scipy` | Yes | `1.18.0` |
| `sklearn` | Yes | `1.9.0` |
| `statsmodels` | Yes | `0.14.6` |
| `hmmlearn` | Yes | `0.3.3` |
| `tigramite` | Yes | `installed` |
| `xgboost` | Yes | `3.3.0` |
| `torch` | Yes | `2.13.0+cpu` |
| `torch_geometric` | Yes | `2.8.0.post1` |
| `gymnasium` | Yes | `1.3.0` |
| `stable_baselines3` | Yes | `2.9.0` |
| `mo_gymnasium` | Yes | `1.3.2` |
| `networkx` | Yes | `3.6.1` |
| `mlflow` | Yes | `3.15.0` |
| `streamlit` | Yes | `1.60.0` |
| `morl_baselines` | No | `morl_baselines pycddlib C-extension build constraint on Windows (gmp.h missing)` |

## Runtime Verification Tests

### torch
- Status: `PASS`
- matmul_shape: `[2, 4]`
- cuda_available: `False`
- device: `cpu`
- cuda_version: `None`

### torch_geometric
- Status: `PASS`
- gatv2_out_shape: `[2, 2]`

### hmmlearn
- Status: `PASS`
- n_components: `2`
- sample_probs_shape: `[5, 2]`

### tigramite
- Status: `PASS`
- val_matrix_shape: `[3, 3, 3]`
- p_matrix_shape: `[3, 3, 3]`

### gymnasium
- Status: `PASS`
- env_check: `PASSED`

### stable_baselines3
- Status: `PASS`
- trained_steps: `128`
- predicted_action: `-0.32808995246887207`

### mo_gymnasium
- Status: `PASS`
- reward_dim: `3`
- obs_shape: `[7]`

