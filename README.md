# FinTSLib

**FinTSLib** is a deep-learning library for time-series modeling with a focus on **financial forecasting**. Built as a fork of [thuml/Time-Series-Library (TSLib)](https://github.com/thuml/Time-Series-Library), it keeps TSLib's broad model zoo and six-task experimental framework while adding finance-oriented capabilities: bundled OHLC market datasets, reversible-instance-normalized (RevIN) baseline variants, custom autoregressive/decomposition forecasters designed for non-stationary financial series, an enhanced evaluation pipeline (R², GPU memory, timing, auto-generated publication figures), and batch sweep runners for reproducible experiments.

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![Built on Time-Series-Library](https://img.shields.io/badge/Built%20on-Time--Series--Library-8A2BE2.svg)](https://github.com/thuml/Time-Series-Library)

---

## Key Features

- **Financial focus** — bundled OHLC datasets for stocks, indices, crypto, and forex, with a `yfinance` fetch tool for adding more instruments.
- **Six task types** — long-term / short-term forecasting, imputation, anomaly detection, classification, and zero-shot forecasting, each with a dedicated experiment pipeline.
- **60+ models** — 41 TSLib baseline architectures (including 7 pretrained foundation models), 19 RevIN-wrapped variants, plus the author's custom financial models and their ablation/gate variants.
- **RevIN-wrapped variants** — TSLib baselines re-packaged with Reversible Instance Normalization to handle distribution shift in non-stationary financial series.
- **Custom financial architectures** — `RevTransLSTM-AR` (autoregressive Transformer–LSTM with cross-attention and closed-loop latent feedback) and `DeReFusion` (decomposition-residual dual-branch fusion), each with a full suite of ablation variants.
- **Automatic model registry** — drop a `.py` file anywhere under `models/` and it is auto-discovered and lazily imported; no manual registration.
- **Enhanced evaluation** — R² metric, GPU peak-memory profiling, train/inference timing, parameter counts, and automatically generated publication-quality figures.
- **Batch sweep runners** — Cartesian sweeps over datasets × windows × seeds × models with MD5-keyed resume and per-experiment logging (parallel and sequential variants).
- **`yfinance` tooling** — scripted market-data download plus descriptive-statistics and result-parsing utilities.
- **CUDA & Apple-MPS support** — train and evaluate on NVIDIA GPUs or Apple Silicon (`--gpu_type mps`); Mamba models require Linux + CUDA (see [Installation](#installation)).

---

## Supported Tasks

The active task is selected with the `--task_name` flag and dispatched in `run.py`.

| Task | `task_name` flag | Experiment module |
|---|---|---|
| Long-term forecasting | `long_term_forecast` | `exp/exp_long_term_forecasting.py` |
| Short-term forecasting | `short_term_forecast` | `exp/exp_short_term_forecasting.py` |
| Imputation | `imputation` | `exp/exp_imputation.py` |
| Anomaly detection | `anomaly_detection` | `exp/exp_anomaly_detection.py` |
| Classification | `classification` | `exp/exp_classification.py` |
| Zero-shot forecasting | `zero_shot_forecast` | `exp/exp_zero_shot_forecasting.py` |

> An unrecognized `--task_name` falls back to the long-term forecasting pipeline.

---

## Model Zoo

Models live under `models/`, split into three families. Any unrecognized model name surfaces a `ValueError` listing all available models. Recommended per-model, per-paper hyperparameters are catalogued in **[`models/tslib_models/PAPER_RECOMMENDED_PARAMS.md`](./models/tslib_models/PAPER_RECOMMENDED_PARAMS.md)**.

### (a) TSLib Baselines — `models/tslib_models/`

41 baseline model files, organized by primary architecture. A few models are hybrids and touch more than one category.

<details>
<summary><b>Transformer-based</b></summary>

Autoformer, Crossformer, ETSformer, FEDformer, Informer, MultiPatchFormer, Nonstationary_Transformer, PAttn, PatchTST, Pyraformer, Reformer, TemporalFusionTransformer, TimeXer, Transformer, iTransformer, KANAD (KAN-based anomaly-detection model).

</details>

<details>
<summary><b>Linear / MLP</b></summary>

DLinear, TiDE, LightTS, TSMixer, TimeMixer, FiLM, WPMixer. *(FiLM and WPMixer also use frequency/wavelet components.)*

</details>

<details>
<summary><b>CNN / TCN</b></summary>

TimesNet, MICN, SCINet.

</details>

<details>
<summary><b>RNN</b></summary>

SegRNN.

</details>

<details>
<summary><b>Frequency / Decomposition</b></summary>

FreTS, TimeFilter, Koopa, MSGNet. *(FiLM and WPMixer overlap here.)*

</details>

<details>
<summary><b>State-space (Mamba)</b></summary>

Mamba, MambaSimple, MambaSingleLayer. *(Require Linux + CUDA — see [Installation](#installation).)*

</details>

### (b) Pretrained Foundation Models — `models/tslib_models/`

Used primarily through the `zero_shot_forecast` task.

| Model | File |
|---|---|
| Chronos | `models/tslib_models/Chronos.py` |
| Chronos2 | `models/tslib_models/Chronos2.py` |
| Moirai | `models/tslib_models/Moirai.py` |
| TimesFM | `models/tslib_models/TimesFM.py` |
| TiRex | `models/tslib_models/TiRex.py` |
| Sundial | `models/tslib_models/Sundial.py` |
| TimeMoE | `models/tslib_models/TimeMoE.py` |

### (c) RevIN-Wrapped Models — `models/revin_models/`

19 TSLib baselines each wrapped with Reversible Instance Normalization (`layers/RevIN.py`), which normalizes each input instance before encoding and denormalizes after decoding to help models cope with distribution shift in non-stationary financial series. All 19 import the shared `RevIN` layer.

<details>
<summary><b>All 19 RevIN-wrapped variants</b></summary>

**Transformer-family:** `revin-Transformer.py`, `revin-iTransformer.py`, `revin-Informer.py`, `revin-Reformer.py`, `revin-Autoformer.py`, `revin-FEDformer.py`, `revin-ETSformer.py`, `revin-TimesNet.py`, `revin-PatchTST.py`, `revin-PAttn.py`, `revin-TransformerEncoder.py`

**Linear / MLP-family:** `revin-DLinear.py`, `revin-TiDE.py`, `revin-LightTS.py`, `revin-TSMixer.py`

**RNN-family:** `revin-LSTM.py`, `revin-GRU.py`, `revin-Seq2SeqLSTM.py`

**Hybrid:** `revin-TransformerLSTM.py`

</details>

### (d) Custom Financial Models — `models/my_models/`

The author's purpose-built architectures for financial forecasting.

#### RevTransLSTM-AR — `models/my_models/revtranslstm_ar/RevTransLSTM-AR.py`

An autoregressive encoder–decoder forecaster. A Transformer encoder produces a global memory; an LSTM decodes step-by-step over `pred_len`, each step querying the encoder memory via cross-attention. It applies RevIN to encoder/decoder inputs and uses a **closed-loop latent feedback** loop: each step's prediction is projected back into latent space (a learnable `Linear(c_out, d_model)`) and residual-added to the cross-attention output to form the next LSTM input.

<details>
<summary><b>RevTransLSTM-AR ablation variants</b> (<code>revtranslstm_ar/ablation_variant/</code>)</summary>

| Variant | What it removes / changes |
|---|---|
| `RevTransLSTM-AR-woarfeedback.py` | Removes AR feedback; decoding is open-loop (feeds `attn_out` directly as next LSTM input). |
| `RevTransLSTM-AR-wocrossattention.py` | Removes cross-attention; decoder uses `lstm_out` as context, encoder output unused. |
| `RevTransLSTM-AR-wolearnablefeedback.py` | Keeps feedback structure but replaces the learnable projection with a fixed all-ones buffer. |
| `RevTransLSTM-AR-worevin.py` | Removes RevIN; inputs not instance-normalized, outputs not de-normalized. |
| `RevTransLSTM-AR-worevinandarfeedback.py` | Removes both RevIN and AR feedback (open-loop). |
| `RevTransLSTM-AR-worevinandarfeedbackandcrossattention.py` | Removes RevIN, AR feedback, and cross-attention (degraded control baseline). |

</details>

#### DeReFusion — `models/my_models/derefusion/DeReFusion.py`

A decomposition-residual dual-branch fusion model. A DLinear branch (seasonal-trend decomposition + independent linear projection) produces the base forecast; a hybrid residual branch (per-feature projection → LSTM → Transformer encoder → projection) produces a residual; the two are combined by direct additive fusion (`base + residual`), wrapped in RevIN normalize/denormalize.

<details>
<summary><b>DeReFusion ablation variants</b> (<code>derefusion/ablation_variant/</code>)</summary>

| Variant | What it tests |
|---|---|
| `DeReFusion-woDy.py` | DLinear base branch only (no residual branch / fusion). |
| `DeReFusion-woLSTM.py` | DLinear + Transformer-only residual branch + direct add. |
| `DeReFusion-woTransformer.py` | DLinear + LSTM-only residual branch + direct add. |

</details>

<details>
<summary><b>DeReFusion gate variants</b> (<code>derefusion/gate_variant/</code>)</summary>

These replace direct additive fusion with a gated fusion `(1 - gate) * base + gate * residual`.

| Variant | Gating mechanism |
|---|---|
| `DeReFusion-gatev1-volatilityaware.py` | **VolatilityAwareGate** — gate driven by the raw input's per-channel std (computed before RevIN). |
| `DeReFusion-gatev2-learnable.py` | **LearnableGate** — per-channel learnable scalar `gate = sigmoid(alpha)`; static across samples/timesteps. |
| `DeReFusion-gatev3-inputconditioned.py` | **InputConditionedGate** — sample-adaptive bottleneck gate producing per-sample/per-channel/per-timestep weights. |

</details>

<details>
<summary><b>Other custom models</b> (top level of <code>models/my_models/</code>)</summary>

- **`TransformerLSTM.py`** — encoder–decoder: a Transformer encoder builds memory from `x_enc`; an LSTM decoder is initialized with the encoder's last-step memory as `h_0` and decodes `x_dec`; a linear Seq2Seq head outputs predictions.
- **`LSTMAttention.py`** — backbone-neck-head: a multi-layer LSTM backbone, a multi-head self-attention neck, and a prediction head mapping the last step to `pred_len * c_out`.

</details>

---

## Financial Datasets

Two bundled collections under `dataset/`, each holding per-instrument daily OHLC CSVs (one file per ticker).

### `dataset/2013_2023/` — 2013-12-02 → 2023-12-29 (8 instruments)

| Type | Instruments |
|---|---|
| Stocks | AAPL (Apple), JPM (JPMorgan), TSMC (Taiwan Semiconductor) |
| Indices | GSPC (S&P 500), NDX (Nasdaq-100), SOX (PHLX Semiconductor), FTSE (FTSE 100), N225 (Nikkei 225) |

### `dataset/2016_2025/` — 2016-01-01 → 2025-12-31 (10 instruments)

| Type | Instruments |
|---|---|
| Stocks | BABA (Alibaba), NVO (Novo Nordisk), TM (Toyota) |
| Indices | GSPC (S&P 500), DJI (Dow Jones), SOX (PHLX Semiconductor) |
| Crypto | BTCUSD (Bitcoin), ETHUSD (Ethereum) |
| Forex | EURUSD (Euro/USD), USDJPY (USD/Yen) |

### CSV Schema

All files share an identical header: `date,Open,High,Low,Close`.

- `date` is `YYYY-MM-DD`; values are split/adjusted floats. **OHLC only — there is no Volume column.**
- Crypto files include weekend dates; equities, indices, and forex follow business-day calendars (weekend gaps), which is why row counts differ across instruments.

### Experiment Configuration

Financial CSVs are loaded through the `custom` data provider (`Dataset_Custom`). The batch runner wires them up as:

| Setting | Value | Meaning |
|---|---|---|
| `--data` | `custom` | use `Dataset_Custom` loader |
| `--features` | `MS` | multivariate input (all OHLC columns), single target output |
| `--target` | `Close` | predict the `Close` price (moved to the last column internally) |
| `--freq` | `b` | business-day frequency for temporal embedding |

> Note: `run.py` defaults differ (`--features M`, `--target OT`, `--freq h`, `--data ETTh1`), so the financial configuration is supplied explicitly rather than relying on the defaults.

### Fetching More Data

`tools/fetch_yfinance_data.py` downloads additional instruments via `yfinance`: it calls `yf.download(ticker, start=..., end=...)`, drops the multi-index Ticker level, renames `Date` → `date`, and slices exactly `["date", "Open", "High", "Low", "Close"]` before writing to `dataset/` — producing files that match the bundled schema. The ticker and date range are edited directly in the script.

---

## Repository Structure

```text
FinTSLib/
├── run.py                              # Main entry point — CLI, seeding, task dispatch
├── run_batch_long_term_forecast.py     # Parallel batch sweep runner (long-term forecast)
├── run_batch_zero_shot_forecast.py     # Sequential batch sweep runner (zero-shot forecast)
├── models/
│   ├── tslib_models/                   # 41 TSLib baselines + 7 foundation models
│   │   └── PAPER_RECOMMENDED_PARAMS.md # Per-model recommended hyperparameters
│   ├── revin_models/                   # 19 RevIN-wrapped baseline variants
│   └── my_models/                      # Custom financial models + ablation/gate variants
│       ├── revtranslstm_ar/            # RevTransLSTM-AR + ablation variants
│       └── derefusion/                 # DeReFusion + ablation & gate variants
├── exp/                                # 6 task pipelines (exp_*.py) + Exp_Basic registry
├── layers/                             # Reusable layers, incl. RevIN.py
├── data_provider/                      # data_factory.py, data_loader.py (Dataset_Custom, etc.)
├── dataset/
│   ├── 2013_2023/                      # 8-instrument OHLC CSVs (2013–2023)
│   └── 2016_2025/                      # 10-instrument OHLC CSVs (2016–2025)
├── utils/                              # metrics.py, visualization.py, tools.py
├── tools/                              # fetch_yfinance_data.py, descriptive_stats.py, parse_result_to_xlsx.py
├── scripts/                            # TSLib-style reproduction shell scripts (per task)
├── requirements/                       # 4-step ordered install files (reqs_1..4.txt)
├── Dockerfile                          # CUDA 12.1 / PyTorch 2.5.1 build
└── docker-compose.yml                  # dev_tslib service with GPU passthrough
```

---

## Installation

Recommended Python: **3.11**. Install proceeds in four ordered steps from `requirements/`.

```bash
# 1. PyTorch + CUDA 12.1 build (torch==2.5.1)
pip install -r requirements/reqs_1.txt

# 2. Core deps + foundation-model libraries
#    (einops, scipy, scikit-learn, pandas, sktime, transformers, huggingface_hub,
#     chronos-forecasting, tirex-ts, timesfm, gluonts, lightning, jax, ...)
pip install -r requirements/reqs_2.txt

# 3. Mamba state-space backend (pinned wheel — see note below)
pip install -r requirements/reqs_3.txt

# 4. Remaining FinTSLib deps (uni2ts, reformer_pytorch, einops, sktime, patool)
pip install -r requirements/reqs_4.txt
```

> **Platform note — Mamba is Linux + CUDA only.** Step 3 installs `mamba_ssm` from a pinned wheel
> (`mamba_ssm-2.2.6.post3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl`), which is a
> CUDA 12 / torch 2.5 / cp311 / `linux_x86_64`-only build. The **Mamba models therefore require
> Linux + CUDA and Python 3.11**, and step 3 is not installable on macOS / CPU. Every other model
> runs on either **CUDA** or **Apple MPS** (`--gpu_type mps`); skip step 3 if you do not need Mamba.

### Docker

A reproducible CUDA environment is provided.

```bash
docker compose up -d --build      # build & start the dev_tslib service
docker compose exec dev_tslib bash
```

The image is based on `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel`, installs the pinned Mamba wheel and `uni2ts`, and runs with GPU passthrough (`NVIDIA_VISIBLE_DEVICES=all`), `shm_size: 8gb`, and a `/workspace` volume.

---

## Quick Start

### (a) Long-term forecast on a bundled stock

Forecast Apple's `Close` price from OHLC inputs using DLinear:

```bash
python run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id AAPL_96_24 \
  --model DLinear \
  --data custom \
  --root_path ./dataset/2013_2023/ \
  --data_path AAPL-2013-2023.csv \
  --features MS \
  --target Close \
  --freq b \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 24 \
  --enc_in 4 --dec_in 4 --c_out 1 \
  --rand_seed 2021
```

> On Apple Silicon, add `--gpu_type mps`. (Channel dims are `4` because the CSVs carry four OHLC columns.)

### (b) Zero-shot foundation-model forecast

Run a pretrained foundation model with no training (`--is_training 0`):

```bash
python run.py \
  --task_name zero_shot_forecast \
  --is_training 0 \
  --model_id BTCUSD_zeroshot \
  --model TimesFM \
  --data custom \
  --root_path ./dataset/2016_2025/ \
  --data_path BTCUSD-2016-2025.csv \
  --features MS \
  --target Close \
  --freq b \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 7
```

> Swap `--model TimesFM` for `Chronos` or `Moirai` to try other foundation models.

### (c) Batch sweep runner

Run a full Cartesian sweep (datasets × windows × seeds × models) with resume and per-experiment logging:

```bash
# Parallel long-term forecast sweep (MAX_PARALLEL=6, seeds 2020–2024)
python run_batch_long_term_forecast.py

# Sequential zero-shot foundation-model sweep (TimesFM / Chronos / Moirai)
python run_batch_zero_shot_forecast.py
```

Both runners shell out to `run.py`, key completed experiments by an MD5 of the command, record progress in `run_batch_progress.log` and failures in `run_batch_failed.log`, and resume where they left off. The long-term runner writes per-experiment stdout/stderr into `run_batch_logs/`; the zero-shot runner streams logs live. The default model list and sweep grids are edited at the top of each script.

---

## Outputs & Evaluation

The forecasting pipeline reports an extended set of metrics and diagnostics beyond upstream TSLib.

**Metrics** (`utils/metrics.py`): MAE, MSE, RMSE, MAPE, MSPE, **R²**, and optionally **DTW** (enable with `--use_dtw`; off by default as it is time-consuming).

**Diagnostics:**
- **GPU peak-memory profiling** — `gpu_mem_peak_mb` via `torch.cuda.max_memory_allocated`.
- **Train / inference timing** — training wall time and per-sample inference speed (`inference_speed_ms`, ms/sample, guarded with `torch.cuda.synchronize()`).
- **Parameter counting** — total and trainable parameter counts.

**Auto-generated figures** (`utils/visualization.py`, 300 DPI, journal serif style), written to `test_results/<setting>/`:

| Figure | Content |
|---|---|
| `fig_prediction_curves` | N sample ground-truth vs. prediction windows with error band |
| `fig_error_analysis` | MSE-per-horizon-step bars + error-distribution histogram |
| `fig_metrics_radar` | Radar over MAE / MSE / RMSE / MAPE / MSPE / R² |
| `fig_error_heatmap` | Absolute-error heatmap (sample × horizon) |
| `fig_pred_true` | Continuous ground-truth vs. prediction curve |
| `fig_dashboard` | 4-panel summary (curve, MSE/horizon, error histogram, metrics caption) |

Figures can also be regenerated standalone via `python -m utils.visualization --input ... --output ...`. The legacy per-window `visual()` PDF dump is still emitted on every 20th test batch.

**Result layout:**
- `results/<setting>/` — `pred.npy`, `true.npy`, `metrics.npy`.
- `result_long_term_forecast.txt` — appended human-readable summary (metrics, params, timing, GPU memory).
- `test_results/<setting>/` — the auto-generated figures above.
- `checkpoints/<setting>/` — trained model checkpoints.

The result text file can be parsed into a spreadsheet with `tools/parse_result_to_xlsx.py`, which produces a `raw` sheet (one row per experiment) and a `summary` sheet (mean/std aggregated over seeds).

---

## Acknowledgements

FinTSLib is built on the excellent [Time-Series-Library (TSLib)](https://github.com/thuml/Time-Series-Library) by THUML, from which it was forked on **2026-02-24**. It inherits TSLib's model implementations, six-task experimental framework, and reproduction scripts, and extends them with finance-specific datasets, models, evaluation, and tooling. Foundation-model support builds on the upstream Chronos, Moirai (uni2ts), TimesFM, TiRex, Sundial, and TimeMoE projects.

## Contributing

This is a **personal research repository**, and external pull requests are not accepted. However, the project is open source under the MIT license and **fork-friendly** — you are welcome to fork it, adapt it for your own experiments, and build on it.

## License

Released under the **MIT License**. See [`LICENSE`](./LICENSE).

- Copyright © 2026 Chih-Chien Hsieh (FinTSLib)
- Copyright © 2021 THUML @ Tsinghua University (Time-Series-Library)
