# RevTransLSTM-AR

**Official code for the paper**
**_A Fair Benchmark of Deep Models for Non-Stationary Stock Price Forecasting: RevTransLSTM-AR as a Complexity Probe_**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![Built on Time-Series-Library](https://img.shields.io/badge/Built%20on-Time--Series--Library-8A2BE2.svg)](https://github.com/thuml/Time-Series-Library)

This repository contains everything needed to reproduce the experiments in the paper: the proposed **RevTransLSTM-AR** model and its ablation variants, the **RevIN-wrapped baseline models** they are compared against, the **daily OHLC stock/index datasets**, and the **shared training and evaluation pipeline** that makes the comparison fair.

---

## Table of Contents

- [Overview](#overview)
- [What makes the benchmark _fair_](#what-makes-the-benchmark-fair)
- [RevTransLSTM-AR — the complexity probe](#revtranslstm-ar--the-complexity-probe)
- [Benchmark models](#benchmark-models)
- [Datasets](#datasets)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Reproducing the benchmark](#reproducing-the-benchmark)
- [Evaluation and outputs](#evaluation-and-outputs)
- [Adding a model](#adding-a-model)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)
- [Contributing](#contributing)

---

## Overview

Daily stock and index prices are strongly **non-stationary**: their mean and variance drift over time and the raw `Close` series carries a unit root (see [`utils/ADFtest.py`](./utils/ADFtest.py)). A common reflex is to throw progressively heavier architectures at the problem — deep Transformers, hybrid recurrent–attention stacks, autoregressive decoders, learnable feedback. This paper asks a simpler question:

> **On non-stationary stock prices, does added architectural complexity actually buy forecasting accuracy — once every model is given the same fair chance?**

To answer it, the repo provides two things:

1. A **fair benchmark** — a single data pipeline, normalization scheme, training protocol, and metric set shared by every model, so differences in results reflect the **backbone**, not the harness.
2. **RevTransLSTM-AR as a complexity probe** — a deliberately complex forecaster (Transformer encoder → autoregressive LSTM decoder → cross-attention → closed-loop latent feedback, all wrapped in RevIN) shipped with a full set of **ablation variants** that remove one component at a time. Sliding RevTransLSTM-AR down its own ablation ladder, and comparing it against simpler RevIN baselines, isolates how much each layer of complexity contributes.

The repository is a focused fork of [thuml/Time-Series-Library (TSLib)](https://github.com/thuml/Time-Series-Library): it keeps TSLib's experiment harness and reuses its layer library, but trims the model set down to exactly what the paper studies.

---

## What makes the benchmark _fair_

Every model in this repository is evaluated under an **identical protocol**, so accuracy differences are attributable to the architecture and not to incidental advantages.

| Dimension | Shared setting |
|---|---|
| **Data loader** | `Dataset_Custom` ([`data_provider/data_loader.py`](./data_provider/data_loader.py)) for all models |
| **Splits** | Chronological **70% / 10% / 20%** train / validation / test; validation and test windows are extended back by `seq_len` for lookback context |
| **Scaling** | `StandardScaler` **fit on the training split only**, then applied to all splits |
| **Normalization** | **RevIN** (Reversible Instance Normalization) applied to _every_ model — the proposed model and all baselines — so the comparison isolates backbone complexity rather than the normalization scheme |
| **Inputs / target** | `--features MS`: all four OHLC columns as input, single `Close` target (`--target Close`) |
| **Training** | Adam optimizer, MSE loss, early stopping on validation loss (`--patience`), learning-rate schedule (`--lradj`) |
| **Seeds** | Controlled by `--rand_seed` (default `2021`); the seed is embedded in each run's `setting` string so multi-seed runs are tracked separately |
| **Metrics** | MAE, MSE, RMSE, MAPE, MSPE, R² (optionally DTW), plus parameter count, train time, inference ms/sample, and GPU peak memory |

> **Reproducibility note.** `run.py` seeds Python `random`, NumPy, and PyTorch (`torch.manual_seed`). It does **not** set `torch.backends.cudnn.deterministic`, so results on CUDA are reproducible up to cuDNN non-determinism. Run several seeds (e.g. `2020`–`2024`) and report mean ± std, as the paper does.

---

## RevTransLSTM-AR — the complexity probe

**File:** [`models/revtranslstm_ar/RevTransLSTM-AR.py`](./models/revtranslstm_ar/RevTransLSTM-AR.py) · **`--model RevTransLSTM-AR`**

RevTransLSTM-AR is an autoregressive encoder–decoder. A Transformer encoder builds a global memory of the input window; an LSTM then decodes the horizon **one step at a time**, querying that memory through cross-attention and feeding each prediction back into the latent state through a learnable projection.

**Forward pass** (`B` = batch, `D` = `d_model`):

```
x_enc [B, seq_len, 4]                      # OHLC input window
  │
  ├─ RevIN(norm)                           # per-instance normalize (stats from x_enc; reused for x_dec)
  ├─ DataEmbedding → Transformer Encoder ──► enc_out [B, seq_len, D]   (cross-attention memory)
  │
  └─ decoder seed = last embedded label-token step  ──► lstm_input [B, 1, D]
        repeat for t = 1 … pred_len:
            lstm_out, hidden = LSTM(lstm_input, hidden)          # [B, 1, D]
            attn_out         = CrossAttention(q=lstm_out,        # query  = LSTM state
                                              k=v=enc_out)        # memory = encoder output
            pred_t           = Linear(D → c_out)(attn_out)       # [B, 1, c_out]   ← collected
            feedback         = Linear(c_out → D)(pred_t)         # closed-loop AR latent feedback
            lstm_input       = attn_out + feedback               # residual-add → next step
  │
  ├─ concat predictions ──► [B, pred_len, c_out]
  └─ RevIN(denorm) ──────► forecast in original price scale
```

The two design choices that define the model — and that the ablations probe — are:

- **Cross-attention decoding:** the LSTM does not decode in isolation; at every step it attends back to the full encoder memory.
- **Closed-loop autoregressive feedback:** each prediction is projected back into the `d_model` latent space (a learnable `Linear(c_out, d_model)`) and **residual-added** to the cross-attention output to form the next LSTM input — a learnable feedback loop rather than open-loop teacher forcing.

RevIN here is an **inline** Reversible Instance Normalization (per-sample, per-channel normalization over the time axis with detached, invertible statistics and affine parameters); a standalone equivalent also lives at [`layers/RevIN.py`](./layers/RevIN.py) and is the one used by all baselines.

### Ablation variants

**Folder:** [`models/revtranslstm_ar/ablation_variant/`](./models/revtranslstm_ar/ablation_variant/). Each is a valid `--model` name (filename without `.py`).

| `--model` | Removes / changes | Resulting behavior |
|---|---|---|
| `RevTransLSTM-AR-woarfeedback` | Closed-loop AR feedback (`out_proj`) | Open-loop decode: `lstm_input = attn_out`; RevIN + cross-attention kept |
| `RevTransLSTM-AR-wocrossattention` | Cross-attention | `attn_out = lstm_out`; encoder runs but its memory is unused; feedback kept |
| `RevTransLSTM-AR-wolearnablefeedback` | _Learnability_ of the feedback projection | Feedback kept but fixed: `Linear` replaced by a constant all-ones buffer |
| `RevTransLSTM-AR-worevin` | RevIN | Raw-scale I/O, no distribution-shift correction; everything else kept |
| `RevTransLSTM-AR-worevinandarfeedback` | RevIN **+** AR feedback | Raw-scale, open-loop; encoder + cross-attention kept |
| `RevTransLSTM-AR-worevinandarfeedbackandcrossattention` | RevIN **+** AR feedback **+** cross-attention | Degenerate LSTM-only baseline (encoder unused, no feedback, no denorm) |

---

## Benchmark models

The comparison set is **eight RevIN-wrapped baselines** in [`models/`](./models/). Every baseline shares the same RevIN normalize/denormalize wrapping as RevTransLSTM-AR, so they sit on a clean **complexity ladder** that isolates backbone capacity:

| Complexity tier | `--model` | Backbone |
|---|---|---|
| **Linear / MLP** | `revin-DLinear` | Series decomposition + per-component linear maps |
| | `revin-TiDE` | Residual-MLP (dense) encoder–decoder |
| **Recurrent (RNN)** | `revin-LSTM` | LSTM encoder → direct multi-step projection |
| | `revin-GRU` | GRU encoder → direct multi-step projection |
| | `revin-Seq2SeqLSTM` | LSTM encoder → autoregressive LSTM decoder |
| **Transformer** | `revin-Transformer` | Vanilla encoder–decoder, full attention |
| | `revin-Informer` | ProbSparse attention + distilling |
| | `revin-Autoformer` | Decomposition + AutoCorrelation attention |

> Names are case-sensitive and the hyphen is part of the model name (e.g. `--model revin-LSTM`). All eight import the shared [`layers/RevIN.py`](./layers/RevIN.py).

---

## Datasets

Eight daily **OHLC** series under [`dataset/`](./dataset/), spanning **2013-12-02 → 2023-12-29**. All files share the header `date,Open,High,Low,Close` (`date` is `YYYY-MM-DD`; values are split/adjusted floats). **There is no Volume column.** Row counts differ because each market follows its own trading calendar.

| Ticker | Instrument | Type | Rows |
|---|---|---|---|
| [AAPL](./dataset/AAPL-2013-2023.csv) | Apple Inc. | Stock | 2537 |
| [JPM](./dataset/JPM-2013-2023.csv) | JPMorgan Chase | Stock | 2537 |
| [TSMC](./dataset/TSMC-2013-2023.csv) | Taiwan Semiconductor | Stock | 2462 |
| [GSPC](./dataset/GSPC-2013-2023.csv) | S&P 500 | Index | 2537 |
| [NDX](./dataset/NDX-2013-2023.csv) | Nasdaq-100 | Index | 2537 |
| [SOX](./dataset/SOX-2013-2023.csv) | PHLX Semiconductor | Index | 2537 |
| [FTSE](./dataset/FTSE-2013-2023.csv) | FTSE 100 | Index | 2544 |
| [N225](./dataset/N225-2013-2023.csv) | Nikkei 225 | Index | 2464 |

**Non-stationarity.** [`utils/ADFtest.py`](./utils/ADFtest.py) runs the Augmented Dickey–Fuller test (via `statsmodels` and `arch`) to quantify the unit-root behavior of each series — the empirical motivation for applying RevIN to every model in the benchmark.

**Loading convention.** Stock CSVs are read through the `custom` provider with multivariate-input / single-target settings:

| Flag | Value | Meaning |
|---|---|---|
| `--data` | `custom` | use `Dataset_Custom` |
| `--features` | `MS` | all OHLC columns in, single target out |
| `--target` | `Close` | predict `Close` (moved to the last column internally) |
| `--freq` | `b` | business-day frequency for the temporal embedding |
| `--enc_in` / `--dec_in` / `--c_out` | `4` / `4` / `1` | 4 OHLC inputs, 1 target output |

> `run.py` inherits TSLib's defaults (`--data ETTh1`, `--features M`, `--target OT`, `--freq h`), so the financial configuration must be passed explicitly, as shown below.

---

## Repository structure

```text
RevTransLSTM-AR/
├── run.py                       # Single entry point — CLI, seeding, device & task dispatch
├── models/
│   ├── revin-Autoformer.py      # 8 RevIN-wrapped baselines (revin-*.py)
│   ├── revin-DLinear.py
│   ├── revin-GRU.py
│   ├── revin-Informer.py
│   ├── revin-LSTM.py
│   ├── revin-Seq2SeqLSTM.py
│   ├── revin-TiDE.py
│   ├── revin-Transformer.py
│   └── revtranslstm_ar/
│       ├── RevTransLSTM-AR.py            # proposed model
│       └── ablation_variant/            # 6 ablation variants
├── exp/                         # Task pipelines + Exp_Basic model registry
│   ├── exp_basic.py             # auto-discovers models under models/
│   └── exp_long_term_forecasting.py     # the pipeline used by the paper
├── layers/                      # Reusable layers (RevIN, attention, embeddings, …)
├── data_provider/               # data_factory.py, data_loader.py (Dataset_Custom)
├── dataset/                     # 8 OHLC CSVs (2013–2023)
├── utils/                       # metrics.py, visualization.py, ADFtest.py, tools.py
├── scripts/                     # Inherited TSLib reproduction scripts (standard benchmarks — see note)
├── requirements/                # Ordered install files reqs_1..4.txt
├── Dockerfile                   # CUDA 12.1 / PyTorch 2.5.1 image
└── docker-compose.yml           # dev service with GPU passthrough
```

> **Note on `scripts/`.** These shell scripts are inherited from upstream TSLib and target the _standard_ academic benchmarks (ETT, ECL, Weather, M4, anomaly/classification datasets). They are **not** the financial experiments of this paper — the stock-price benchmark is run through `run.py` as documented below.

---

## Installation

**Recommended Python: 3.11.** Dependencies are split into four ordered files in [`requirements/`](./requirements/).

### Minimal install (everything the paper needs)

The published models depend only on PyTorch and the standard scientific/attention stack — **not** on the Mamba or foundation-model libraries that the upstream stack also lists.

```bash
# 1. PyTorch (CUDA 12.1 build — torch==2.5.1)
pip install -r requirements/reqs_1.txt

# 2. Core scientific + attention stack
#    (numpy, scipy, scikit-learn, pandas, matplotlib, einops, reformer-pytorch,
#     sktime, sympy, PyWavelets, tqdm, …)
pip install -r requirements/reqs_2.txt
```

> **CPU / Apple Silicon:** `requirements/reqs_1.txt` pins the CUDA 12.1 wheel. If you do not have an NVIDIA GPU, install the matching CPU or MPS build of `torch==2.5.1` from [pytorch.org](https://pytorch.org/get-started/locally/) instead of step 1, then run step 2.

### Optional extras (not required by this repo's models)

`requirements/reqs_3.txt` and `requirements/reqs_4.txt` install the Mamba state-space backend and the foundation-model libraries (Chronos, TimesFM, TiRex, Moirai/uni2ts). **None of these are imported by the published model set**, so you can skip them.

```bash
pip install -r requirements/reqs_3.txt          # mamba_ssm — Linux x86_64 + CUDA 12 + Python 3.11 + torch 2.5 ONLY
pip install -r requirements/reqs_4.txt --no-deps # uni2ts and friends
```

> `reqs_3.txt` pins a `mamba_ssm` wheel built for `cu12 / torch2.5 / cp311 / linux_x86_64` only; it will not install on macOS, Windows, ARM, or other Python versions.

### Docker (optional)

A CUDA environment is provided for full-stack reproduction:

```bash
# The Dockerfile installs from a single consolidated requirements.txt:
cat requirements/reqs_*.txt > requirements.txt
docker compose up -d --build
docker compose exec dev_tslib bash
```

The image builds on `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel`, runs with GPU passthrough (`NVIDIA_VISIBLE_DEVICES=all`), `shm_size: 8gb`, and a `/workspace` volume. It targets the _complete_ stack including the optional Mamba/foundation dependencies.

---

## Reproducing the benchmark

All experiments run through [`run.py`](./run.py), which seeds the RNGs, selects the device, and dispatches to the long-term-forecasting pipeline.

### Train and evaluate the proposed model

```bash
python run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id AAPL_96_24 \
  --model RevTransLSTM-AR \
  --data custom \
  --root_path ./dataset/ \
  --data_path AAPL-2013-2023.csv \
  --features MS --target Close --freq b \
  --seq_len 96 --label_len 48 --pred_len 24 \
  --enc_in 4 --dec_in 4 --c_out 1 \
  --d_model 512 --n_heads 8 --e_layers 2 --d_layers 1 \
  --train_epochs 10 --batch_size 32 --learning_rate 0.0001 --patience 3 \
  --rand_seed 2021
```

### Run a baseline or an ablation

Swap `--model` for any name from the [benchmark set](#benchmark-models) or [ablation table](#ablation-variants); every other flag stays the same:

```bash
python run.py --model revin-LSTM        ... # an RNN baseline
python run.py --model revin-Transformer ... # a Transformer baseline
python run.py --model RevTransLSTM-AR-woarfeedback ... # an ablation
```

### Device selection

- **CUDA** is auto-detected and used by default.
- **Apple Silicon:** add `--gpu_type mps` (this also ensures MPS memory is cleared correctly).
- **CPU:** add `--no_use_gpu`.

### Full sweep

The paper reports results over **all 8 instruments × multiple forecast horizons × several seeds**. Reproduce it by looping the command above over `--data_path`, `--pred_len`, and `--rand_seed` (e.g. seeds `2020`–`2024`), then aggregating the per-run metric lines (see below). Each run is keyed by a unique `setting` string, so repeated runs never collide.

---

## Evaluation and outputs

The long-term-forecasting pipeline ([`exp/exp_long_term_forecasting.py`](./exp/exp_long_term_forecasting.py)) reports an extended set of metrics and diagnostics.

**Metrics** ([`utils/metrics.py`](./utils/metrics.py)): `MAE, MSE, RMSE, MAPE, MSPE, R²` — and optionally **DTW** with `--use_dtw` (off by default; time-consuming).

**Diagnostics** (printed and appended to the result log):
- **GPU peak memory** — `gpu_mem_peak_mb` via `torch.cuda.max_memory_allocated` (CUDA only).
- **Training wall time** and **inference speed** — `inference_speed_ms` (ms/sample, guarded by `torch.cuda.synchronize()`).
- **Parameter counts** — total and trainable.

**Output layout** (relative to the working directory; all are git-ignored):

| Path | Contents |
|---|---|
| `results/<setting>/` | `pred.npy`, `true.npy`, `metrics.npy` (the 6 core metrics) |
| `test_results/<setting>/` | auto-generated figures + per-window preview PDFs |
| `checkpoints/<setting>/checkpoint.pth` | best model by validation loss |
| `result_long_term_forecast.txt` | appended one-line summary per run (all metrics + diagnostics) |

**Publication figures** ([`utils/visualization.py`](./utils/visualization.py), 300 DPI, serif/journal style, PNG):

| File | Content |
|---|---|
| `fig_prediction_curves.png` | Sample ground-truth vs. prediction windows with error band |
| `fig_error_analysis.png` | MSE-per-horizon-step bars + error-distribution histogram |
| `fig_metrics_radar.png` | Radar over MAE / MSE / RMSE / MAPE / MSPE / R² |
| `fig_error_heatmap.png` | Absolute-error heatmap (sample × horizon) |
| `fig_pred_true.png` | Continuous ground-truth vs. prediction curve |
| `fig_dashboard.png` | 4-panel summary |

Figures can be regenerated standalone from saved arrays:

```bash
python -m utils.visualization --input results/<setting>/ --output test_results/<setting>/
```

---

## Adding a model

The registry ([`exp/exp_basic.py`](./exp/exp_basic.py)) **auto-discovers** models: drop a `.py` file anywhere under [`models/`](./models/) that defines a class named `Model`, and its filename (without `.py`) becomes the `--model` string — no manual registration. This is exactly how the RevIN baselines (`revin-*.py`) and the RevTransLSTM-AR family are wired in. An unknown `--model` name raises an error listing the discovered models.

---

## Citation

If you use this code or build on the benchmark, please cite the paper:

```bibtex
@article{hsieh2026revtranslstmar,
  title   = {A Fair Benchmark of Deep Models for Non-Stationary Stock Price
             Forecasting: RevTransLSTM-AR as a Complexity Probe},
  author  = {Hsieh, Chih-Chien},
  year    = {2026}
}
```

> Bibliographic details (journal, volume, DOI) will be completed upon publication.

---

## Acknowledgements

This project is a focused fork of the [Time-Series-Library (TSLib)](https://github.com/thuml/Time-Series-Library) by THUML @ Tsinghua University. It reuses TSLib's experiment harness and layer library, and adds the stock-price datasets, the RevIN-wrapped baselines, and the RevTransLSTM-AR model family studied in the paper.

---

## License

Released under the **MIT License** — see [`LICENSE`](./LICENSE).

- Copyright © 2026 Chih-Chien Hsieh
- Copyright © 2021 THUML @ Tsinghua University (Time-Series-Library)

---

## Contributing

This is a **personal research repository** accompanying a publication, so external pull requests are not accepted (see [`CONTRIBUTING.md`](./CONTRIBUTING.md)). It is open source under MIT and **fork-friendly** — you are welcome to fork it, adapt it, and build on the benchmark. Bug reports and questions can be raised as Issues.
