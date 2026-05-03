# LossRadar / Entropy-based traffic analysis

This repository contains **epoch-based network flow analysis**: detectors read fixed-format flow files, inject synthetic attack traffic for evaluation, and write **JSON** under `<epoch_dataset>/json/`. **Helpers** turn those JSONs into PDFs and plots; **Mutual_Info** tools add MI-specific detection and aggregate **accuracy / precision / recall** reports.

---

## Prerequisites

- **Python 3** (3.8+ recommended)
- **Core Python packages** (install as needed):

  ```text
  pip install numpy matplotlib scikit-learn seaborn
  ```

- **PCAP → single text file of flows** — [Scapy](https://scapy.net/): `pip install scapy`  
  Script: `Helpers/pcapToFlows.py`

- **PCAP → per-epoch flow files** — [tshark](https://www.wireshark.org/docs/man-pages/tshark.html) (Wireshark CLI). The script prints install hints if `tshark` is missing.  
  Script: `Helpers/pcapToEpochFlows.py`

## Detectors (`Detectors/`)

All **CLI-capable** detectors below accept:

- **`directory`** — folder with `epoch_*.txt` (optional; if omitted, many scripts prompt interactively).
- **`--sparse`** — attacks only in random windows of epochs (evaluates “sparse” attacks).
- **`--attack-prob`** — probability a window is attacked in sparse mode (default commonly `0.3`).
- **`--window-size`** — epochs per attack window in sparse mode (default varies; often `10`).

Outputs are written under **`<directory>/json/`** unless noted.

**Filename collisions:** `Entropy_Baseline`, `Entropy_Stream_ZS`, `Entropy_JS`, and `Entropy_Renyi` (and sparse variants) all use the same pattern `{dataset_name}_{baseline}_{malicious}.json`. Only run **one** of these per `json/` folder for a given configuration, or use separate dataset directories so results are not overwritten.

| Script | What it does | Output naming (typical) |
|--------|----------------|-------------------------|
| `Entropy_Baseline.py` | Baseline vs polluted entropy deltas over epochs | `{dataset}_{baseline}_{mal}.json` |
| `Entropy_Stream_ZS.py` | Stream analysis **plus per-epoch Z-scores, composite score, and attack flag** in JSON (same filename pattern as Baseline; different content) | `{dataset}_{baseline}_{mal}.json` |
| `Entropy_JS.py` | Jensen–Shannon / KL style divergence vs baseline | `.json` under `json/` |
| `Entropy_Renyi.py` | Rényi-entropy style analysis | `{name}_{b}_{m}.json` or `*_sparse.json` |
| `Entropy_Cosine.py` | CMS + **cosine similarity** vs baseline | `{name}_{b}_{m}_ml.json` |
| `Entropy_CUSUM.py` | CUSUM change detection on stream features | `*_cusum.json` / `*_cusum_sparse.json` |
| `Entropy_HHH.py` | Heavy-hitter + chi-squared style detector | `*_hhh.json` / `*_hhh_sparse.json` |
| `Entropy_IsolationFor.py` | **Isolation Forest** on features + epoch labels | `*_ml.json` (same pattern as cosine for plot tools) |
| `Entropy_EpochClf.py` | Per-epoch entropy with **injected malicious %** (simpler argv) | `entropy_epoch_analysis_<folder>_<pct>pct.json` (parent of dataset dir) |
| `EntropyML.py` | **Random forest** flow classifier: train/simulate, or `python EntropyML.py <flowfile> [label]` | Model `traffic_classifier.pkl`; file mode prints classification |
| `Entropy_PreUpdate.py` | Pre-update / cluster-style analysis; `python ... <file> [cluster_name]` or run without args for simulation | Console + internal metrics |

**Not a standalone CLI:** `Entropy_Primitive.py` is a **library** of primitives (sketches, parsing); import it from other code rather than running it as the main entry point.

**`Mutual_Info/Entropy_MI_Detector.py`** (MI detector, run like the others):

```bash
python Mutual_Info/Entropy_MI_Detector.py /path/to/epoch_dir
# optional: --sparse --attack-prob 0.3 --window-size 10
```

Writes files such as **`mi_<baseline>_<malicious>_<z_threshold>.json`** (and sparse variants) into **`json/`**.

> **Note:** `Detectors/Entropy_MI.py` is a **plotting / PDF tool for Entropy_Stream-style JSON** (MI timelines, boxplots). It is **not** the same as `Mutual_Info/Entropy_MI_Detector.py`. Use the paths above to avoid mixing them up.

> **Note:** `Entropy_Stream_ZS.py` accepts `--threshold-z`, `--threshold-composite`, and `--threshold-flow-ratio` and prints them, but the run functions call `build_detection_block` with **built-in defaults** (2.5, 2.0, 1.05). To use custom thresholds, pass them through `build_detection_block` in code or change the defaults in that function.

---

## Plot and report helpers (`Helpers/`)

| Script | Input | Output |
|--------|--------|--------|
| `generatePlotsBaseline.py` | `python generatePlotsBaseline.py <json_folder>` | PDFs in **`<parent of json>/results/`** |
| `generatePlotsJS.py` | `python generatePlotsJS.py <json_folder>` | PDFs in **`../results/`** relative to the json folder |
| `generatePlotsCUSUM.py` | `python generatePlotsCUSUM.py <json_folder>` | One PDF per JSON; **`../results/`** (see file docstring) |
| `generatePlotsCosine.py` | `python generatePlotsCosine.py [dir]` | Reads `*_ml*.json`; PDFs in **`json/reports/`** (or under the path you pass) |
| `generateIsolationPlots.py` | `python generateIsolationPlots.py [dir]` | PNGs + metrics under **`json/plots/`** for `*_ml*.json` |
| `generatePlotsEpoch.py` | `python generatePlotsEpoch.py <json_file> [out.pdf]` | Multi-page PDF for **Entropy_EpochClf**-style or stream-style JSON (see script) |

Use the helper that matches the detector output you produced (baseline vs JS vs CUSUM vs `*_ml.json`, etc.).

---

## Mutual information: analysis and charts (`Mutual_Info/`)

After running **`Entropy_MI_Detector.py`**, aggregate metrics and figures:

| Script | Purpose |
|--------|---------|
| `analysisEveryEpoch.py` | **`python analysisEveryEpoch.py <json_dir>`** — builds **`analysis_report.json`**, optional **`--pdf`**, plots under **`--plot-dir`**. Full-mode MI reports. |
| `analysisSparse.py` | Same family for **sparse** filenames (`*_sparse.json`). |
| `Plot_MI_Results.py` | Rich PDF visualization: **`python Plot_MI_Results.py <json_or_dir> [-o out.pdf]`**, **`--batch-reports`**, **`--minimal`**, etc. |

These scripts expect MI JSON layouts (`metadata`, `baseline`, `epochs` with `z_scores`, etc.). File naming patterns are documented in the script headers (e.g. `mi_5.0_1.0_2.5.json`).

---

## Optional scripts (`scripts/`)

- `compare_cosine_vs_mi_sparse.py` — compares cosine vs MI in sparse settings (see `--help`).
- `generate_cosine_similarity_illustration.py` — illustration / figures for cosine similarity (see `--help`).
