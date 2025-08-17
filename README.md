# Quantitative Comparison of Simulated and Experimental Fluorescence

This repository provides a **reproducible, quantitative workflow** to evaluate excitonic-model simulations against **experimental fluorescence** data. It compares both **time-resolved decays** and **steady-state spectra** and ranks models using a composite score that balances agreement across domains.

---

## Why this is needed

Simulations generate many trajectories under different parameters. Selecting the “best” model by eye is subjective. Here, we:

- Normalize and compare **time decays** and **spectra** on a common footing.  
- Evaluate the **overall spectral shape** in 650–760 nm **and** the **red shoulder** in 710–740 nm.  
- Include **lifetime** information via bi-exponential fits.  
- Combine everything into a **single, interpretable score** for ranking.

---

## How it works (at a glance)

1. **Normalization**
   - **Time**: each simulated decay is normalized to its **own maximum**.
   - **Spectrum**: each spectrum (sim and experimental) is normalized to its **peak within 650–760 nm**.

2. **Error metrics**
   - $E_{\text{time}}$: MSE between normalized simulated decay and experimental bi-exponential on the same time grid.  
   - $E_{\text{wl}}$: MSE between normalized spectra **only within 650–760\,nm**.  
   - $E_{\text{area}}$: relative AUC error in the **red-shoulder ROI (710–740\,nm)**.  
   - $E_{\tau}$: relative error in **average lifetime** from a bi-exponential fit to the normalized simulated decay.

3. **Scoring summary**
   - Errors are **cap-normalized** and combined with a **weighted average** plus a **balance penalty**; lower is better. Detailed equations are below.

---

## Mathematical definitions 

### Cap-normalization

Each error is normalized and capped at 1.0:

```math
E_{i,\text{norm}} = \min\!\left( \frac{E_i}{E_{i,\max}}, 1.0 \right)
```

with fixed caps:

```math
E_{\text{time},\max} = 0.05, \quad
E_{\text{wl},\max} = 0.05, \quad
E_{\text{area},\max} = 0.20, \quad
E_{\tau,\max} = 0.10
```

### Final score

The combined score (lower is better):

```math
S = \tfrac{1}{2}\Big( w_{\text{time}}E_{\text{time,norm}}
+ w_{\text{wl}}E_{\text{wl,norm}}
+ w_{\text{area}}E_{\text{area,norm}}
+ w_{\tau}E_{\tau,\text{norm}} \Big)
+ \tfrac{1}{2}\sqrt{E_{\text{time,norm}} E_{\text{wl,norm}}}, \quad
w_i = 0.25
```

### Hard penalty

If either primary error is too large, the score is strongly penalized:

```math
\text{If } E_{\text{time,norm}} > 0.9 \text{ or } E_{\text{wl,norm}} > 0.9,
\quad S \leftarrow 10 S
```

---

## Repository structure (core analysis)

The code is organized into small, single-purpose classes (all inside `Best_model_pick.py`):

- **`SpectrumLoader`** — load experimental spectrum; load/return sorted sim pairs.  
- **`Normalizer`** — time normalization (`y/max(y)`), spectrum normalization (peak in 650–760 nm).  
- **`Metrics`** — compute: time MSE, wavelength MSE (650–760 nm), ROI area error (710–740 nm), lifetime error.  
- **`Scorer`** — cap/normalize errors and compute the final score with the balance and penalty terms.  
- **`SimulationAnalyzer`** — orchestrates the run: scan folders, compute errors, rank models, export CSV/NPY, and plot.

```
calc/model_12/
  Best_model_pick.py      # main analysis & classes
  ... your Run_* data ... # each run has Analysis_Data/Data/*.npy
```

---

## Expected input format

- **Experimental spectrum**: `.npy` with two columns `[wavelength_nm, intensity]` (unsorted is OK).  
- **Simulated outputs** live under each run, e.g.:
  ```
  Run_FL16_ISC5_.../
    Analysis_Data/Data/
      time_resolved_<R>_start_time_....npy        # two columns [time, population]
      wavelength_resolved_<R>_start_time_....npy   # two columns [wavelength, intensity]
  ```
  Files are paired by the **ratio** `<R>` using the regex  
  `(time|wavelength)_resolved_(.*?)_start_time`.

---

## Installation

```bash
# Python 3.9+
pip install numpy pandas matplotlib scipy
```

---

## Quickstart

```python
from Best_model_pick import SimulationAnalyzer

# 1) Point to your run root and experimental spectrum
base_dir = "/path/to/main_folder"
exp_wavelength_file = "/path/to/exp_spectrum.npy"

# 2) Experimental time-decay parameters (bi-exponential, amplitudes in %)
exp_time_params = dict(a1_pct=60.0, tau1=1.2, a2_pct=40.0, tau2=5.0, tau_avg=(0.6*1.2 + 0.4*5.0))

# 3) Create analyzer (defaults: wavelength_range=(650,760), top N=10)
an = SimulationAnalyzer(
    base_dir=base_dir,
    exp_wavelength_file=exp_wavelength_file,
    exp_time_params=exp_time_params,
    wavelength_range=(650, 760),
    N_best=10
)

# 4) Run the full analysis
an.run_analysis()

# Artifacts:
# - outputs/best_model_errors.csv
# - outputs/best_model_detailed.png
# - plots/best_models/combined_plot_*.png
# - npy_data/<run_name>/time_resolved_<R>.npy (3 columns: t, sim, exp)
# - npy_data/<run_name>/wavelength_resolved_<R>.npy (3 columns: λ, sim, exp)
```

---

## What the score looks at (exactly)

- **Overall spectral agreement in 650–760 nm** — after normalization to the in-window peak, we compute $E_{\text{wl}}$ on that window only.  
- **Red shoulder (710–740 nm)** — integrate normalized intensities over 710–740 nm and compute the relative area error, $E_{\text{area}}$.  
- **Decay shape + lifetime** — compare the normalized simulated decay to the experimental bi-exponential (MSE), and fit a bi-exponential to get the simulated average lifetime for $E_{\tau}$.  
- **Balance requirement** — the geometric-mean term $\sqrt{E_{\text{time}}E_{\text{wl}}}$ penalizes models that only fit one domain.

---

## Reproducible outputs

- **CSV summary:** `outputs/best_model_errors.csv` with rank, file names, all four raw errors, and the final score.  
- **NPY bundles:** for top models we save side-by-side arrays with simulation **and** experimental references (3 columns), ready for plotting.  
- **Figures:** detailed plot for the best model and a grid of top models.  
- **(Optional)** An “error space” scatter (`outputs/error_space.png`) visualizes trade-offs between time vs wavelength fits.

---

## Interpreting results

- **Lower score = better overall agreement** (time + spectrum + shoulder + lifetime).  
- If a model looks good in one domain but poor in the other, the **balance term** and **hard penalty** will push it down the ranking.  
- Check the top-N plots to see **where** remaining discrepancies live.

---

## Troubleshooting

- **No runs found:** verify `base_dir/Run_*` exists and contains `Analysis_Data/Data`.  
- **No matched pairs:** confirm both files share the same ratio string after `time_resolved_` / `wavelength_resolved_`.  
- **Weird scaling:** ensure the experimental spectrum covers **650–760 nm**; we normalize to the **in-window** peak.

---

## License

[MIT](LICENSE) (or your preferred license)
