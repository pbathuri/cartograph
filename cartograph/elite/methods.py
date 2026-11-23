"""Method hints — the textbook move and the worth-trying tangent, per technical area.

When the user is mid-task (a CUDA kernel, a time-series model, an MD run), the best help is often not a
search result but a reminder of the established method they may not know by name, plus one less-obvious
technique from an adjacent field that is worth a quick experiment. Each entry: trigger keywords, the
textbook methods (safe, well-understood), and tangents (experimental / cross-field, with when-to-try).
Curated, field-agnostic, extend freely. Matching is keyword-based so it costs nothing at serve time.
"""
from __future__ import annotations

import re

METHODS: dict[str, dict] = {
    "cuda_kernels": {
        "kw": ("cuda", "kernel", "gpu", "warp", "shared memory", "occupancy", "nvcc", "thread block", "triton", "gemm"),
        "textbook": [
            "Coalesce global loads (consecutive threads -> consecutive addresses); check with Nsight Compute memory workload.",
            "Tile through shared memory and pad the tile (+1 column) to avoid bank conflicts.",
            "Reduce within a warp with __shfl_down_sync before touching shared memory; one atomic per block.",
            "Place the kernel on the roofline first (arithmetic intensity vs peak) to know if it is memory- or compute-bound.",
            "Overlap H2D/D2H copies with compute using streams + pinned memory.",
        ],
        "tangents": [
            ("Kernel fusion of elementwise chains (fewer global round-trips)", "memory-bound pipelines of small ops"),
            ("Tensor cores via WMMA / CUTLASS templates, even for non-ML GEMM-like loops", "FP16/BF16 tolerable, dims multiple of 8/16"),
            ("Write the kernel in Triton first to find the right tiling, then port", "fast iteration on block sizes"),
            ("Thread coarsening (more work per thread) to cut index math and sync", "low arithmetic intensity, many tiny threads"),
        ],
    },
    "sparse_computing": {
        "kw": ("sparse", "spmv", "spmm", "csr", "csc", "coo", "sparsity", "graph partition"),
        "textbook": [
            "Pick the format by access pattern: CSR for row SpMV, CSC for column ops, ELL/SELL-C-sigma for GPUs with regular rows.",
            "Reorder (RCM / METIS) to improve locality before tuning the kernel.",
            "Measure in effective bandwidth (bytes moved / time) — SpMV is memory-bound.",
        ],
        "tangents": [
            ("Merge-path / load-balanced SpMV for skewed row lengths", "power-law degree distributions"),
            ("Mixed-precision values with full-precision accumulation", "bandwidth-bound and tolerance allows"),
            ("Block-sparse formats to reach tensor cores", "structured sparsity / dense sub-blocks"),
        ],
    },
    "time_series": {
        "kw": ("time series", "timeseries", "forecast", "arima", "lstm", "autoregress", "seasonal", "regime",
               "volatility", "garch", "kalman", "lookahead", "walk forward", "returns"),
        "textbook": [
            "Validate with walk-forward (expanding/rolling) splits; purge + embargo around the test window to stop leakage.",
            "Baselines first: naive/seasonal-naive, ARIMA/ETS — a deep model must beat these out of sample.",
            "State-space form + Kalman filter for noisy latent dynamics and missing data.",
            "Check stationarity (ADF/KPSS), difference or model the trend explicitly.",
            "GARCH-family for volatility clustering; evaluate with proper scoring (QLIKE, CRPS), not MSE alone.",
        ],
        "tangents": [
            ("Patch-based transformers (PatchTST) or pretrained foundation models (Chronos, TimesFM) as zero-shot baselines", "short history, many related series"),
            ("Conformal prediction for time series (EnbPI / ACI) for calibrated intervals", "need coverage guarantees under drift"),
            ("Hidden Markov / Markov-switching models for regimes, used as features", "behaviour changes between regimes"),
            ("Wavelet or spectral (FFT) features for multi-scale periodicity", "mixed daily/weekly/intraday cycles"),
            ("Signature transforms (rough-path theory) as path features", "irregular sampling, order matters"),
        ],
    },
    "quant_finance": {
        "kw": ("backtest", "alpha", "factor", "portfolio", "sharpe", "drawdown", "trading", "risk parity", "momentum"),
        "textbook": [
            "Deflated Sharpe ratio + count every trial you ran (multiple-testing correction).",
            "Triple-barrier labelling and meta-labelling to separate side from size (López de Prado).",
            "Costs, slippage and capacity in the backtest from day one.",
        ],
        "tangents": [
            ("Hierarchical Risk Parity instead of mean-variance", "unstable covariance, many assets"),
            ("Combinatorial purged CV to get a distribution of backtest paths", "need a PBO estimate"),
            ("Fractional differentiation to keep memory while gaining stationarity", "features lose signal when differenced"),
        ],
    },
    "molecular_dynamics": {
        "kw": ("molecular dynamics", "gromacs", "amber", "openmm", "md simulation", "xtc", "trajectory", "force field",
               "rmsd", "docking", "free energy"),
        "textbook": [
            "Equilibrate (NVT then NPT) and check energy, temperature, density and RMSD plateaus before production.",
            "PME for long-range electrostatics; verify cutoff and timestep (2 fs with constrained H-bonds).",
            "Block averaging / autocorrelation time for honest error bars.",
        ],
        "tangents": [
            ("Enhanced sampling (metadynamics, umbrella sampling, REST2) for rare events", "barrier crossings not seen in plain MD"),
            ("Markov state models (PyEMMA / deeptime) over many short runs", "kinetics from aggregated trajectories"),
            ("ML force fields / learned collective variables", "need accuracy beyond classical FF or a good CV"),
        ],
    },
    "signals": {
        "kw": ("impulse response", "convolution", "fourier", "laplace", "z-transform", "filter", "lti", "sampling",
               "fir", "iir", "eog", "eeg", "blink", "biosignal"),
        "textbook": [
            "Band-pass the signal to the physiological band first (e.g. EOG blinks ~0.1-15 Hz) with a zero-phase filter (filtfilt).",
            "Characterize LTI systems by impulse response; convolution in time = multiplication in frequency.",
            "Detect events with a matched filter / template correlation and an adaptive threshold.",
        ],
        "tangents": [
            ("Dynamic time warping for template matching across speeds", "event shape varies in duration"),
            ("Continuous wavelet transform for transient events", "short bursts on a drifting baseline"),
            ("Self-supervised pretraining on unlabelled recordings, then few-shot fine-tuning", "few labelled events"),
        ],
    },
    "ml_experiment": {
        "kw": ("train", "model", "classifier", "fine-tune", "finetune", "dataset", "overfit", "accuracy", "loss"),
        "textbook": [
            "Strong baseline + ablation table; change one thing at a time.",
            "Calibration (reliability diagram, temperature scaling) before trusting probabilities.",
            "Group-aware splits (by subject/session) to avoid identity leakage.",
        ],
        "tangents": [
            ("Sharpness-aware minimization (SAM) for small data", "train/val gap with tiny datasets"),
            ("Test-time augmentation + ensembling of checkpoints (SWA)", "cheap accuracy/calibration gain"),
        ],
    },
    "hpc_parallel": {
        "kw": ("mpi", "openmp", "slurm", "hpc", "parallel", "scaling", "cluster", "linpack", "hpl", "benchmark"),
        "textbook": [
            "Report strong and weak scaling against Amdahl/Gustafson expectations.",
            "Pin threads/ranks to cores and NUMA domains; measure before and after.",
            "For HPL: tune NB, P x Q grid (near-square, P <= Q) and memory fill (~80-90%).",
        ],
        "tangents": [
            ("Communication-avoiding algorithms / overlapping MPI with compute", "network-bound at scale"),
            ("Roofline per kernel with likwid or Nsight to find the true bottleneck", "unclear why scaling stalls"),
        ],
    },
}


def _hit(text: str, kw: str) -> bool:
    return re.search(rf"(?<![a-z]){re.escape(kw)}(?![a-z])", text) is not None


def match_methods(prompt: str, *, limit: int = 2) -> list[dict]:
    """Areas the prompt touches, strongest first, with textbook moves and tangents."""
    t = (prompt or "").lower()
    scored = []
    for area, m in METHODS.items():
        n = sum(_hit(t, k) for k in m["kw"])
        if n:
            scored.append((n, area))
    scored.sort(reverse=True)
    out = []
    for _, area in scored[:limit]:
        m = METHODS[area]
        out.append({"area": area, "textbook": m["textbook"][:3],
                    "tangents": [{"try": a, "when": w} for a, w in m["tangents"][:2]]})
    return out
