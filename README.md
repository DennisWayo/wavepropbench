## WavePropBench

This repository provides a quantitative benchmarking framework comparing:
- Full-wave FDTD simulations (MEEP)
- Envelope-based NLSE propagation (SSFM)

The goal is to evaluate the consistency, fidelity limits, and divergence mechanisms between classical electromagnetic simulations and reduced-order envelope models used in quantum photonic circuit pre-design workflows.

By quantum-inspired modeling, we refer to classical electromagnetic and envelope solvers commonly used prior to field quantization and quantum noise modeling in integrated photonics.


### Physical System

We simulate Gaussian pulse propagation at:
- Carrier frequency: 193.5 THz
- Wavelength: 1.55 μm
- Media:
- Vacuum (n = 1.0)
- Dielectric waveguide (n = 1.5)

### Methods used:
- MEEP (FDTD) — carrier-resolved full-wave solver
- SSFM (NLSE) — split-step Fourier envelope solver

## Benchmark Summary

1. Temporal Domain
- SSFM preserves a compact Gaussian envelope
- MEEP resolves carrier-scale interference and numerical dispersion

| Metric                    | SSFM      | MEEP     |
|---------------------------|-----------|----------|
| FWHM (fs)                 | 49.34     | 177.14   |
| RMS Width (fs)            | 14.93     | 53.15    |
| Relative RMS Width Error  | Reference | 0.7192   |

- The divergence arises from physical and numerical effects captured by full-wave solvers but intentionally excluded from envelope models.


2. Spectral Domain
- Carrier frequency preserved in both methods
- FFT spectral centroids tightly aligned near 0.16 THz
- Spectral leakage minimal
- Gaussian spectral profile retained

Despite envelope-width differences, frequency-domain behavior remains consistent.


3. Group Delay Consistency

Arrival times:
- Vacuum: ~120 timesteps
- Dielectric: ~160 timesteps

Both solvers reproduce group velocity reduction accurately.
