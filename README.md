## WavePropBench

This repository provides a quantitative benchmarking framework comparing:
- ull-wave FDTD simulations (MEEP)
- Envelope-based NLSE propagation (SSFM)

The goal is to evaluate the consistency, fidelity limits, and divergence mechanisms between classical electromagnetic simulations and reduced-order envelope models used in quantum photonic circuit pre-design workflows.

By quantum-inspired modeling, we refer to classical electromagnetic and envelope solvers commonly used prior to field quantization and quantum noise modeling in integrated photonics.


### Physical System

We simulate Gaussian pulse propagation at:
	•	Carrier frequency: 193.5 THz
	•	Wavelength: 1.55 μm
	•	Media:
	•	Vacuum (n = 1.0)
	•	Dielectric waveguide (n = 1.5)

### Methods used:
	•	MEEP (FDTD) — carrier-resolved full-wave solver
	•	SSFM (NLSE) — split-step Fourier envelope solver
