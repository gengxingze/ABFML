# ABFML Tutorials

Welcome to **ABFML** — a machine learning force field (MLFF) framework based on **PyTorch**.  


## 1. Train model
This tutorial section introduces **three representative examples**, each demonstrating a distinct use case of ABFML:

Build and fit a simple Lennard–Jones-like potential using ABFML’s modular interface

| Example                         | Description                                                                         |
|---------------------------------| ----------------------------------------------------------------------------------- |
| [Mg–Cu](mgcu.md)                | Train a Deep Potential–Smooth Edition model for Mg–Cu alloy systems                 |
| [FCC–Au](au.md)                 | Learn to train a Behler–Parrinello type neural network model for pure Cu            |
| [Custom Model](custom_model.md) | Build and fit a simple Lennard–Jones-like potential using ABFML’s modular interface |

!!! tip "💡 Tip"
    All tutorials can be found in the `examples/` directory of the [ABFML GitHub repository](https://github.com/gengxingze/ABFML).  
    Each includes ready-to-run scripts and datasets.


## 2. Results Analysis
➡️ See: [Using ABFML with ASE](../tutorials/result_analysis.md)

## 3. Running Simulations in LAMMPS and ASE

Once the model is trained, you can integrate it into **LAMMPS** for large-scale atomistic simulations.

This section covers:
- Exporting ABFML-trained models to LAMMPS format
- Setting up `pair_style abfml` and simulation input scripts
- Running MD simulations with MLFFs

ABFML provides seamless integration with **ASE (Atomic Simulation Environment)** for structure optimization, NEB, and MD simulations.

Topics include:
- Loading trained ABFML models into ASE calculators
- Running energy/force evaluations
- Performing relaxation and trajectory analysis

➡️ See: [Using ABFML with ASE](../tutorials/simulation.md)

---

## 5. Utilities and Tools

ABFML also offers several useful utilities to assist your workflow:
- Dataset preparation and splitting
- Log parsing and visualization
- Model inspection and export

➡️ See: [Utilities and Tools](../tutorials/tools.md)

---

## 6. Recommended Workflow

A typical ABFML workflow looks like this:

```mermaid
graph LR
A[Prepare Dataset] --> B[Configure input.json]
B --> C[Train Model with ABFML]
C --> D[Validate Results]
D --> E[Export to LAMMPS/ASE]
E --> F[Run Simulations]
