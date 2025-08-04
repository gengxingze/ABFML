# Installation

## 1. Installing ABFML

To quickly get started with **ABFML**, set up an isolated Python environment and install the package.

```bash
# 1. Create a new conda environment
conda create -n abfml python=3.11

# 2. Activate the environment
conda activate abfml

# 3. Install ABFML

# Option 1: Install from source (recommended for development)
cd path/to/ABFML
pip install .

# Option 2: Install from PyPI (for general use)
pip install abfml
```

---

## 2. Installing ABFML-LAMMPS (Optional)

This step is only required if you plan to run large-scale molecular dynamics simulations 
with **LAMMPS** using ABFML-generated force fields.  
For initial model development and testing, ABFML works directly with **ASE** and you can skip this section.

### 2.1 Prerequisites

1. **LAMMPS**  
   Download and build LAMMPS for your platform:  
   [LAMMPS Official Website](https://lammps.sandia.gov/)

2. **Libtorch**  
   Download the PyTorch C++ API (Libtorch).  
   If your system's toolchain is older, use the **Pre-cxx11 ABI** version.

---

### 2.2 Integration Steps

#### 1. Copy ABFML files into LAMMPS
```bash
cp abfml/lmp/ABFML lammps/src
cp abfml/lmp/Makefile.mpi lammps/src/MAKE
```

#### 2. Modify `Makefile.mpi`
Edit `lammps/src/MAKE/Makefile.mpi` to match your system paths for compilers and libraries:
```makefile
CCFLAGS += -I/PATH/libtorch/include/
CCFLAGS += -I/PATH/libtorch/include/torch/csrc/api/include/
CCFLAGS += -I/PATH/libtorch
LINKFLAGS += -L/PATH/libtorch/lib/ -ltorch -lc10 -ltorch_cpu
```

#### 3. Build LAMMPS with ABFML
```bash
export LD_LIBRARY_PATH=/path/libtorch/lib:$LD_LIBRARY_PATH
cd lammps/src
make yes-abfml
make mpi
```

---

### 2.3 Running LAMMPS with ABFML
Example of using ABFML in an input script:
```lammps
pair_style abfml model.pt
pair_coeff * * 29 30
```
Here, element numbers (e.g., `29`, `30`) represent element types.

---

## Notes
- Ensure that the Libtorch version matches the PyTorch version used for training the ABFML model.
- Always verify the correct `LD_LIBRARY_PATH` before running LAMMPS.
