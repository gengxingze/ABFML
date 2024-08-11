# ABFML 

### 1. Introduction
ABFML is a software platform developed on the popular deep learning framework PyTorch. It provides extensions and support for third-party packages like LAMMPS and ASE. The primary goal of ABFML is to offer researchers a user-friendly platform for the rapid, simple, and efficient development, fitting, and application of new machine learning force field models. By leveraging PyTorch's powerful tensor operations, ABFML simplifies the complex process of developing force field models and accelerates their application in molecular simulations.

### 2. Installation

#### 2.1 Installing ABFML
To get started with ABFML, follow these steps to set up the environment and install the necessary components:

1. **Create a new conda environment**:
   ```bash
   conda create -n abfml python=3.11
   ```

2. **Activate the conda environment**:
   ```bash
   conda activate abfml
   ```
   This command activates the `abfml` environment, making it the current working environment.

3. **Install PyTorch**:
   ```bash
   conda install pytorch
   ```

4. **Navigate to the ABFML directory**:
   ```bash
   cd path/abfml
   ```


5. **Install ABFML**:
   ```bash
   pip install .
   ```
   This installs ABFML from the local directory.

#### 2.2 Installing ABFML-LAMMPS
To integrate ABFML with LAMMPS, follow these steps:

1. **Download LAMMPS**:
   Obtain the LAMMPS package suitable for your system from the [LAMMPS website](https://lammps.sandia.gov/). This will be the molecular dynamics engine that ABFML will interact with.

2. **Download Libtorch**:
   Libtorch is the PyTorch C++ API.

3. **Copy ABFML files to LAMMPS**:
   ```bash
   cp abfml/lmp/ABFML lammps/src
   cp abfml/lmp/Makefile.mpi lammps/src/MAKE
   ```
   These commands copy the necessary ABFML files into the LAMMPS source directory.

4. **Modify the Makefile**:
   Edit `Makefile.mpi` in the `lammps/src/MAKE` directory to match your system's configuration, specifically setting the paths to your compilers and libraries.
   ```bash
   CCFLAGS += -I/PATH/libtorch/include/
   CCFLAGS += -I/PATH/libtorch/include/torch/csrc/api/include/
   CCFLAGS += -I/PATH/libtorch

   LINKFLAGS += -L/PATH/libtorch/lib/ -ltorch -lc10 -ltorch_cpu
   ```
5. **Compile LAMMPS with ABFML**:
   Navigate to the `src` directory of LAMMPS:
   ```bash
   export LD_LIBRARY_PATH=path/libtorch/lib:$LD_LIBRARY_PATH
   cd src
   make yes-abfml
   make mpi
   ```
   **Note**: Ensure your GCC compiler is version 9.0 or above, as older versions may not support the necessary features.

### 3. Run

To run an example using ABFML, follow these steps:

1. **Activate the ABFML environment**:
   ```bash
   conda activate abfml
   ```
   This ensures that all necessary dependencies are available.

2. **Navigate to the example directory**:
   ```bash
   cd abfml/example/dp
   ```

3. **Run**:
   ```bash
   abfml train input.json
   ```
   This command starts the training process using the configurations specified in `input.json`.

4. **Run lammps**:
   ```bash
   pair_style abfml model.pt
   pair coeff * * 29 30
   ```
   Use the element number to represent the element type

