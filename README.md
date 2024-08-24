___
# ABFML 

## 1. Introduction
ABFML is a software platform developed on the popular deep learning framework PyTorch. It provides extensions and support for third-party packages like LAMMPS and ASE. The primary goal of ABFML is to offer researchers a user-friendly platform for the rapid, simple, and efficient development, fitting, and application of new machine learning force field models. By leveraging PyTorch's powerful tensor operations, ABFML simplifies the complex process of developing force field models and accelerates their application in molecular simulations.

## 2. Installation

### 2.1 Installing ABFML
To get started with ABFML, follow these steps to set up the environment and install the necessary components:
```bash
# 1. Create a new conda environment:
   conda create -n abfml python=3.11

# 2. Activate the conda environment:
   conda activate abfml
#   This command activates the `abfml` environment, making it the current working environment.

# 3. Install PyTorch:
   conda install pytorch

# 4. Navigate to the ABFML directory:
   cd path/abfml

# 5. Install ABFML:
   pip install .
#   This installs ABFML from the local directory.
```
### 2.2 Installing ABFML-LAMMPS
To integrate ABFML with LAMMPS, follow these steps:
```bash
# 1. Download LAMMPS:
#   Obtain the LAMMPS package suitable for your system from the [LAMMPS website](https://lammps.sandia.gov/). This will be the molecular dynamics engine that ABFML will interact with.

# 2. Download Libtorch:
#   Libtorch is the PyTorch C++ API.

# 3. Copy ABFML files to LAMMPS:  
   cp abfml/lmp/ABFML lammps/src
   cp abfml/lmp/Makefile.mpi lammps/src/MAKE
#   These commands copy the necessary ABFML files into the LAMMPS source directory.

# 4. Modify the Makefile:
#   Edit `Makefile.mpi` in the `lammps/src/MAKE` directory to match your system's configuration, specifically setting the paths to your compilers and libraries.  
   CCFLAGS += -I/PATH/libtorch/include/
   CCFLAGS += -I/PATH/libtorch/include/torch/csrc/api/include/
   CCFLAGS += -I/PATH/libtorch
   LINKFLAGS += -L/PATH/libtorch/lib/ -ltorch -lc10 -ltorch_cpu
   
# 5. Compile LAMMPS with ABFML:
#   Navigate to the `src` directory of LAMMPS and install: 
   export LD_LIBRARY_PATH=path/libtorch/lib:$LD_LIBRARY_PATH
   cd src
   make yes-abfml
   make mpi
```
**Note**: Ensure your GCC compiler is version 9.0 or above, as older versions may not support.

## 3. Run

To run an example using ABFML, follow these steps:
```bash
# 1. Activate the ABFML environment:
   conda activate abfml

# 2. Navigate to the example directory:
   cd abfml/example/dp
  
# 3. Run:
   abfml train input.json
#   This command starts the training process using the configurations specified in `input.json`.

# 4. Quick check for model correctness
   abfml check -m model.pt -d float32
# 5. Run lammps:  
   pair_style abfml model.pt
   pair coeff * * 29 30
  
#   Use the element number to represent the element type
```
## 4. Build a new model
Build your own model input in the input file and specify the path to the model's py file
```json
{
  "model_setting": {
          "name": "user_defined",
          "model_path": "D:/Work/PyCharm/ABFML/example/new-MLFF/LJ-model.py",
          "field_name": "LJ",
          "normal_name": "LJNormal",
          "descriptor" : {
                         "epsilon": 1.0,
                         "sigma": 1.5
          }
  }
}
```
Then you can  use the relevant modules for training, validation and simulation