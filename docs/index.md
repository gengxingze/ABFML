# ABFML

**ABFML** (Apply, Build, Fit Machine Learning force field) is a Python-based package designed to simplify the development, 
screening, and tuning of new machine learning force field (MLFF) models.  

Unlike tools that focus on a single model, ABFML is specifically intended for constructing and applying **new MLFF 
models and descriptors**. Its modular design simplifies operations, facilitates integration of training and simulation, 
and allows researchers to leverage existing MLFF models or develop and select models tailored to their specific problems.

!!! important "Important"
    The project **ABFML** is licensed under [GNU LGPLv3.0](https://www.gnu.org/licenses/lgpl-3.0.html).  
    If you use this code in any future publications, please cite the following references:

    - Xingze Geng, Jianing Gu, Gaowu Qin, Lin-Wang Wang,and Xiangying Meng. “ABFML: A problem-oriented package for rapidly 
      creating, screening, and optimizing new machine learning force fields.”** J. Chem. Phys.* 162, 052502 (2025)     
    
    In addition, please see [this page](citation.md) for more citation details.
---

## Features
- Rapid construction of MLFF models based on physical descriptors
- Support for large-scale molecular dynamics via **LAMMPS** and **ASE**
- Modular design for easy extension and research customization

---

## Easy Install
```bash
pip install abfml
```
For more details, check the [Installation Guide](install.md) and [Tutorials](tutorials/index.md).

---

## Documentation
- [Installation](install.md)
- [Tutorials](tutorials/index.md)
- [API Reference](api/method.md)
- [Citation](citation.md)

---

