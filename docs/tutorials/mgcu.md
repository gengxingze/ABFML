# MgCu Alloy MLFF Example

## 1. Prepare the Dataset

Prepare a dataset containing **total energy, atomic forces, and Virial tensors** from DFT calculations.

* Example dataset path: `../data/test.extxyz`

---

## 2. Parameter Configuration

Create a file named `input.json` with the following content:

```json
{
  "name": "model_parameters",
  "global_setting": {
    "type_map":[29],
    "neighbor": [100],
    "cutoff": 6.0,
    "dtype": "float64",
    "logger_file": "mlff.log"
  },
  "model_setting": {
    "name": "dp_se_e2_a",
    "descriptor": {
      "R_max": 6.0,
      "R_min": 0.5,
      "smooth_fun": "poly2_r",
      "network_size": [25,50,100],
      "bias": true,
      "resnet_dt": false,
      "activate_function": "tanh",
      "M2": 16
    },
    "fitting_net": {
      "network_size": [200,200,200],
      "bias": true,
      "resnet_dt": true,
      "activate_function": "tanh"
    }
  },
  "data_setting": {
    "file_format": "extxyz",
    "shuffle": true,
    "train_file":  ["../data/test.extxyz"],
    "valid_file":  ["../data/test.extxyz"]
  },
  "train_setting": {
    "epoch": 1,
    "optimizer":"Adam",
    "save_check": 5,
    "batch_size": 1,
    "print_freq": 1,
    "model_file": "model.pt"
  },
  "loss_setting": {
    "start_energy_weight": 1.0,
    "limit_energy_weight": 10.0,
    "start_force_weight": 100.0,
    "limit_force_weight": 10.0,
    "start_virial_weight": 1.0,
    "limit_virial_weight": 1.0,
    "start_ei_weight": 0.0,
    "limit_ei_weight": 0.0,
    "loss_file": "loss.curve"
  },
  "learning_rate": {
    "type_name": "exp",
    "decay_step": 500,
    "decay_rate": 0.95,
    "start_lr": 1.0e-3,
    "limit_lr": 1.0e-6
  }
}
```

> 🔹 Notes:
>
> * `type_map`: 29 corresponds to Cu. Modify according to your atomic types.
> * `neighbor`: Maximum number of neighbors considered per atom.
> * `descriptor`: Descriptor settings affecting local atomic environment representation.
> * `fitting_net`: Fitting network parameters.
> * `loss_setting`: Weights for energy, forces, and Virial; can change during training.
> * `learning_rate`: Exponential decay learning rate.

---

## 3. Train the Model

Run the following command in the terminal:

```bash
abfml train input.json
```

* Output files:

  * Model weights: `model.pt`
  * Log: `mlff.log`
  * Loss curve: `loss.curve`

Training output example:

```
[INFO]- ################################################################################################
[INFO]- #                         _       ______   ________  ____    ____  _____                       #
[INFO]- #                        / \     |_   _ \ |_   __  ||_   \  /   _||_   _|                      #
[INFO]- #                       / _ \      | |_) |  | |_ \_|  |   \/   |    | |                        #
[INFO]- #                      / ___ \     |  __'.  |  _|     | |\  /| |    | |   _                    #
[INFO]- #                    _/ /   \ \_  _| |__) |_| |_     _| |_\/_| |_  _| |__/ |                   #
[INFO]- #                   |____| |____||_______/|_____|   |_____||_____||________|                   #
[INFO]- #                                                                                              #
[INFO]- ################################################################################################
[INFO]- num_threads : 16
[INFO]- +-------------------------------------- train data file ---------------------------------------+
[INFO]- | ################ file name ############### Atom number ## nframes ######### Element ######## |
[INFO]- |                              ../data/test.extxyz   128     2                MgCu
[INFO]- | Number of files found:    1, number of images:   2
[INFO]- +----------------------------------------------------------------------------------------------+
[INFO]- +-------------------------------------- valid data file ---------------------------------------+
[INFO]- | ################ file name ############### Atom number ## nframes ######### Element ######## |
[INFO]- |                              ../data/test.extxyz   128     2                MgCu
[INFO]- | Number of files found:    1, number of images:   2
[INFO]- +----------------------------------------------------------------------------------------------+
[WARNING]- [LR Warning] Need ~674 epochs to reach limit_lr, but training is set to 1 epochs.
[INFO]- ########################################--Start  Train--########################################
[INFO]- +------------------------------------------Start Loop------------------------------------------+
[INFO]- |-- Train Epoch:  1, batch size:  1, total iters:    2 --------------------------------------|
[INFO]- | Train Iters:     1 ,Loss: 1.83e+00 ,Time: 0.149 ,E_tot: 1.39e+00 ,Force: 3.85e-01 ,Virial: 6.04e-02
[INFO]- | Train Iters:     2 ,Loss: 1.99e+00 ,Time: 0.084 ,E_tot: 8.37e-01 ,Force: 4.56e-01 ,Virial: 6.96e-01
[INFO]- | Train Information: Loss: 6.67e-01 ,Time: 10.601 ,E_tot: 2.04e-01 ,Force: 1.50e-01 ,Virial: 3.12e-01
[INFO]- |-- Valid Epoch:  1, batch size:  1, total iters:    2                                       |
[INFO]- | Valid Iters:     1 ,Loss: 2.11e-01 ,Time: 0.038 ,Force: 5.63e-02 ,Virial: 1.29e-01 ,E_tot: 2.57e-02
[INFO]- | Valid Iters:     2 ,Loss: 2.16e-01 ,Time: 0.030 ,Force: 6.87e-02 ,Virial: 1.20e-01 ,E_tot: 2.70e-02
[INFO]- | Valid Information: Loss: 2.12e-01 ,Time: 2.722 ,Force: 6.15e-02 ,Virial: 1.24e-01 ,E_tot: 2.66e-02
[INFO]- +------------------------------------------ End Loop ------------------------------------------+
[INFO]- | Total training time:   26.07s, total training iters:     2
[INFO]- ######################################## --End  Train-- ########################################
```

> 🔹 Tip: Adjust `epoch` for more training steps, and `print_freq` to control logging frequency.

---

## 4. Validation and Result Analysis

### 4.1 Validate Model Performance

```bash
abfml valid -m model.pt -f "../data/test.extxyz" -n 10 -p
```

Example output:

```
[INFO]- | Valid Information: Loss: 5.57e-01 ,Time: 4.201 ,Virial: 3.78e-01 ,Force: 1.02e-01 ,E_tot: 7.73e-02
[INFO]- +----------------------------------------------------------------------------------------------+
[INFO]- |                                       Quality Of Data                                        |
[INFO]- | Key                       DFT Mean         DFT Sigma      Predict Mean     Predict Sigma     |
[INFO]- +----------------------------------------------------------------------------------------------+
[INFO]- | energy                 -4.3369e+02            0.2720       -4.4358e+02            0.2268     |
[INFO]- | forces                  2.8776e-05            0.3801       -1.3878e-19            0.3648     |
[INFO]- | virials                -8.8748e+00           14.4830       -3.1681e+01           56.3676     |
[INFO]- +----------------------------------------------------------------------------------------------+
[INFO]- |                                    Evaluation Indicators                                     |
[INFO]- | Key                           RMSE               MSE               MAE                R²     |
[INFO]- +----------------------------------------------------------------------------------------------+
[INFO]- | energy                  9.8935e+00        9.7882e+01        9.8934e+00        -1322.1402     |
[INFO]- | forces                  1.0254e-01        1.0514e-02        7.9312e-02            0.9272     |
[INFO]- | virials                 4.8348e+01        2.3376e+03        2.3044e+01          -10.1441     |
[INFO]- +----------------------------------------------------------------------------------------------+
```

> 💡 Tip: You can also load the trained model into LAMMPS or ASE for molecular dynamics simulations.
