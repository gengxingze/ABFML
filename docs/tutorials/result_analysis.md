# Results Analysis

In this section, we analyze the performance of the trained machine learning force field (MLFF) model through several key visualization methods.
These plots provide intuitive insights into **model convergence, accuracy, and error distribution**, which are critical for evaluating and improving MLFF models.

---

## 1. Convergence Curve

The convergence curve shows how the total loss (and its components such as energy and force loss) evolve during training.
It reflects whether the training process is stable and if the model parameters are converging properly.

**Purpose in MLFF training:**

* Monitor model stability and overfitting.
* Evaluate the balance between energy and force fitting.
* Determine whether more epochs or different learning rate schedules are needed.

**Example plot:**

```

```

![Convergence Curve](../images/convergence.png)

---

## 2. Parity Plot 

The parity (diagonal) plot compares the predicted values (energy or force) with the reference DFT values.
Each point represents one structure (for energy) or one atomic component (for force).

**Purpose in MLFF training:**

* Visualize how well the MLFF reproduces DFT data.
* Identify systematic bias (if points deviate from the diagonal).
* Quantify the root-mean-square error (RMSE) and correlation (R²).

**Example plot:**

```python
```

![Parity Plot](../images/parity.png)

---

## 3. Error Distribution Plot

The error distribution (histogram) shows how the prediction errors (ΔE = E_pred − E_ref or ΔF = F_pred − F_ref) are distributed.
A narrow, symmetric distribution centered around zero indicates a well-fitted and unbiased model.

**Purpose in MLFF training:**

* Identify systematic over- or underestimation.
* Evaluate uncertainty and variance in model predictions.
* Compare model performance across datasets or architectures.

**Example plot:**

```python
```

![Error Distribution](images/error_distribution.png)

---

## Summary

| Plot Type              | Description                      | Insight Provided                             |
| ---------------------- | -------------------------------- | -------------------------------------------- |
| **Convergence Curve**  | Tracks loss over training steps  | Convergence stability, overfitting detection |
| **Parity Plot**        | Compares MLFF vs DFT predictions | Accuracy and bias visualization              |
| **Error Distribution** | Histogram of prediction errors   | Spread and symmetry of model performance     |

These plots together form a comprehensive evaluation toolkit for MLFF model training and validation.
They help researchers quickly identify performance bottlenecks and optimize model hyperparameters for better accuracy and transferability.
