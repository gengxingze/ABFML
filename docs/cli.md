# Command-Line Interface (CLI)

ABFML provides a command-line interface for quick model checking, training, and validation.  
You can display general help using:
```bash
abfml --help
```

---
Check the validity and inference capability of a trained model.
```bash
abfml check -m model.pt -d float32
```
- **Purpose**: Ensures the model file can be loaded and used for inference.

---

### `train`
Start training a machine learning force field model based on a configuration file.
```bash
abfml train input.json
```
- **Purpose**: Launches the training process using parameters defined in `input.json`.

---

### `valid`
Validate a trained model using a test dataset.
```bash
abfml valid -m model.pt -f "../data/test.extxyz" -n 10
```
- **Purpose**: Evaluates model performance with sp

## Available Commands

### `check`ecified test data and sample size.

---

## Help for Commands
Each command provides detailed parameter information using `-h`:
```bash
abfml train -h
abfml valid -h
abfml check -h
```
