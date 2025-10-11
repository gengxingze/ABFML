# Custom Model Training

This tutorial demonstrates how to build and train a **custom force field model** in ABFML.  
It illustrates how users can implement their own potential form by subclassing `FieldModel`  
and integrating it into ABFML’s modular interface.

---

## 1. Define a Custom Model

To build a new model, specify `"name": "user_defined"` in your parameter file.  
You should also define the model location `"model_path"` and the required descriptor settings.  
If normalization is needed, include the normalization class name as well.

Here’s a minimal working example showing how to define a user model:

```python
from abfml import FieldModel
from typing import List, Dict, Any
import torch


class New(FieldModel):
    def __init__(self, type_map: List[int], cutoff: float, neighbor: int, config: Dict[str, Any]):
        super(New, self).__init__(type_map=type_map, cutoff=cutoff, neighbor=neighbor)

    def field(self, element_map: torch.Tensor, Zi: torch.Tensor, Zij: torch.Tensor,
              Nij: torch.Tensor, Rij: torch.Tensor, n_ghost: int):
        # Simple example: quadratic interaction
        Ei = Zij ** 2
        Etot = Ei.sum()
        return Etot, Ei
```

!!! tip "💡 Tip"
    All custom models should inherit from `FieldModel` and implement the `field()` function.  
    The `field()` method defines how total and per-atom energies are computed from descriptor tensors.

---

## 2. Example Parameter Configuration (`input.json`)

Below is a minimal configuration file example to run your custom model:

```json
{
  "name": "user_defined",
  "model_path": "./models/new_model.pt",
  "descriptor": {
    "type_map": [29],
    "neighbor": [100],
    "cutoff": 6.0
  },
  "normalize": {
    "type": "None"
  }
}
```

---

## 3. Run the Training

You can train your custom model using ABFML’s command-line interface:

```bash
abfml train --input input.json --data ./data/train_dataset.npz
```

The training process will automatically detect the `"user_defined"` type  
and use your custom model implementation.

---

## 4. Model Validation

Once trained, quickly validate the model’s **symmetry** and **energy conservation** using:

```bash
abfml check --model ./models/new_model.pt
```

This command performs automatic checks to ensure your model’s physical correctness.

---


💡 **Tip**  
You can find this example under the  
[`examples/custom_model/`](https://github.com/gengxingze/ABFML/tree/main/examples/custom_model)  
directory in the ABFML GitHub repository.
