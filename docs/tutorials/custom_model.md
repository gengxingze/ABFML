# Custom Model Training

This tutorial demonstrates how to build and train a **custom force field model** in ABFML.  
It illustrates how users can implement their own potential form by subclassing `FieldModel`  
and integrating it into ABFML’s modular interface using the **registration** workflow.

---

## 1. Define a Custom Model

To build a new model, specify `"name": "user_defined"` in your parameter file.  
You should also define the model location `"model_path"` and the registered model name  
`"field_name"`. If normalization is needed, include the normalization class name as well.

Here’s a minimal working example showing how to define a user model:

```python
from abfml.core.model.method import FieldModel
from abfml.core.model.builder import register_model, BaseModelBuilder
from typing import List, Dict, Any
import torch


class New(FieldModel):
    def __init__(self, type_map: List[int], cutoff: float, neighbor: int, config: Dict[str, Any]):
        super(New, self).__init__(type_map=type_map, cutoff=cutoff, neighbor=neighbor)
        self.config = config

    def field(self, element_map: torch.Tensor, Zi: torch.Tensor, Zij: torch.Tensor,
              Nij: torch.Tensor, Rij: torch.Tensor, n_ghost: int):
        # Simple example: quadratic interaction
        Ei = Zij ** 2
        Etot = Ei.sum()
        return Etot, Ei


@register_model("New")
class UserModelBuilder(BaseModelBuilder):
    def build(self):
        return New(
            type_map=self.param.GlobalSet.type_map,
            cutoff=self.param.GlobalSet.cutoff,
            neighbor=self.param.GlobalSet.neighbor,
            config=self.param.user_defined.config
        )
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
  "model_path": "example/LJ/LJ-model.py",
  "field_name": "LJ",
  "normal_name": "LJNormal",
  "descriptor": {
    "epsilon": 1.0,
    "sigma": 1.5
  }
}
```

---

## 3. Run the Training

You can train your custom model using ABFML’s command-line interface:

```bash
abfml train  input.json 
```

The training process will automatically detect the `"user_defined"` type  
and build the model from the class registered under `"field_name"`.

---

## 4. Model Validation

Once trained, quickly validate the model’s **symmetry** and **energy conservation** using:

```bash
abfml check -m ./models/new_model.pt
```

This command performs automatic checks to ensure your model’s physical correctness.

---


💡 **Tip**  
You can find this example under the  
[`examples/custom_model/`](https://github.com/gengxingze/ABFML/tree/main/examples/custom_model)  
directory in the ABFML GitHub repository.
