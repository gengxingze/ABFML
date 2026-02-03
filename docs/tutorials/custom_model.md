# Custom Model Training

ABFML provides a flexible **Registry Mechanism** that allows users to build, register, and apply their own Machine Learning Force Fields (MLFF) without modifying the core source code.

---

## 1. Overview
To implement a custom model, you need to follow three step
- Inherit FieldModel: Define the energy calculation logic.
- Inherit NormalModel: (Optional) Define data normalization logic.
- Inherit BaseModelBuilder & Register: Use the @register_model decorator to link your code to the configuration system.

## 2. Example: Lennard-Jones (LJ) Potential
The following example demonstrates how to implement a classic LJ potential using the registry system.

A. The Model Logic (LJ)
The field method is the core where the physics happens. ABFML handles the neighbor list and tensor dimensions for you.

```Python
import torch
import torch.nn as nn
from abfml.core.model.method import FieldModel
from typing import List, Dict, Any

class LJ(FieldModel):
    def __init__(self, type_map: List[int], cutoff: float, neighbor: List[int], config: Dict[str, Any]):
        super(LJ, self).__init__(type_map=type_map, cutoff=cutoff, neighbor=neighbor)
        self.config = config
        # Define parameters as trainable or fixed
        self.epsilon = nn.Parameter(torch.tensor([self.config['epsilon']]))
        self.sigma = nn.Parameter(torch.tensor([self.config['sigma']]))

    def field(self, element_map, central_atoms, neighbor_indices, 
              neighbor_types, neighbor_vectors, n_ghost) -> Dict[str, torch.Tensor]:
        """
        neighbor_vectors shape: [batch, n_atom, n_neighbor, 4] 
        where 4 represents [rij, xij, yij, zij]
        """
        device = neighbor_vectors.device
        dtype = neighbor_vectors.dtype
        
        rij = neighbor_vectors[:, :, :, 0]
        mask = (rij > 1e-5) # Avoid division by zero
        
        inv_r = torch.zeros(rij.shape, dtype=dtype, device=device)
        inv_r[mask] = 1.0 / rij[mask]
        
        # 12/6 Lennard-Jones: E = 4*epsilon * [(sigma/r)^12 - (sigma/r)^6]
        ratio = self.sigma * inv_r
        Eij = 0.5 * 4 * self.epsilon * (torch.pow(ratio, 12) - torch.pow(ratio, 6))
        
        Ei = torch.sum(Eij, dim=-1, keepdim=True)
        Etot = torch.sum(Ei, dim=1)
        
        return {'Etot': Etot, 'Ei': Ei}
```
B. The Builder and Registration (UserModelBuilder)
The builder acts as the bridge between your configuration file and the model instance.

```Python
from abfml.core.model.builder import register_model, BaseModelBuilder

@register_model("LJ")
class UserModelBuilder(BaseModelBuilder):
    def build(self):
        """
        The 'self.param' object contains all data from your configuration file.
        """
        return LJ(
            type_map=self.param.GlobalSet.type_map,
            cutoff=self.param.GlobalSet.cutoff,
            neighbor=self.param.GlobalSet.neighbor,
            config=self.param.user_defined.config
        )
```
C. Configuration Usage
Once the model is registered with the string "LJ", you can invoke it in your config.json as follows:
```json
"model_setting": {
          "name": "user_defined",
          "model_path": "../LJ-model.py",
          "field_name": "LJ",
          "normal_name": "LJNormal",
          "descriptor" : {
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
and use your custom model implementation.

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
