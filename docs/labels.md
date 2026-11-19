# 物理标签说明

唯一的标签定义位于 `abfml/core/schema.py`，通过 `LABELS` 和不可变的 `LabelSpec` 描述。
单位、作用域、分量形状和归一化规则不再由日志、评估和绘图分别维护。
`N` 表示存储的原子数，`C` 表示能量中心数，`B` 表示完整结构批次数。

| 标签 | 作用域 | 模型或批次形状 | 内部单位 | 损失归一化 | 训练误差显示 |
|---|---|---|---|---|---|
| energy | structure | 标量或 `[B]` | eV | 每结构除以原子数 | 每原子误差，meV/atom |
| atomic_energies | center | `[C]` | eV | 无 | eV |
| forces | atom | `[N,3]` | eV/A | 无，按分量统计 | meV/A |
| virial | structure | `[3,3]` 或 `[B,3,3]` | eV | 每结构除以原子数 | 每原子分量误差，meV/atom |
| charges | atom | `[N]` | e | 无 | e |
| total_charge | structure | 标量或 `[B]` | e | 无 | e |
| dipole | structure | `[B,3]` | e*A | 无 | e*A |
| born_effective_charges | atom | `[N,3,3]` | e | 无 | e |
| atomic_virial | center | `[C,3,3]` | eV | 无 | eV |

`A` 为 Å 的 ASCII 写法，避免终端编码问题。`virial` 不除以体积；应力才使用
`stress = -virial / volume`，单位 eV/A³。此处没有改变已有 virial 分量和符号约定。
逐原子能量和逐中心 virial 在完整训练结构中 `C=N`；LAMMPS 中中心仅为 local，
存储原子可包含 ghost，必须区分 `C` 与 `N`。

## 字段含义

- `scope`：`atom`、`center` 或 `structure`。
- `shape`：单个实体的分量形状，例如力为 `(3,)`，virial 为 `(3,3)`；不包含 N/C/B 维。
- `unit`：内部数值和 evaluate 返回指标的单位。
- `per_atom`：需要按结构原子数归一化时，对应的衍生指标名；当前为 energy_per_atom、virial_per_atom。
- `display_unit` / `display_scale`：日志显示单位与倍数；不修改模型输出、监督标签或 extxyz 数值。
- `sample_field`：是否为 AtomicSample 的固定标签字段；其余标签使用 extra_labels。
- `source`：衍生量的来源，不代表任意衍生量都会被自动计算。

衍生量包括 energy_per_atom、virial_per_atom、force_norm 和 stress。
损失与训练/评估指标共用 `normalize_label()`，对 energy、virial 按各结构原子数归一化；
电荷、偶极和其他标签不会自动除以原子数。

## 评估指标

evaluate 保留并向用户返回 MAE、RMSE 和相对误差，按完整数据集的标签分量统计：

```text
MAE  = sum(abs(prediction - reference)) / count
RMSE = sqrt(sum((prediction - reference)^2) / count)
relative_error_percent = 100 * RMSE / mean(abs(reference))
```

参考绝对值均值为零时，相对误差为 None，表格显示 N/A。能量和 virial 同时报告原始总量
与每原子指标；分子、分母采用一致归一化。MAE 和 RMSE 的返回值使用内部单位，
不会因为训练日志显示 meV 而变成 meV。
训练结束的最佳模型表仍显示 RMSE 与相对误差；独立 evaluate 的返回值和 CLI 输出保留 MAE。

## 扩展标签

已有标签的作用域固定，collate 不允许通过 label_scopes 将其改成另一个作用域。
未知标签仍可使用 extra_labels 和显式 label_scopes，不强制封闭标签集合；没有单位定义时
明确标记为 `label_units`。

需要框架识别新标签的形状、作用域和单位时，可在模型扩展模块导入阶段补充说明，例如：

```python
from abfml.core.schema import LABELS, LabelSpec

LABELS["polarizability"] = LabelSpec("structure", (3, 3), "A^3")
```

这会使样本校验、批次作用域、评估切片与单位展示读取同一份说明。
数据源的读取映射和具体模型预测仍需由各自实现提供；标签说明不自动训练或预测新物理量。
center 量不会自动套用 atom 作用域；完整结构的中心标签如需作为 extra_labels 监督，
应显式声明其完整结构拼接方式。结构电荷标签也不会自动变成模型输入或启用长程静电。
