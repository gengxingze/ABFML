# 模型训练示例

在项目根目录运行；每个模型目录包含独立的 `input.json`。

| 目录 | model.name | 描述子 | 当前默认输出（相对项目根目录） |
|---|---|---|---|
| acsf | acsf | 高斯径向、窄角向与宽角向 | `examples/acsf/results/` |
| deepse2a | deep_pot_se_e2_a | 二体全坐标，含 m2=8 | `examples/deepse2a/results/` |
| deepse2r | deep_pot_se_e2_r | 二体径向，无 m2 | `examples/deepse2r/results/` |
| deepse3 | deep_pot_se_e3 | 三体角向，无 m2 | `examples/deepse3/results/` |
| nep | nep | 径向与角向 NEP 描述子 | `examples/nep/results/` |
| lj | lennard_jones | LJ 双体演示势 | `examples/lj/results/` |
| harmonic | harmonic_pair | 自定义谐振双体演示势 | `examples/harmonic/results/` |

```bash
abfml train examples/acsf/input.json
abfml train examples/deepse2a/input.json
abfml train examples/deepse2r/input.json
abfml train examples/deepse3/input.json
abfml train examples/nep/input.json
abfml train examples/lj/input.json
abfml train examples/harmonic/input.json
```

所有配置复用 `data/test.extxyz` 中的 Cu 数据，使用固定种子划分 80% 训练、20% 验证。
默认 CPU、float64、batch_size=2；训练轮数均为 50 个 epoch，可用 `--epochs 1` 做短程检查。
输入与输出路径相对 JSON 所在目录解析，具体输出位置见上表及配置的 output.dir。
小型固定数据集默认缓存邻居；更换大数据集时可将 `data.loader.neighbor_cache` 改为 `none`。

这些是运行示例，未经超参数优化。LJ 用于演示相同训练接口，不代表它能准确拟合 Cu 数据。
Harmonic 配置通过 `model_modules: ["harmonic"]` 加载同目录的 `harmonic.py`，无需修改框架。
该双参数谐振势用于验证自定义模型流程，不包含能量基线或平滑截断，不适合作为 Cu 的实际材料势。
数据只有 Cu，因此 DeepPot-SE 的 `type_slots` 只有一项；更换元素时需同步调整
`common.elements` 和各元素的邻居槽位数。ACSF 的 max_neighbors 是总邻居容量。

三个 DeepPot-SE 示例使用完整的嵌入网络层宽列表，以下为配置方式示意：

```json
"embedding": {
  "network_size": [8, 16, 16],
  "activation": "tanh",
  "bias": true,
  "residual": true
}
```

`network_size` 按顺序列出每层宽度，包含最后一层；标量输入宽度 1 自动补入。
对应网络为 `1 → 8 → 16 → 16`，末项决定嵌入维度，重复的 16 表示两个独立层。
DeepPot-SE2A 的 `m2` 不能大于 `network_size` 的末项。

以 ACSF 为例，训练后可运行：

```bash
abfml evaluate examples/acsf/results/model.pt examples/data/test.extxyz --output examples/acsf/results/evaluation
abfml check examples/acsf/results/model.pt examples/data/test.extxyz
```

上述评估用于检查流程，真实泛化误差应使用独立测试集。
