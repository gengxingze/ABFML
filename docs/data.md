# 数据层

`ASEReader → AtomicDataset → DataLoader / StructureCollator → (ModelInput, 标签字典)`

Reader 逐帧解析文件；Dataset 保存样本并提供索引；Collator 组装批次和可选邻居。
邻居图、搜索、后端选择和缓存统一由 `abfml.core.neighbor` 提供，reader 不再包含邻居逻辑。
模型通过 `neighbor_mode` 声明是否需要外部邻居。小型固定结构数据集可启用
`neighbor_cache="memory"`，跨批次和 epoch 复用；模型自行建边时自动跳过。

```python
from abfml.data import ASEReader, AtomicDataset, create_dataloader

dataset = AtomicDataset(ASEReader("structures.extxyz"))
loader = create_dataloader(dataset, model=model, batch_size=8, neighbor_cache="memory")
for inputs, labels in loader:
    output = model(inputs)
```

接口、张量约定、作用域与缓存内存说明见[架构与数据流](architecture.md)。
可运行示例见[项目入口](../README.md)。
