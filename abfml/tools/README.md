# MLFF 分析工具

所有命令统一由 `abfml/cli.py` 提供：`abfml plot ...`，或 `python -m abfml plot ...`。
工具不读取 input.json，输入文件、模型和分析参数通过命令或函数参数传入。

## 输出约定

- 默认：在当前目录保存一张 300 dpi PNG 组图，并在终端打印统计表；不导出 CSV / NPZ。
- `--output`：额外导出数值文件，仍保存到当前目录。
- `--output DIR`：图片及显式请求的数值文件保存到 DIR。
- `predictions` 本身只画组图，配对数值通过专用 `pairs` 命令导出。
- 相同目录下的同名文件会覆盖；旧版本产生的其他文件不会自动删除。

Python 函数默认 `output="."`、`export=False`（支持数值导出的函数）；
显式传入 `output` 只改变目录，要导出数值需另外设置 `export=True`。
函数返回统计结果，命令行负责将它们打印成表格。

## 标签分布与分组对比

```powershell
abfml plot labels ../data/test.extxyz
abfml plot labels --dataset structures=../data/structures.extxyz --dataset test=../data/test.extxyz
abfml plot labels ../data/test.extxyz --output
abfml plot labels ../data/test.extxyz --output comparison
```

仅生成 `labels.png`，每标签一个子图，同标签的各组使用共同分箱和概率密度。
终端打印组名、标签、单位、分量数、最小值、最大值、均值和标准差。
`--bins` 默认 60。显式输出数值时，增加 `labels_summary.csv` 和各标签的分箱计数 CSV。

支持现有能量、每原子能量、力分量、力模长、virial、每原子 virial、应力与 charges。
应力由 `-virial/volume` 得到。分量数不是结构数，缺失标签不会填零。
多个位置文件合并成一组；多组对比使用可重复的 `--dataset NAME=FILE`，与位置文件二选一。
同名组多个文件合并。该分析不判断重复结构或数据泄漏。

```python
from abfml.tools import plot_labels
plot_labels("test.extxyz")
plot_labels({"Train": "train.extxyz", "Valid": ["valid1.extxyz", "valid2.extxyz"]})
rows = plot_labels("test.extxyz", output="plots", export=True)
```

`read_labels(files)` 可返回标签到 NumPy 数组的映射；
`plot_distributions(groups, output=".", bins=60, export=False)` 可直接处理具名数值分组。

## 邻居统计

```powershell
abfml plot neighbors ../data/test.extxyz --cutoff 4 5 6 --slots 96
abfml plot neighbors ../data/test.extxyz --cutoff 4 5 6 --slots 96 --output
```

仅生成 `neighbors.png`：不同截断半径分别占一个子图。
终端打印各半径下的结构数、中心数、最小/最大/平均邻居数、P95 和容量溢出百分比。
这些最小值和最大值是邻居数量，不是距离。cutoff 单位 Å。
采用完整有向邻居表，包含周期镜像；`--slots` 只统计 count > slots 的比例，不截断邻居。
显式输出数值时，增加 `neighbors_summary.csv` 和逐中心、分元素计数 `neighbors.csv`。

```python
from abfml.tools import plot_neighbors
rows = plot_neighbors("test.extxyz", cutoffs=[4., 5., 6.], slots=96)
```

## 描述子与 PCA

```powershell
abfml plot descriptors --model results/best.pt --dataset test=../data/test.extxyz --batch-size 8
abfml plot descriptors --model results/best.pt --dataset test=../data/test.extxyz --level structure
```

仅生成 `descriptors.png`，包括共享 PCA 与通道均值/标准差两个子图。
`--level atom`（默认）：每个原子中心一个点，描述局域环境。
`--level structure`：对每个结构所有中心的描述符逐通道取算术平均，一个结构一个点。
平均池化减少原子数差异带来的尺度影响，但会丢失局域分布信息，不是唯一的结构描述方法；
多元素体系这里对所有中心共同平均，不按元素分别池化。
两个层级均先确定分析样本，再共同拟合 PCA 和计算通道均值/标准差，不对 PCA 坐标再求平均。
终端打印层级、结构数、中心数、实际点数、通道数，以及 PC1/PC2 的解释方差百分比。
固定一个模型权重，所有组共同中心化并拟合 PCA，不额外标准化通道。
模型需提供 `properties["descriptors"]`，形状为中心数 × 通道数。
原子层级少于两个中心、结构层级合计少于两个结构时报错，单通道时 PC2 补零。
较大数据集建议预先抽样，完整 SVD 会占用内存。
显式 `--output` 时额外生成 descriptors.npz、pca.csv、explained_variance.csv、channels.csv。
NPZ 保存 level / pooling；原子层级 CSV 记录结构序号、原子序号与元素，结构层级记录结构序号与中心数。
Python 调用使用同名参数：`plot_descriptors(model_path, datasets, level="structure")`。

## 预测组图与两列导出

```powershell
abfml plot predictions evaluation/predictions.extxyz --labels energy_per_atom forces
abfml plot pairs evaluation/predictions.extxyz --label forces
abfml plot pairs evaluation/predictions.extxyz --label energy_per_atom --output energy.csv
```

组图仅输出 `parity.png`，打印各组、各标签的分量数、RMSE 和 MAE。
默认使用每原子能量、力分量、每原子 virial 及其他已有标签，不重复绘制总量。
四边封闭、刻度朝内、横纵轴等比例，包含 y=x 参考线和误差标注。
`pairs` 是显式导出操作，默认写入当前目录的 LABEL.csv；其 `--output` 参数指定 CSV 文件名。
CSV 严格为 `reference,prediction` 两列，按结构及张量 C 顺序展平。
力为 x、y、z；virial 为 xx、xy、xz、yx、yy、yz、zx、zy、zz。标签使用框架原始单位。
`energy_per_atom` 和 `virial_per_atom` 分别除以各结构原子数。
重绘和导出均不调用模型；同时兼容历史 JSONL 预测文件。

训练结束自动生成 loss.png；训练/验证合计少于 50 万结构时，用最佳模型生成合并的 parity.png。
训练不保存评估明细；独立 evaluate 保存 predictions.extxyz 和 parity.png，并返回/打印指标。
extxyz 用 ref_* / pred_* 区分参考与预测，保留元素、坐标、晶胞、PBC 和标签形状。
详细流程见 [训练与评估](../../docs/workflows.md)。

## 损失曲线与公共接口

`plot_loss("loss.out")` 输出当前目录的 loss.png，Train/Valid 同标签共轴，耗时不绘图。
训练结束自动调用，不提供单独 loss 命令。`log=True` 对正值指标使用对数轴。
工具公共函数可从 `abfml.tools` 导入，绘图库按需加载；安装时已包含 matplotlib。
原 tools.mlff、tools.cli、plot_splits 及 splits 命令已移除。
