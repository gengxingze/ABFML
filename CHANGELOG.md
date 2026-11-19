# 更新说明

## 2026.11.19

本次更新将 ABFML 整理为面向问题导向力场开发的 PyTorch 框架，统一自定义模型、数据组织、训练评估及原子模拟接口。

### 模型与数据组织

- 引入 `AtomicModel`、`ModelInput` 和 `ModelOutput` 契约，区分几何需求、近邻来源及力的计算方式。
- 使用展平原子存储与 CSR 有向近邻图组织变长结构，按模型需要转换为 Dense 视图。
- 集中定义标签的作用域、形状、单位和归一化规则，电荷字段统一为 `charges`。
- 通过 `ModelSpec` 和 `model_modules` 支持外部模型注册与配置解析，增加完整 Harmonic 示例。
- 整理 ACSF、DeepPot-SE、NEP 和 Lennard-Jones 实现，模型专用统计与配置放回对应模型文件。

### 训练、评估与检查

- 提供断点续训、微调、最佳检查点选择及标准模型保存与加载。
- 训练日志显示模型结构、参数量、设备、精度、运行环境及阶段耗时。
- 训练结束自动生成 `loss.png`；训练和验证合计少于 500,000 个结构时生成合并的 `parity.png`。
- 独立评估打印 MAE、RMSE 和相对误差，并可保存一个包含参考与预测的 `predictions.extxyz` 和一张预测组图。
- 相对误差定义为 `100 * RMSE / mean(abs(reference))`，分母为零时显示 `N/A`。
- `check` 默认生成可复现的测试结构，以双精度检查对称性及能量导数一致性，无需训练数据。

### 分析与原子模拟

- 统一 `abfml plot` 入口，支持标签分布、近邻数量、原子/结构描述符 PCA、预测组图及两列数据导出。
- 提供 ASE Calculator，衔接单点计算、结构优化与原子模拟。
- LAMMPS 使用嵌入 Python 的 `pair_style abfml`，直接加载模型或检查点，无需 TorchScript 或 LibTorch。
- LAMMPS 默认使用 `cpu`、`eager`、`float32`，可显式选择 `compile` 和 `float64`。
- `pair_coeff` 使用元素符号；逐中心 virial 保留九分量，全局六分量采用对称部分。

### 文档与打包

- 新增完整用户手册及按任务拆分的文档网站，支持搜索、公式、流程图与深浅色模式。
- 增加本地预览和 GitHub Pages 构建发布工作流。
- 源码注释、文档字符串和命令帮助统一为英语。
- 完善项目元数据、论文及文档链接，增加 `lammps` 可选安装依赖。
- 修正测试对旧示例原子数、邻居容量及本地 `tmp` 目录的依赖，支持在独立检出目录执行回归验证。

### 迁移注意事项

- 这是一次接口重构，旧版 Python 导入路径、配置字段、模型文件和检查点不保证兼容。
- 旧 `valid` 命令改为 `evaluate`；当前用法为 `abfml evaluate MODEL FILE ...` 和 `abfml check MODEL [STRUCTURE]`。
- 配置文件中的相对数据路径和输出目录相对于配置文件所在目录解析。
- `best.pt` 保存最优监控指标对应的训练状态；`model.pt` 保存最后一轮的标准模型，二者不一定相同。
- 已移除编译导出流程；LAMMPS 通过 Python 桥接运行。旧的桥文件目录参数和原子序数映射写法不再使用。
- LAMMPS 默认精度为 `float32`；与 Python 双精度结果比较时需显式指定 `precision float64`。

安装与完整示例见 [README](README.md) 和 [用户手册](docs/manual.md)。
