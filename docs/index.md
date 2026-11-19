---
hide:
  - toc
---

<div class="home-hero" markdown>

<p class="eyebrow">MACHINE LEARNING FORCE FIELDS · PYTORCH</p>

# ABFML

<p class="lead">面向物理问题，构建与验证机器学习力场。</p>

<p class="description">构建自己的描述符和力场模型，复用训练与验证流程，再通过统一接口接入 ASE 与 LAMMPS，将方法开发连接到原子模拟。</p>

[快速开始](quickstart.md){ .md-button .md-button--primary }
[探索模型与方法](custom.md){ .md-button }

</div>

<div class="entry-grid" markdown>

<div class="entry-card" markdown>

## 开始训练

安装环境、准备数据，完成训练与独立评估。

[安装与环境 →](install.md) · [配置与训练 →](training.md)

</div>

<div class="entry-card" markdown>

## 开发新模型

定义描述符与预测逻辑，复用训练和验证流程。

[自定义模型 →](custom.md) · [接口与注册 →](model-extension.md)

</div>

<div class="entry-card" markdown>

## 自定义模型用于模拟

内置与自定义模型共用 LAMMPS Pair 接口，无需逐模型编写 C++。

[ASE →](ase-and-checks.md) · [LAMMPS →](lammps.md)

</div>

</div>

## 一套 LAMMPS 接口，运行自定义力场

**任意满足 ABFML 接口契约的自定义模型，都可以通过统一的 `pair_style abfml` 在 LAMMPS 中运行。** 模型实现并注册后，可复用训练、检查和保存流程，由 Python 桥接直接加载，无需为每个模型单独编写 C++ Pair 类，也无需 TorchScript 导出或 LibTorch 链接。

当前接口支持 `local` 几何、`external` 近邻，并要求正确处理本地中心与 local/ghost 原子、提供模拟请求的物理量。多层消息传递等模型还需验证其依赖范围与 MPI 分域方式是否相容。

[实现自定义模型 →](custom.md) · [编译与使用 LAMMPS 插件 →](lammps.md)

## 从局域环境到力场应用

<figure class="framework-figure" markdown>

![ABFML 数据组织、模型训练与模拟应用流程](assets/architecture.svg)

<figcaption>统一原子与近邻表示连接模型开发、数值验证和模拟应用；模型输出遵循相同的物理量约定。</figcaption>

</figure>

## 运行第一个示例

在项目根目录执行，以附带的 Cu 数据完成一轮 ACSF 训练：

```bash
python -m pip install -e .
abfml train examples/acsf/input.json --epochs 1
abfml check examples/acsf/results/best.pt
```

接下来可以[评估预测误差](evaluation.md)、[分析数据与描述符](tools.md)，或阅读[完整用户手册](manual.md)。该示例用于熟悉流程，一轮训练不代表模型已经收敛。

## 方法与实现

| 研究内容 | 文档 |
| --- | --- |
| 原子中心对称函数 | [ACSF 描述符与配置](acsf.md) |
| 可学习环境嵌入 | [DeepPot-SE](deepse.md) |
| 径向基与角向矩收缩 | [NEP](nep.md) |
| 能量导数和张量约定 | [力与 Virial](mechanics.md) |
| 变长结构、近邻图与模型契约 | [架构与数据流](architecture.md) |

## 论文与引用

X. Geng, J. Gu, G. Qin, L.-W. Wang, and X. Meng. **ABFML: A problem-oriented package for rapidly creating, screening, and optimizing new machine learning force fields.** *J. Chem. Phys.* **162**, 052502 (2025). [DOI: 10.1063/5.0247559](https://doi.org/10.1063/5.0247559)

[引用条目](project.md#引用) · [GPL-3.0-only](license.md) · [文档维护与发布](publishing.md)
