# ABFML 的 LAMMPS 接口

本接口通过动态插件在 LAMMPS 中调用 Python 模型，直接加载 ABFML 的 `model.pt` 或训练检查点（如 `best.pt`），无需 TorchScript、AOT 导出或 LibTorch。插件样式为 `abfml`，编译产物为 `libabfml.so`。

以下命令面向 Linux 集群。已使用的 LAMMPS 版本为 **22 Jul 2025 — Update 3**。

## 1. 环境准备

需要 C++17 编译器、CMake 3.18 及以上、MPI，以及提供开发头文件和共享库的 Python 环境。LAMMPS 和插件应使用相同的编译器及 MPI 环境，源码头文件与 LAMMPS 共享库必须匹配。

先激活 Python 环境，再加载集群模块，例如：

```bash
conda activate abfml
module load gcc/16.1.0 mpi/2021.17

export ABFML_ROOT=/data/home/gengxz/work/ABFML
export LAMMPS_ROOT=/data/home/gengxz/program/lammps-22Jul2025

python -m pip install -e "${ABFML_ROOT}[lammps]"
python -c "import torch, abfml; from abfml.interfaces.lammps import Bridge; print(torch.__version__)"
```

模块名称和路径按实际环境修改。编译和运行时均使用这个 Python 环境。桥接模块位于 `abfml.interfaces.lammps`，无需额外复制 Python 桥文件。

## 2. 编译 LAMMPS

```bash
cmake -S "$LAMMPS_ROOT/cmake" -B "$LAMMPS_ROOT/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=mpicxx \
    -DBUILD_MPI=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DPKG_PLUGIN=ON

cmake --build "$LAMMPS_ROOT/build" -j 8
"$LAMMPS_ROOT/build/lmp" -h
```

本插件自行嵌入 Python，不要求开启 LAMMPS 的 `PYTHON` 包。

如果 CMake 提示传统 Make 构建产生的头文件或已安装包冲突，按提示清理后重新配置。以下命令会移除传统 Make 安装到 `src` 的包及生成文件，仅在遇到该冲突时使用：

```bash
make -C "$LAMMPS_ROOT/src" no-all purge
```

## 3. 编译 ABFML 插件

```bash
cmake -S "$ABFML_ROOT/lmp/ABFML" -B "$ABFML_ROOT/lmp/ABFML/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=mpicxx \
    -DLAMMPS_DIR="$LAMMPS_ROOT" \
    -DPython_EXECUTABLE="$(command -v python)" \
    -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"

cmake --build "$ABFML_ROOT/lmp/ABFML/build" -j 4
```

输出为 `$ABFML_ROOT/lmp/ABFML/build/libabfml.so`，可以直接加载，无需再执行安装命令。

`LAMMPS_DIR` 是 **LAMMPS 源码根目录**。默认在其 `src` 查找头文件，在 `build` 或 `build/src` 查找共享库。若使用其他构建目录，可在配置命令中额外指定：

```bash
-DLAMMPS_INCLUDE_DIR=/absolute/path/lammps/src \
-DLAMMPS_LIBRARY=/absolute/path/build/liblammps.so
```

修改 C++ 后重新执行 `cmake --build`。只修改 Python 桥代码时，使用可编辑安装的环境无需重新编译插件，重新启动 LAMMPS 即可。

## 4. 输入文件与参数

以下为单元素 Cu 的最小 `in.lmp`。在包含 `data.lmp`、`model.pt` 的目录运行，并替换实际插件路径：

```lammps
units metal
atom_style atomic
newton on

read_data data.lmp

plugin load /data/home/gengxz/work/ABFML/lmp/ABFML/build/libabfml.so
pair_style abfml model.pt cpu
pair_coeff * * Cu

neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes

thermo 1
thermo_style custom step atoms pe press pxx pyy pzz pxy pxz pyz
run 0
```

`run 0` 计算当前构型的能量、力和压力，不推进动力学。模型路径相对于运行目录解析；截断半径从模型读取。

完整语法：

```lammps
pair_style abfml MODEL [DEVICE] [mode eager|compile] [precision float32|float64]
pair_coeff * * ELEMENT1 ELEMENT2 ...
```

| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| `MODEL` | 模型或检查点路径 | 必填 |
| `DEVICE` | PyTorch 设备，如 `cpu`、`cuda:0` | `cpu` |
| `mode` | 普通执行或 `torch.compile` | `eager` |
| `precision` | 模型参数、浮点 buffer 和输入的计算精度 | `float32` |

设备参数必须位于关键字之前；`mode` 与 `precision` 的顺序可以交换。例如：

```lammps
pair_style abfml model.pt
pair_style abfml model.pt cpu mode eager precision float64
pair_style abfml model.pt cuda:0 mode compile precision float32
```

**默认采用 float32，不沿用模型保存精度。** 返回 LAMMPS 时转换成 double，不会恢复单精度计算损失的信息；需要双精度比较时显式指定 `precision float64`。

`pair_coeff` 按 LAMMPS 原子类型顺序填写元素符号，数量必须与原子类型数相同。例如 `pair_coeff * * Cu O` 将类型 1、2 分别映射为 Cu、O。仅接受大小写正确的元素符号，不接受原子序数，且元素必须属于模型支持的类型。

## 5. 运行与并行

在已申请的计算资源内运行，并激活编译插件所用的 Python 环境、加载相同的编译器和 MPI 模块：

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# 单进程
"$LAMMPS_ROOT/build/lmp" -in in.lmp

# 四个 MPI 进程；启动命令按集群调度系统调整
mpiexec -n 4 "$LAMMPS_ROOT/build/lmp" -in in.lmp
```

每个 MPI 进程独立加载一份模型，桥接层将 PyTorch CPU 线程数设为 1。LAMMPS 提供本地中心原子的完整有向 CSR 近邻表，以及包含周期像的相对位移；Python 不重新构建近邻。本地和 ghost 原子的力由模型返回，再由 LAMMPS 完成反向通信。

多 GPU 运行需要在启动环境中安排进程与设备的对应关系；字面值 `cuda:0` 会使所有进程选择各自可见的第 0 张 GPU。

`compile` 使用 `torch.compile(..., dynamic=True)`，首次调用包含编译开销，尺寸变化也可能触发重新编译。应通过预热后的实际体系测量加速效果。编译失败会明确报错，可显式改回 `eager`。当前接口每步仍存在输入复制、设备传输和输出回传开销。

## 6. 模型要求与输出

模型须使用 `geometry_mode='local'`、`neighbor_mode='external'`，并正确处理本地中心与 local/ghost 原子。自定义模型还需确认其依赖范围与 MPI 分域方式相容。

| 输出 | 使用条件 | 每个进程中的形状 |
| --- | --- | --- |
| `forces` | 始终需要 | `[nlocal + nghost, 3]` |
| `energy` | 请求总势能 | 本地中心能量之和的标量 |
| `atomic_energies` | 请求逐原子势能 | `[nlocal]` |
| `virial` | 请求全局 virial / 压力 | `[3, 3]`，本地贡献 |
| `atomic_virial` | 请求逐原子九分量 virial | `[nlocal, 3, 3]` |

能量求导力和直接预测力的模型均可接入，但必须提供本次 LAMMPS 请求的物理量。Python 返回的 `charges` 目前不会写入 LAMMPS 原子电荷，也不提供额外的电荷平衡或长程静电计算。

当前仅支持 `units metal`，面向无化学键拓扑的原子体系，并要求 `newton on`。建议先比较单进程 `run 0` 与 Python/ASE 的能量、力和 virial，再比较多进程结果，最后运行动力学。

## 7. Virial 与逐原子九分量

对于能量求导模型，相对向量采用 `r_ij = r_j - r_i`，中心原子归属定义为：

$$
W_i = -\sum_j r_{ij}\otimes\frac{\partial E}{\partial r_{ij}},
\qquad W = \sum_i W_i.
$$

逐原子结果完整保留九分量，不转置、不对称化。读取示例：

```lammps
compute av all centroid/stress/atom NULL virial
dump dv all custom 1 atomic_virial.dump id type c_av[1] c_av[2] c_av[3] c_av[4] c_av[5] c_av[6] c_av[7] c_av[8] c_av[9]
```

顺序为 `xx yy zz xy xz yz yx zx zy`。这里借用 `centroid/stress/atom` 的九分量接口，实际采用 ABFML 的**中心原子归属**，并非几何质心分配；六分量 `stress/atom` 不受支持。

全局 virial 写入 LAMMPS 的六分量接口，非对角项取完整张量的对称部分，例如 `(W[0,1] + W[1,0]) / 2`。

模型原始 virial 单位为 eV，不除体积。上述 `compute` 按 LAMMPS 应力约定带负号，并进行 `metal` 单位换算，输出属于压力乘体积量，不能直接当作模型原始的 eV 数组。比较时需统一符号、体积及单位；`thermo` 压力还可能包含动能项。

## 8. 常见问题

| 问题 | 处理方式 |
| --- | --- |
| CMake 提示传统 Make 构建冲突 | 按第 2 节清理旧生成文件，再重新配置 |
| 找不到 Python 开发文件或 pybind11 | 确认环境提供 Python 头文件和共享库，显式指定 `Python_EXECUTABLE`、`pybind11_DIR` |
| `GLIBCXX_*` / `CXXABI_*` not found | 加载编译 LAMMPS 时使用的 GCC 模块，避免使用系统旧版 C++ 运行库 |
| `No module named abfml` | 在构建插件所用的 Python 环境安装当前 ABFML，并在运行时激活该环境 |
| 找不到 `liblammps.so` 等共享库 | 检查库路径和编译环境，必要时将实际库目录加入 `LD_LIBRARY_PATH` |
| `Unrecognized pair style` | 先加载本次编译的 `libabfml.so`，再使用 `pair_style abfml ...` |
| 参数数量或 Usage 错误 | 不再传入旧桥文件目录；使用第 4 节的参数格式 |
| 元素映射报错 | 每个原子类型填写一个标准元素符号，并确认模型支持该元素 |
| 请求的模型输出缺失 | 检查对应物理量及其形状是否符合第 6 节 |

插件的 Python 对象在运行期间持续存在，不支持在对象仍存活时卸载插件。重新编译或更新环境后，应重新启动 LAMMPS。

官方参考：[CMake 构建](https://docs.lammps.org/Build_cmake.html)、[插件加载](https://docs.lammps.org/plugin.html)。
