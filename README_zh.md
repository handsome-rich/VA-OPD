<div align="center">

# 👁️ VA-OPD

### Visual-Advantage On-Policy Distillation

**基于视觉优势的视觉语言模型在策略蒸馏**

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2605.21924-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.21924)
[![License](https://img.shields.io/badge/Code-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-yellow.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Stars](https://img.shields.io/github/stars/handsome-rich/VA-OPD?style=social)](https://github.com/handsome-rich/VA-OPD)

[🇬🇧 English](README.md) · [📄 论文](https://arxiv.org/abs/2605.21924) · [📊 实验结果](#-实验结果) · [🚀 快速开始](#-快速开始) · [📚 引用](#-引用)

<img src="assets/teaser.png" width="96%" alt="VA-OPD 提高依赖视觉细节的 token 的蒸馏权重，增强学生模型的视觉依赖。">

</div>

---

**VA-OPD** 使用冻结的教师模型，判断学生生成的回答中哪些 token 依赖细粒度视觉信息。教师分别在原图与像素化图像条件下对同一回答评分，两者对数概率差的正部定义为该 token 的**视觉优势（Visual Advantage，VA）**。

VA 从两个层面指导蒸馏：根据相对视觉依赖**重加权同一问题的多条 rollout**，并对**高 VA 和低 VA 两组 token 分别计算反向 KL 均值**。这样，稀疏的视觉监督获得更大的训练权重，同时保留语言监督。

在论文的 Qwen3-VL-8B → 2B 设置下，VA-OPD 相比 Standard OPD 在全部八个评测集上提升，**数学推理均分提高 2.9 个百分点，视觉理解均分提高 1.5 个百分点**。

---

## ✨ 特点

| | |
|---|---|
| 👁️ **教师衡量视觉依赖** | 比较原图与像素化图像下的 token 概率，保持图像尺寸与视觉 token 对齐。 |
| 🔄 **两级加权** | 提高视觉依赖更强的 rollout 的权重，分别归一化高 VA 与低 VA token 的 KL。 |
| 🧮 **直接优化反向 KL** | 在学生生成的轨迹上计算全词表 student-to-teacher KL。 |
| 📈 **跨规模与数据集收益** | 论文在 4B、8B、32B 教师以及 Geometry3K、ViRL39K 数据上均报告了提升，学生固定为 2B。 |
| 🧩 **聚焦核心方法** | 提供独立目标函数、论文参数配置、Standard OPD 基线及评测协议说明。 |

---

## 🧭 方法流程

<div align="center">
<img src="assets/method.png" width="96%" alt="VA-OPD 流程：学生生成回答，教师在两种图像条件下评分，再进行 rollout 重加权与分组反向 KL 蒸馏。">
</div>

### 1. 计算视觉优势

每个图像与问题生成 $K=4$ 条回答。对于每个回答 token，在原图和像素化图像条件下使用冻结教师评分；问题和回答前缀完全相同：

$$
a_t = \max\!\left(\log p_T(y_t \mid v,q,y_{<t})-\log p_T(y_t \mid \tilde v,q,y_{<t}),\;0\right).
$$

像素化图像 $\tilde v$ 先通过双线性插值缩小至原图**宽、高各 10%**，再通过最近邻插值恢复原尺寸。像素化图像只用于计算 VA；KL 的目标始终来自教师在**原图**条件下的分布。

### 2. 重加权 rollout，并划分 token 组

在每条回答的有效 token 上计算平均 VA，再在同一问题的 $K$ 条回答之间进行标准化，以温度 $\tau=1$ 的 softmax 得到 rollout 权重。每条回答中 VA 最高的 **20%** token 组成 $V$，其余 token 组成 $L$。

### 3. 优化分组反向 KL

$$
\mathcal L = \frac{1}{B}\sum_{b=1}^{B}\sum_{k=1}^{K}w_b^{(k)}
\left[
\frac{\lambda}{|V_b^{(k)}|}\sum_{t\in V_b^{(k)}}\mathrm{KL}_t
+\frac{1-\lambda}{|L_b^{(k)}|}\sum_{t\in L_b^{(k)}}\mathrm{KL}_t
\right],
\qquad \lambda=0.5,
$$

其中 $\mathrm{KL}_t=D_{\mathrm{KL}}(p_S(\cdot\mid v,q,y_{<t})\,\|\,p_T(\cdot\mid v,q,y_{<t}))$。问题与 padding token 不计入损失；教师评分和加权系数不参与梯度传播；最终目标按**问题**取平均。实现见 [`va_opd/objective.py`](va_opd/objective.py)，细节见[复现说明](docs/reproduction.md)。

运行时复用原图教师前向，同时提供 VA 与 KL 所需的分数，再增加一次像素化图像教师前向；数据并行 rank 之间只同步 rollout 统计量。

---

## 📊 实验结果

以下为**论文实验结果**，统一采用 **temperature=1.0 的 avg@8**；各数据集的官方指标约定见[评测协议](docs/evaluation.md)。

### Qwen3-VL-8B → Qwen3-VL-2B · Geometry3K

| 方法 | WeMath | MathVista | MathVerse | 数学均分 | HalluB | AI2D | MMMU | MMStar | OCRBench | 视觉均分 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 36.8 | 63.9 | 19.6 | 40.1 | 52.6 | 77.8 | 49.1 | 56.1 | 85.9 | 64.3 |
| CoT-SFT | 39.7 | 64.4 | 23.9 | 42.7 | 51.0 | 75.7 | 48.7 | 57.1 | 86.2 | 63.7 |
| Off-policy KD | 38.9 | 65.3 | 24.3 | 42.8 | 52.3 | 76.0 | 49.3 | 57.4 | 85.5 | 64.1 |
| GRPO | 44.1 | 64.6 | 24.9 | 44.5 | 54.0 | 77.5 | **52.9** | 57.8 | 84.8 | 65.4 |
| PAPO | 44.8 | 64.5 | 25.8 | 45.0 | 54.0 | 77.8 | 52.7 | 58.0 | 84.7 | 65.4 |
| Standard OPD | 43.3 | 63.7 | 29.1 | 45.4 | 52.0 | 75.8 | 50.9 | 59.7 | 84.7 | 64.6 |
| **VA-OPD** | **46.6** | **66.4** | **31.9** | **48.3** | **54.5** | **78.2** | 51.5 | **59.9** | **86.4** | **66.1** |

### 教师规模与训练数据

| 教师 → 学生 | 训练数据 | Standard OPD 数学 / 视觉 | VA-OPD 数学 / 视觉 | 提升：数学 / 视觉 |
|---|---|---:|---:|---:|
| 4B → 2B | Geometry3K | 45.3 / 64.4 | **47.4 / 65.2** | +2.1 / +0.8 |
| 8B → 2B | Geometry3K | 45.4 / 64.6 | **48.3 / 66.1** | +2.9 / +1.5 |
| 32B → 2B | Geometry3K | 51.1 / 65.3 | **54.8 / 67.3** | +3.7 / +2.0 |
| 8B → 2B | ViRL39K | 46.4 / 65.5 | **50.2 / 68.0** | +3.8 / +2.5 |

### 消融与训练效率

<div align="center">
<img src="assets/ablation.png" width="96%" alt="论文的组件消融及训练轨迹：VA-OPD 的准确率与视觉优势同时提升。">
<br>
<img src="assets/efficiency.png" width="60%" alt="论文在八张 A100 上测得的 MathVerse 准确率与训练时间关系。">
</div>

在论文的 8×A100 设置下，VA-OPD 达到 Standard OPD 最终 MathVerse 准确率约需 **6.5 小时**，Standard OPD 约需 **19.3 小时**，达到相同准确率的速度提高约 **3 倍**。

---

## 🚀 快速开始

### 1. 安装

使用 Python 3.10+ 及与训练依赖兼容的 CUDA 环境。训练依赖固定 **vLLM 0.17.1**，其依赖会解析为 **PyTorch 2.10.0**，并使用 **Transformers ≥4.57、<5**。请使用兼容的 [PyTorch CUDA 版本](https://pytorch.org/get-started/locally/)。

```bash
git clone https://github.com/handsome-rich/VA-OPD.git
cd VA-OPD

pip install -e ".[train,dev]"
pip install flash-attn --no-build-isolation
```

训练运行时使用仓库中的 **verl / EasyR1** 框架。VA-OPD 核心目标函数与公开训练配置位于 `va_opd/` 和 `configs/`，框架许可声明保留在 [NOTICE](NOTICE) 中。

运行目标函数、配置、数据准备与指标汇总检查：

```bash
pip install -e ".[data,dev]"
pytest -q
python -m va_opd.train --config configs/va_opd.yaml --dry-run
```

### 2. 准备 Geometry3K

```bash
python -m va_opd.prepare_data \
  --dataset geometry3k \
  --output-dir data/geometry3k
```

配置保留官方 **2,101 条训练样本**，以种子 **42** 从官方验证划分中选取 **200 条样本**用于选择 checkpoint。准备脚本会在 manifest 中记录选取种子和样本 ID。

### 3. 训练 VA-OPD 或 Standard OPD

```bash
# 默认：Qwen3-VL-8B-Instruct 教师 → Qwen3-VL-2B-Instruct 学生。
bash scripts/train.sh

# 相同训练配置，使用均匀加权的 Standard OPD 目标。
METHOD=opd bash scripts/train.sh

# 更换教师规模。
TEACHER_PATH=Qwen/Qwen3-VL-4B-Instruct bash scripts/train.sh
TEACHER_PATH=Qwen/Qwen3-VL-32B-Instruct bash scripts/train.sh
```

模型路径支持 Hugging Face 标识或本地 checkpoint 目录。论文训练设置使用单节点 **8×A100-80GB** 和 bf16 精度。

### 4. 使用 ViRL39K

先准备 Geometry3K，以使用同一套 200 条样本的 checkpoint 选择集。

```bash
python -m va_opd.prepare_data \
  --dataset virl39k \
  --output-dir data/virl39k

DATASET=virl39k bash scripts/train.sh
```

### 5. 评测

先在留出的 Geometry3K 集合上选择 checkpoint，再评测八个 benchmark。使用各 benchmark 发布的输入提示词，以 **temperature=1.0 为每个样本生成八条回答**，提取首次明确提交的答案，并使用官方 evaluator 与判分提示词评分。数据划分、评分设置与导出格式见 [`docs/evaluation.md`](docs/evaluation.md)。

按该文档中的格式导出官方评分结果，再执行汇总：

```bash
python -m va_opd.evaluation --results official_results.json --output scores.json
```

测试覆盖目标函数、配置、数据准备和指标汇总，并包含两个 rank 的 Gloo 检查。

---

## ⚙️ 论文参数

默认参数见 [`configs/va_opd.yaml`](configs/va_opd.yaml)。Standard OPD 使用同一组训练参数，以控制比较条件。

| 参数 | 论文默认设置 |
|---|---|
| 学生 / 教师 | Qwen3-VL-2B-Instruct / Qwen3-VL-8B-Instruct |
| 每步问题数 / rollout 数 | 16 个问题 / 每个问题 4 条回答 |
| 训练轮数 | 5 |
| 优化器 | AdamW，β₁=0.9，β₂=0.95，weight decay=0.1 |
| 学习率 / 调度 | 1×10⁻⁶ / cosine decay，5% linear warmup |
| 训练采样 | temperature=0.7，top-p=0.95 |
| 回答长度上限 | 4,096 token；超长回答截断 |
| 精度 / 梯度累积 | bf16 / 不使用梯度累积 |
| VA 参数 | 像素化比例=0.10，τ=1.0，ε=10⁻⁶，高 VA 比例=0.2，λ=0.5 |
| 评测 | temperature=1.0，avg@8 |
| 需要模型判分时 | `gpt-4o-2024-08-06`，temperature=0，官方 benchmark 提示词 |
| Checkpoint 选择 | 200 条留出的 Geometry3K 样本 |

Prompt 长度上限为 **8,192 token**。图像预处理、样本选取、token 排序及分布式归一化细节见[实现说明](docs/reproduction.md)。

---

## 📚 引用

```bibtex
@article{liu2026vaopd,
  title={Visual-Advantage On-Policy Distillation for Vision-Language Models},
  author={Liu, Ruiqi and Lv, Xiaolei and Li, Gengsheng and Zhu, Ximo and
          Wang, Zhiheng and Zhang, Zhengbo and Chen, Junkai and Li, Zhiheng and
          Li, Bo and Gao, Jun and Wu, Shu},
  journal={arXiv preprint arXiv:2605.21924},
  year={2026}
}
```

## 📬 联系与许可

- **问题反馈：** [VA-OPD Issues](https://github.com/handsome-rich/VA-OPD/issues)
- **邮箱：** `ruiqi.liu24@nlpr.ia.ac.cn`
- **代码：** [Apache License 2.0](LICENSE)，框架许可声明见 [NOTICE](NOTICE)。
- **插图：** 从作者提供的论文源文件渲染，论文插图的权利归其作者所有。

---

<div align="center">
<sub>👁️ 让依赖视觉的 token，获得应有的学习权重。</sub>
</div>
