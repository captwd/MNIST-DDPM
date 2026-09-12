# MNIST-DDPM

从零实现的 [DDPM](https://arxiv.org/abs/2006.11239)（Ho et al., 2020）+ 统一协议下的 MNIST 分类模型横向对比，以及 DDPM 生成数据的下游可用性评估。

> 在 [explainingai-code/DDPM-Pytorch](https://github.com/explainingai-code/DDPM-Pytorch) 基础上：修复了数据路径/归一化不一致等问题，新增统一对比框架 `compare/` 与一键实验脚本 `tools/run_comparison.py`，并附[静态结果页](demo/index.html)。

## 实验设计

两条链路，全部模型在**同一份官方 MNIST 测试集（10k）**上评估 Top-1 准确率：

```
链路一（真实数据）：官方 MNIST 60k → 统一预处理 [-1,1] → ANN / CNN / ViT / SVM / KNN / RF / LR → 测试集评估
链路二（DDPM 合成）：已训练 DDPM → 1000 步反向采样 2000 张 → 最优分类器打伪标签（置信度≥0.9，保留 94.2%）
                     → 仅用合成数据从零训练 ANN / CNN / ViT → 同一测试集评估
```

**对齐协议**：官方 60k/10k 划分 · `Normalize((0.5,),(0.5,)) → [-1,1]` · batch 64 · 10 epochs · Adam(1e-3)+StepLR(5,0.5) · CrossEntropy · seed 42 · num_workers=0 · 传统 ML 输入同样对齐到 [-1,1]。

## 结果

### 真实数据训练（RTX 5060 · 2026-09 实测）

| 模型 | 参数量 | 训练时长 (s) | 最佳测试准确率 |
|---|---|---|---|
| **CNN** ★ | 421,834 | 92.3 | **0.9931** |
| SVM (RBF) | — | 198.4 | 0.9792 |
| ViT (16 patch · dim128 · 6 层) | 1,199,882 | 238.6 | 0.9732 |
| RandomForest (100 树) | — | 25.4 | 0.9705 |
| KNN (k=5) | — | 9.3 | 0.9688 |
| ANN (784-15-10 · Sigmoid) | 11,935 | 85.0 | 0.9338 |
| LogisticRegression | — | 10.5 | 0.9225* |

### DDPM 合成数据训练（1,883 张 · 伪标签来自 CNN）

| 模型 | 最佳测试准确率 | vs 真实数据训练 |
|---|---|---|
| **CNN** ★ | **0.9601** | −3.3pp |
| ViT | 0.8052 | −16.8pp |
| ANN | 0.6658 | −26.8pp |

\* LR 100 迭代未收敛（ConvergenceWarning），数值为下限。

**主要结论**

1. **CNN 压倒性第一**：卷积归纳偏置在 28×28 小图 + 短训练下收益巨大；ViT 参数最多却落后约 2pp 且仍在爬升（欠训练特征——ViT 需大数据/强增广）。
2. **SVM-RBF 是被低估的强基线**：反超 ViT 与 ANN，距 CNN 仅 1.4pp——"深度全面占优"在 MNIST 上不成立，只有 CNN 占优。
3. **DDPM 生成数据下游可用**：只用真实数据量 1/32 的合成样本，CNN 即达 96%；掉点主因是数据规模/多样性而非模型容量（容量最小的 ANN 掉最狠）。

## 仓库结构

```
├── config/default.yaml        # DDPM 配置（1000 步线性调度 · U-Net 28×28 单通道）
├── dataset/mnist_dataset.py   # MNIST 数据集（PNG / torchvision IDX 双模式）
├── models/unet_base.py        # U-Net（仿 diffusers 结构）
├── scheduler/                 # 线性噪声调度器（前向加噪 / 反向采样）
├── tools/
│   ├── train_ddpm.py          # 训练 DDPM
│   ├── sample_ddpm.py         # 采样生成
│   └── run_comparison.py      # ★ 一键对比实验（本工作新增）
├── compare/                   # ★ 统一对比框架（本工作新增）
│   ├── data.py                #   统一数据加载（复用仓库根 MNIST/raw，零下载）
│   ├── models.py              #   ANN / CNN / ViT
│   ├── train_utils.py         #   统一训练协议 + 逐 epoch 曲线
│   ├── traditional.py         #   SVM / KNN / RF / LR
│   └── ddpm.py                #   DDPM 批量采样 + 伪标签
├── other/                     # 早期独立脚本（已对齐路径与归一化，保留作参考）
├── demo/index.html            # ★ 纯静态实验报告页（设计 + 结果 + 分析）
└── compare_results/           # 曲线图 / 生成预览 / summary（大文件不入库）
```

## 快速开始

```bash
# 环境：Python 3.10+，见 requirements.txt
pip install -r requirements.txt

# 1) 训练 DDPM（40 epochs，输出到 default/）
python -m tools.train_ddpm

# 2) 采样预览（输出 default/samples/）
python -m tools.sample_ddpm

# 3) 一键对比实验（深度 3 模型 × 真实/合成 + 传统 ML + DDPM 采样 2000 张）
python -m tools.run_comparison
#     --num-synthetic 60000   加大合成数据规模
#     --with-augment          增加真实+合成混合训练组
#     --skip-generation       复用已生成的合成数据
#     --smoke                 小样本快速验证管线

# 产物：compare_results/summary.md · summary.csv · curves/*.png · ddpm_preview.png
```

GPU 显存 ≥ 4GB 即可；无 GPU 时脚本自动回落 CPU（建议减小 `--num-synthetic`）。

详细实验设计、逐 epoch 曲线与结果分析见 **[demo/index.html](demo/index.html)**（纯静态，可直接浏览器打开或 GitHub Pages 托管）。

## Citation

```bibtex
@misc{ho2020denoising,
      title={Denoising Diffusion Probabilistic Models},
      author={Jonathan Ho and Ajay Jain and Pieter Abbeel},
      year={2020},
      eprint={2006.11239},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
