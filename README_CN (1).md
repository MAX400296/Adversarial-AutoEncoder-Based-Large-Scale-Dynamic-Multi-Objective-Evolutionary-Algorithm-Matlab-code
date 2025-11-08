# AAE-DMOEA-Reproduction（非官方复现）
本仓库为论文 **《Adversarial AutoEncoder-Based Large-Scale Dynamic Multi-Objective Evolutionary Algorithm》**（IEEE TEVC 2024）的**个人复现版**。目标是复现其核心框架：在**动态多目标**场景下，利用**对抗自编码器（AAE）** + **进化算法（MOEA）**实现快速、高质量的**初始种群迁移**与优化。

> 论文要点：通过**余弦角趋势** + **马尔可夫链**预测未来环境中帕累托解的方向性辅助信息；用**迁移损失**训练AAE，使解码样本落入“预测的局部区域”，并在新环境中生成高质量初始种群【8†Adversarial AutoEncoder-Based Large-Scale Dynamic Multi-Objective Evolutionary Algorithm†L246-L318】【8†Adversarial AutoEncoder-Based Large-Scale Dynamic Multi-Objective Evolutionary Algorithm†L319-L390】。

> ⚠️ **声明**：非官方实现，仅用于科研与教学。与原作者无关，错误与不足由本人负责。

---

## ✨ 复现点
- **辅助信息提取**：历史 POS/POF 匹配，计算差分向量的**余弦角**时间序列，离散化并训练**马尔可夫链**预测下一环境的角度区间 `[δ_lower, δ_upper]`【8†Adversarial AutoEncoder-Based Large-Scale Dynamic Multi-Objective Evolutionary Algorithm†L246-L318】。
- **AAE + 迁移损失**：在自编码器上施加**方向约束**，并用对抗网络将聚合后验匹配到简单先验（如 `𝒩(0,I)`）以保持多样性【8†Adversarial AutoEncoder-Based Large-Scale Dynamic Multi-Objective Evolutionary Algorithm†L319-L390】。
- **初始种群生成器**：从先验采样，经解码后结合历史距离统计，合成多样且收敛性好的候选解，作为新环境的初始种群【8†Adversarial AutoEncoder-Based Large-Scale Dynamic Multi-Objective Evolutionary Algorithm†L391-L471】。

---

## 🧱 建议目录结构（按你代码实际调整）
```
.
├─ main.py
├─ configs/
│  └─ df_default.yaml
├─ core/
│  ├─ extractor.py
│  ├─ aae.py
│  ├─ generator.py
│  ├─ moea/
│  │  ├─ nsga2.py
│  │  └─ moead.py
│  └─ utils.py
├─ benchmarks/
│  ├─ df_suite.py
│  └─ metrics.py
├─ results/
│  ├─ runs/
│  ├─ fronts/
│  └─ figures/
└─ requirements.txt
```

---

## 🔧 环境安装
- Python ≥ 3.8
- 建议：PyTorch ≥ 1.12（或 TensorFlow ≥ 2.9，自行更换实现）
- 依赖：NumPy、SciPy、tqdm、Matplotlib、PyYAML

```bash
conda create -n aae-dmoea python=3.10 -y
conda activate aae-dmoea
pip install -r requirements.txt
```

`requirements.txt` 示例（按你代码实际为准）：
```
torch>=1.12
numpy>=1.23
scipy>=1.9
tqdm>=4.64
matplotlib>=3.6
pyyaml>=6.0
```

---

## 🚀 快速上手（命令行示例，按需替换入口/参数）
### 1）在 DF（CEC2018 动态多目标）基准上跑全流程
```bash
python main.py \
  --benchmark DF \
  --dimensions 1000 \
  --moea nsga2 \
  --pop_size 200 \
  --change_period 50 \
  --time_steps 20 \
  --aae_epochs 400 \
  --aae_batch 128 \
  --latent_dim 32 \
  --beta 0.2 \
  --seed 42 \
  --out_dir results/runs/df_nsga2_1k
```

### 2）仅训练 AAE（消融/调试）
```bash
python main.py \
  --stage aae_only \
  --aae_epochs 300 \
  --aae_batch 128 \
  --latent_dim 32 \
  --beta 0.2 \
  --out_dir results/runs/aae_only
```

### 3）用已训练好的 AAE 生成新环境初始种群
```bash
python main.py \
  --stage generate_init \
  --checkpoint results/runs/aae_only/checkpoints/aae_last.pt \
  --benchmark DF \
  --dimensions 300 \
  --moea nsga2 \
  --pop_size 200 \
  --out_dir results/runs/init_pop
```

### 4）评估 IGD/HV 并画图
```bash
python main.py \
  --stage evaluate \
  --run_dir results/runs/df_nsga2_1k \
  --metrics igd hv \
  --plot \
  --save_csv results/runs/df_nsga2_1k/metrics.csv
```

---

## 🧪 关键参数说明
| 参数 | 含义 |
|---|---|
| `--benchmark` | 基准套件，论文使用 DF（CEC2018 动态多目标） |
| `--dimensions` | 决策变量维度（30–1000+） |
| `--moea` | 选用的基础 MOEA（如 `nsga2` / `moead` / 你的实现） |
| `--pop_size` | 种群规模 |
| `--change_period` | 环境变化周期（按代数或评估次数定义） |
| `--time_steps` | 环境状态数量 |
| `--aae_epochs` | AAE 每次训练轮数 |
| `--aae_batch` | AAE 训练批大小 |
| `--latent_dim` | AAE 潜空间维度 |
| `--beta` | 余弦角状态离散的子区间步长（马尔可夫链部分）【8†Adversarial AutoEncoder-Based Large-Scale Dynamic Multi-Objective Evolutionary Algorithm†L318-L390】 |
| `--seed` | 随机种子 |
| `--out_dir` | 输出目录 |
| `--checkpoint` | AAE 权重路径（用于 `generate_init`/恢复） |

---

## 🧠 方法细节（与论文要点对应）
1. **辅助信息提取**：在目标空间按欧氏距离匹配历史 POF，求解 POS 的配对；计算相邻差分向量的余弦，得到每个解的一维角度序列；离散成状态后训练**离散马尔可夫链**，预测下一时刻角度区间【8†Adversarial AutoEncoder-Based Large-Scale Dynamic Multi-Objective Evolutionary Algorithm†L286-L318】【8†Adversarial AutoEncoder-Based Large-Scale Dynamic Multi-Objective Evolutionary Algorithm†L246-L318】。  
2. **AAE + 迁移损失**：用迁移损失强制解码输出 `x'` 满足预测的角度区间（只约束方向，不约束距离，降低负迁移），并用对抗正则把聚合后验拉到高斯先验以增强多样性【8†Adversarial AutoEncoder-Based Large-Scale Dynamic Multi-Objective Evolutionary Algorithm†L319-L390】。  
3. **初始种群生成**：从先验采样/解码，按历史 POS 间距离统计设定步长范围，结合单位方向向量合成候选解，作为新环境 `t+1` 的初始化【8†Adversarial AutoEncoder-Based Large-Scale Dynamic Multi-Objective Evolutionary Algorithm†L391-L471】。

---

## 📈 输出目录
- `runs/<name>/logs/`：日志
- `runs/<name>/checkpoints/`：AAE 权重
- `fronts/`：各时间步的 POF
- `figures/`：IGD/HV 曲线、可视化
- `metrics.csv`：指标表

---

## 🔁 复现实验建议
- 固定 `--seed`；记录环境 `conda env export > env.yml`。
- 需要更强确定性时：
```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python - <<'PY'
import torch
torch.use_deterministic_algorithms(True)
PY
```

---

## 🧩 接入你的 MOEA
实现 `core/moea/<你的算法>.py`，暴露：
```python
def evolve(problem, init_pop, budget, **kwargs) -> dict:
    return {"pos": X_t, "pof": Y_t, "history": ...}
```
框架会在每次环境变化时调用。

---

## ❓常见问题
- **变化后效果不明显** → 提高 `aae_epochs` / 增大 `latent_dim` / 减小 `beta`（更细的角度状态）。  
- **多样性不足** → 增大 `pop_size` / 放宽角度区间 / 对解码样本加高斯扰动。  
- **判别器不稳定** → 降低其学习率 / 加梯度惩罚或谱归一化。

---

## 🔗 引用原论文
```bibtex
@article{Li2024AAE,
  title={Adversarial AutoEncoder-Based Large-Scale Dynamic Multi-Objective Evolutionary Algorithm},
  author={Li, Chenyang and Yen, Gary G. and He, Zhenan},
  journal={IEEE Transactions on Evolutionary Computation},
  year={2024},
  doi={10.1109/TEVC.2024.3412049}
}
```

---

## 📄 许可证
本复现代码以 MIT 协议发布。原论文的文字/图示等版权以期刊/作者规定为准。
