# AAE-DMOEA-Reproduction
_Unofficial reproduction of_ **"Adversarial AutoEncoder-Based Large-Scale Dynamic Multi-Objective Evolutionary Algorithm"** (IEEE TEVC 2024).  
This project reimplements the **AAE-DMOEA** framework: an Adversarial AutoEncoder (AAE) + MOEA pipeline for large-scale dynamic multi-objective optimization (DMOP).

> Paper: Adversarial AutoEncoder-Based Large-Scale Dynamic Multi-Objective Evolutionary Algorithm (Li, Yen, He).  
> DOI: 10.1109/TEVC.2024.3412049. Key ideas include auxiliary-information extraction via **cosine-angle trends** + **Markov chain predictor**, an **AAE** trained with a **transfer loss**, and a generator that yields high-quality initial populations for new environments【8†Adversarial AutoEncoder-Based Large-Scale Dynamic Multi-Objective Evolutionary Algorithm†L246-L318】【8†Adversarial AutoEncoder-Based Large-Scale Dynamic Multi-Objective Evolutionary Algorithm†L319-L390】.

> ⚠️ **Disclaimer**: This is a personal, unofficial reproduction for research/education. Not affiliated with the original authors. Any inconsistencies or bugs are my own. If the original author finds that the code does not match the actual implementation, please contact me.

---

## ✨ What’s inside
- **Auxiliary info extraction** from historical POS/POF using cosine angle trends + Markov chain predictor【8†Adversarial AutoEncoder-Based Large-Scale Dynamic Multi-Objective Evolutionary Algorithm†L246-L318】.
- **AAE training with transfer loss** to constrain decoded samples into predicted local regions of the next environment【8†Adversarial AutoEncoder-Based Large-Scale Dynamic Multi-Objective Evolutionary Algorithm†L319-L390】.
- **Initial population generator** sampling from a matched latent prior to produce diverse, convergent individuals for the new environment【8†Adversarial AutoEncoder-Based Large-Scale Dynamic Multi-Objective Evolutionary Algorithm†L391-L471】.
- Plug-and-play with **your MOEA of choice** (e.g., NSGA-II/MOEA-D/LMOEA).

---

## 🧱 Suggested repo structure
> Your actual code may differ; update paths accordingly.

```
.
├─ main.py                    # Entry: orchestration (detect change, extract aux, train AAE, generate init pop, evolve)
├─ configs/
│  └─ df_default.yaml         # Benchmark/algorithm settings
├─ core/
│  ├─ extractor.py            # cosine/Markov chain; POS-POF matching
│  ├─ aae.py                  # encoder/decoder/discriminator + transfer loss
│  ├─ generator.py            # latent sampling & population synthesis
│  ├─ moea/
│  │  ├─ nsga2.py
│  │  └─ moead.py
│  └─ utils.py
├─ benchmarks/
│  ├─ df_suite.py             # CEC2018 dynamic problems (a.k.a. DF)
│  └─ metrics.py              # IGD/HV etc.
├─ results/
│  ├─ runs/                   # logs, checkpoints
│  ├─ fronts/                 # POF snapshots
│  └─ figures/                # plots
└─ requirements.txt
```

---

## 🔧 Environment
- Python ≥ 3.8
- One of: **PyTorch ≥ 1.12** or **TensorFlow ≥ 2.9** (this repo assumes PyTorch by default)
- NumPy, SciPy, tqdm, Matplotlib
- (Optional) CuDNN/CUDA for GPU

### Install (conda)
```bash
conda create -n aae-dmoea python=3.10 -y
conda activate aae-dmoea
pip install -r requirements.txt
```

Example `requirements.txt` (edit to match your code):
```
torch>=1.12
numpy>=1.23
scipy>=1.9
tqdm>=4.64
matplotlib>=3.6
pyyaml>=6.0
```

---

## 🚀 Quickstart
> Below are **typical** commands. If your entrypoint or flags are different, adapt accordingly.

### 1) Run the full pipeline on the DF (CEC2018) suite
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

### 2) Only train the AAE (ablation / debug)
```bash
python main.py \
  --stage aae_only \
  --aae_epochs 300 \
  --aae_batch 128 \
  --latent_dim 32 \
  --beta 0.2 \
  --out_dir results/runs/aae_only
```

### 3) Use pre-trained AAE to generate an initial population
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

### 4) Evaluate IGD/HV over time
```bash
python main.py \
  --stage evaluate \
  --run_dir results/runs/df_nsga2_1k \
  --metrics igd hv \
  --plot \
  --save_csv results/runs/df_nsga2_1k/metrics.csv
```

---

## 🧪 Key arguments
| Flag | Meaning |
|---|---|
| `--benchmark` | Benchmark suite. Use `DF` (CEC2018 dynamic multi-objective) as in the paper. |
| `--dimensions` | #decision variables (30–1000+ supported) |
| `--moea` | Base MOEA (e.g., `nsga2`, `moead`, or your own) |
| `--pop_size` | Population size |
| `--change_period` | Environmental change period (evaluations or generations) |
| `--time_steps` | #environmental states to simulate |
| `--aae_epochs` | AAE training epochs per change |
| `--aae_batch` | Batch size for AAE |
| `--latent_dim` | Latent space dim in AAE |
| `--beta` | Sub-interval step for discretizing cosine states in the Markov chain【8†Adversarial AutoEncoder-Based Large-Scale Dynamic Multi-Objective Evolutionary Algorithm†L318-L390】 |
| `--seed` | RNG seed |
| `--out_dir` | Output root |
| `--checkpoint` | Path to AAE weights (for `generate_init`/resume) |

---

## 🧠 Method details (as implemented)
### 1) Auxiliary information extraction
- Match POS across neighboring environments in **objective space** via Euclidean distance (Munkres for assignment) to obtain aligned pairs `x_t ↔ x_{t-1}`【8†Adversarial AutoEncoder-Based Large-Scale Dynamic Multi-Objective Evolutionary Algorithm†L286-L318】.
- For each aligned trajectory, compute cosine of consecutive difference vectors to form a 1D time-series per solution; discretize to states and fit a **discrete Markov chain** to predict the next-angle interval `[δ_lower, δ_upper]`【8†Adversarial AutoEncoder-Based Large-Scale Dynamic Multi-Objective Evolutionary Algorithm†L246-L318】.

### 2) AAE with transfer loss
- Train encoder `E`, decoder `D`, and discriminator `A`.  
- **Transfer loss** encourages decoded `x'` to satisfy the predicted cosine-interval constraint (directional locality) for the next environment; adversarial regularization matches aggregated posterior to a simple prior (e.g., `𝒩(0,I)`)【8†Adversarial AutoEncoder-Based Large-Scale Dynamic Multi-Objective Evolutionary Algorithm†L319-L390】.

### 3) Initial population generation
- Sample latent `z ~ 𝒩(0,I)` and decode; project steps along unit directions around the current POS with magnitudes based on historical distance statistics to synthesize diverse, convergent initial candidates for `t+1`【8†Adversarial AutoEncoder-Based Large-Scale Dynamic Multi-Objective Evolutionary Algorithm†L391-L471】.

---

## 📈 Outputs
- `runs/<name>/logs/` – training/evolution logs
- `runs/<name>/checkpoints/` – AAE weights per change step
- `fronts/` – archived POF snapshots (per t)
- `figures/` – IGD/HV curves, POS/POF visualizations
- `metrics.csv` – tabular metrics across time

---

## 🔁 Reproducibility
- Set `--seed` and log your `conda env export > env.yml`.
- Determinism in PyTorch (optional):
```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python - <<'PY'
import torch
torch.use_deterministic_algorithms(True)
PY
```

---

## 🧩 Extending to your MOEA
Implement interface in `core/moea/<your_algo>.py` exposing:
```python
def evolve(problem, init_pop, budget, **kwargs) -> dict:
    return {"pos": X_t, "pof": Y_t, "history": ...}
```
AAE-DMOEA will call this at each environment step.

---

## 🐞 Troubleshooting
- **No improvement after change** → increase `aae_epochs`, enlarge `latent_dim`, or reduce `beta` (finer angle states).  
- **Mode collapse / poor diversity** → raise `pop_size`, widen angle intervals, or add Gaussian jitter on decoded samples.  
- **Unstable discriminator** → reduce its lr, or apply gradient penalty / spectral norm.

---

## 🔗 Reference
If this reproduction helps, please cite the original paper:
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

## 📄 License
MIT (for this reproduction). Check the original paper's license/rights for figures or text reuse.
