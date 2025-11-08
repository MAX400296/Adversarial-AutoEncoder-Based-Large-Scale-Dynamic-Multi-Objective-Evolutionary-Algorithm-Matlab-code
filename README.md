# AAE-DMOEA-Reproduction (MATLAB)
_Unofficial personal reproduction of_ **“Adversarial AutoEncoder-based Large-Scale Dynamic Multi-Objective Evolutionary Algorithm (AAE‑DMOEA)”** (IEEE TEVC 2024).  
This repo is **MATLAB** code (not Python).

> ⚠️ ⚠️ Disclaimer: This is a personal, unofficial reproduction for research/education. Not affiliated with the original authors. Any inconsistencies or bugs are my own. If the original author finds that the code does not match the actual implementation, please contact me.

---

[📘 中文版说明 / Read in Chinese](README_CN.md)


## 🔧 Requirements
- **MATLAB R2021b+**
- Toolboxes:
  - **Deep Learning Toolbox** (uses `dlarray`, custom training loops in `AAE.m`)
  - **Statistics and Machine Learning Toolbox** (uses `pdist2`)
- OS: Windows / Linux / macOS (tested on MATLAB R2021b)

Optional:
- Parallel Computing Toolbox (for faster MOEA runs).

---

## 📁 Folder Structure (from the uploaded code)
```
Public code/
├─ DemoMain.m                       # Demo entry script
├─ AAE/
│  ├─ AAEDMOEA.m                    # Main pipeline (framework entry)
│  ├─ AAE.m                         # Adversarial AutoEncoder (training/inference)
│  ├─ Extracting_auxiliary_information.m   # Cosine/Markov auxiliary info
│  ├─ Markov_chain_predictor.m
│  ├─ LMOEADS/                      # Base MOEA (variant of LMOEA/LMOEAD*)
│  │  ├─ LMOEADS1.m
│  │  ├─ DecompositionSelection.m
│  │  ├─ DirectedSampling.m
│  │  ├─ DominationSelection.m
│  │  ├─ DoubleReproduction.m
│  │  ├─ LMOEAInitialization.m
│  │  ├─ NDSort.m
│  │  ├─ SOLUTION.m
│  │  └─ UniformPoint.m
│  └─ munkres/
│     ├─ munkres.m                  # Hungarian assignment
│     └─ license.txt
└─ Benchmark/
   ├─ TestFunctions.m               # DF1 test function (CEC2018 dynamic suite style)
   └─ pof/POF-nt5-taut10-DF1-*.txt  # Reference POFs for plotting/comparison
```

---

## 🚀 How to Run
### (A) Quick Demo (recommended)
1. Open MATLAB.
2. **Add repo to path** (including subfolders):
   ```matlab
   addpath(genpath('Public code'));
   ```
3. Run the demo:
   ```matlab
   run('Public code/DemoMain.m');
   ```

The demo will:
- Configure DF1 with default settings.
- Run **AAEDMOEA** over multiple environmental changes.
- Plot the **true POF** vs **optimized POF** for the 30th change step.

> The plotting block in `DemoMain.m` does:
> ```matlab
> scatter(True_Y(:,1),True_Y(:,2),'b'); hold on;
> scatter(EPF(1,:),EPF(2,:),'r'); hold off;
> legend({'TruePOF','Optimized POF'});
> ```

### (B) Programmatic Call
If you want to call the core pipeline directly:
```matlab
addpath(genpath('Public code'));

% 1) Define the problem (only DF1 is provided in this demo)
dim = 100;
Problem = TestFunctions('DF1', dim);

% 2) Runtime settings
T_parameters = [  % [nt, taut, T_max]
    10  5   100
    10  10  200
    10  25  500
    10  50  1000
     1  10  200
     1  50  1000
    20  10  200
     5  10  300
];
group   = 1;                % select a row above (e.g., 8 --> [5 10 300] to match provided POF files)
popSize = 100;
MaxIt   = 5000 * T_parameters(group,2);   % demo logic

% 3) Run the framework
res = AAEDMOEA(Problem, popSize, MaxIt, T_parameters, group);
```

**What is `T_parameters = [nt, taut, T_max]`?**  
- `nt`   — change frequency factor (dataset tag in POF files, e.g., `nt5`).  
- `taut` — change period / severity factor (**also used to scale `MaxIt`**).  
- `T_max`— total time horizon (number of change steps or evaluations for the run).  
In the provided POF files the tag is **`POF-nt5-taut10-DF1-<t>.txt`**, which corresponds to `nt=5`, `taut=10` (row **8** above: `[5 10 300]`).

---

## 🧠 Method Notes (as implemented in MATLAB)
- **Auxiliary Info Extraction** (`Extracting_auxiliary_information.m`):
  - Match historical POS/POF (Hungarian method in `munkres.m`).
  - Build per-solution **cosine-angle** time series, then a
    **discrete Markov chain** to predict the next interval `[δ_lower, δ_upper]`.
- **AAE w/ Transfer Loss** (`AAE.m`):
  - Custom training loop with `dlarray` (Deep Learning Toolbox).
  - Direction-only constraint (via predicted cosine interval) guides decoded samples into the **local region** of the next environment.
- **Initial Population Generation**:
  - Decode latent samples and perturb along unit directions scaled by **historical distance stats**, forming a diverse, convergent init population for `t+1`.
- **Base MOEA**: The `LMOEADS` folder contains the working MOEA components used inside `AAEDMOEA.m`.

---

## 📦 Outputs
`AAEDMOEA` returns a cell array `res` with one element per change step `T`:
```matlab
res{T}.turePOF   % reference true POF (from Benchmark/pof)
res{T}.POS       % decision variables (Pareto set at step T)
res{T}.POF       % objective values (Pareto front at step T)
res{T}.initPop   % initial population used at step T
res{T}.initPOF   % objective values of the initial population
```

---

## 🐞 Tips & Troubleshooting
- If plots show little improvement after changes, increase `taut` (thus `MaxIt`), or experiment with `popSize`.
- If AAE training is unstable, reduce learning rates in `AAE.m` (`settings.lrD`, `settings.lrG`), or increase `settings.maxepochs`.
- Make sure **Deep Learning Toolbox** is installed (errors around `dlarray` indicate a missing toolbox).

---

## 📄 License
This reproduction code is MIT-licensed. External code under their respective licenses (e.g., `munkres.m`).

## 🔗 Citation
Please cite the original paper if this work is useful:
```
Li, C., Yen, G. G., & He, Z. (2024).
Adversarial AutoEncoder-based Large-Scale Dynamic Multi-Objective Evolutionary Algorithm.
IEEE Transactions on Evolutionary Computation. DOI: 10.1109/TEVC.2024.3412049
```
