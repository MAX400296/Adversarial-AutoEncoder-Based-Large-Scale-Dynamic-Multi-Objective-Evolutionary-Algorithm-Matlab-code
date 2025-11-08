# AAE-DMOEA-Reproduction（MATLAB/非官方复现）
这是论文 **《Adversarial AutoEncoder-based Large-Scale Dynamic Multi-Objective Evolutionary Algorithm (AAE‑DMOEA)》** 的**个人 MATLAB 复现**。并非 Python。

> ⚠️ 说明：非官方实现，仅用于科研与教学。如果发现错误可以与我联系，与原作者无关，如果原作者发现代码与实际不符请与我联系。

---

[English Version](README.md)


## 🔧 依赖环境
- **MATLAB R2021b 或更高**
- 工具箱：
  - **Deep Learning Toolbox**（`AAE.m` 使用 `dlarray` 自定义训练循环）
  - **Statistics and Machine Learning Toolbox**（`pdist2`）
- 可选：Parallel Computing Toolbox（加速）。
- 需要安装cuda以实现matlab调用GPU，相关教程可以搜索

---

## 📁 代码结构（来源于上传内容）
```
Public code/
├─ DemoMain.m                       # 演示入口
├─ AAE/
│  ├─ AAEDMOEA.m                    # 主流程（框架入口）
│  ├─ AAE.m                         # 对抗自编码器（训练/推理）
│  ├─ Extracting_auxiliary_information.m   # 辅助信息（余弦/马尔可夫）
│  ├─ Markov_chain_predictor.m
│  ├─ LMOEADS/                      # 基础 MOEA 组件
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
│     ├─ munkres.m                  # 匈牙利算法
│     └─ license.txt
└─ Benchmark/
   ├─ TestFunctions.m               # DF1 测试函数（CEC2018 风格）
   └─ pof/POF-nt5-taut10-DF1-*.txt  # 参考 POF（用于作图/对比）
```

---

## 🚀 运行方法
### （A）快速演示（推荐）
1. 打开 MATLAB。
2. **添加路径**（含子目录）：
   ```matlab
   addpath(genpath('Public code'));
   ```
3. 运行演示：
   ```matlab
   run('Public code/DemoMain.m');
   ```

演示脚本会：
- 配置 DF1 的默认参数；
- 运行 **AAEDMOEA**，跨多个环境变化；
- 绘制第 30 次环境变化时的 **真实 POF** 与 **优化后 POF** 对比。

> 作图示例（见 `DemoMain.m` 尾部）：
> ```matlab
> scatter(True_Y(:,1),True_Y(:,2),'b'); hold on;
> scatter(EPF(1,:),EPF(2,:),'r'); hold off;
> legend({'TruePOF','Optimized POF'});
> ```

### （B）按函数调用（可自定义）
```matlab
addpath(genpath('Public code'));

% 1) 定义问题（演示只提供 DF1）
dim = 100;
Problem = TestFunctions('DF1', dim);

% 2) 运行参数
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
group   = 1;                % 选用上面某一行（例如 8 → [5 10 300]，与提供的 POF 文件标签一致）
popSize = 100;
MaxIt   = 5000 * T_parameters(group,2);   % 演示脚本中的设置

% 3) 运行框架
res = AAEDMOEA(Problem, popSize, MaxIt, T_parameters, group);
```

**关于 `T_parameters = [nt, taut, T_max]`：**  
- `nt`   —— 变化频率因子（POF 文件名里有 `nt5` 标签）；  
- `taut` —— 变化周期/强度（并用于计算 `MaxIt`）；  
- `T_max`—— 总时域（一次运行的变化步数或评估上限）。  
提供的 POF 文件名是 **`POF-nt5-taut10-DF1-<t>.txt`**，对应 `nt=5`、`taut=10`（上表第 **8** 行 `[5 10 300]`）。

---

## 🧠 方法要点（MATLAB 实现）
- **辅助信息**（`Extracting_auxiliary_information.m`）：
  - 用匈牙利算法（`munkres.m`）在目标空间匹配历史 POS/POF；
  - 构造每个解的**余弦角**时间序列，训练**离散马尔可夫链**预测下一步角度区间 `[δ_lower, δ_upper]`。
- **AAE + 迁移损失**（`AAE.m`）：
  - 基于 `dlarray` 的自定义训练循环；
  - 只约束**方向**（角度区间），将解码样本限制在下一环境的**局部区域**。
- **初始化生成**：
  - 先验采样 → 解码 → 沿单位方向、按历史距离统计设定步幅，生成多样且易收敛的新环境初始种群。
- **基础 MOEA**：`LMOEADS` 目录提供了演示用的 MOEA 组件，供框架调用。

---

## 📦 输出说明
`AAEDMOEA` 返回 cell 数组 `res`，每个变化步 `T` 含：
```matlab
res{T}.turePOF   % 参考真值 POF（Benchmark/pof）
res{T}.POS       % 决策变量（第 T 步的帕累托集）
res{T}.POF       % 目标值（第 T 步的帕累托前沿）
res{T}.initPop   % 第 T 步使用的初始化种群
res{T}.initPOF   % 初始化种群的目标值
```

---

## 🐞 常见问题
- **变化后改进有限** → 调大 `taut`（从而提高 `MaxIt`），或增大 `popSize`；
- **AAE 训练不稳定** → 降低 `AAE.m` 中的学习率（`settings.lrD`, `settings.lrG`），或增加 `settings.maxepochs`；
- **找不到 `dlarray`** → 未安装 Deep Learning Toolbox。

---

## 📄 许可证
本复现以 MIT 发布；第三方代码按其各自许可证（如 `munkres.m`）。

## 🔗 引用原论文
```
Li, C., Yen, G. G., & He, Z. (2024).
Adversarial AutoEncoder-based Large-Scale Dynamic Multi-Objective Evolutionary Algorithm.
IEEE Transactions on Evolutionary Computation. DOI: 10.1109/TEVC.2024.3412049
```
