# 本机已下载模型：论文预览图版

本次不训练、不修复模型输出、不改误差。输入权重：

- `outputs/downloads/server_models_20260922_215418/paper_coupled_clean_3090_r2_m16.pt`
- `outputs/downloads/server_models_20260922_215418/paper_coupled_clean_3090_r3_m32.pt`

两份保存的是epoch13（12轮Proposal后的第一轮Joint）的best状态；不能由此推断整个服务器任务只运行了13轮。计划60轮不是保存epoch。当前资格标记不满足正式报告门槛，因此本次是这些具体权重的本机固定样本预览，而不是已通过资格认证的最终论文成绩。

## 图版

顶层目录 `outputs/figures/local_models_paper_20260922/`。
框架图与参数表共享；M16/M32各自有独立子目录，六方法仍各6组，不把两个Ours混成一组。

- 六方法：6列（Ours、Park、Liang、Dung、Kang、Luo）×5行（Synthetic、UJI、NaturalEarth、USGS、IndustrialOffset），每来源一个固定随机案例。
- Ours：5列×5行，每来源5个例子。从本轮已评估样本中优先选拟合较好的不同group，保存具体选择规则及ID。这是精选展示，不能用于估计整体通过率。
- 曲线图标识参考曲线、输入点、拟合曲线、控制多边形/顶点、内部节点的曲线位置及一维参数域节点。面板指标来自同一已测拟合，绘图不重算拟合。
- 统计表：四核心指标（MSE、通过率、平均内部K、完整时间）外加最大平方误差。统计使用全部61条已尝试曲线，分来源每方法报告；异常保留分母，有限输出但超阈值也算不通过。

最大误差不取平方根，定义为离散对应点的最大平方欧氏残差；源内最坏值是对该来源所有有限曲线的峰值再取max。它不同于最大单曲线MSE、坐标RMSE、连续Hausdorff距离。原始参考点误差和192点输入网格误差分开保存。

## 固定本机试验

每容量61曲线：合成K4..24各1条（21条）；各外部来源10条（40条）。源内采用旧有固定种子的group轮转选择，不是前一条龙的10%逐样本随机协议，二者不能混报。seed20000，selection_seed20260922；两个模型输入来源与选择种子相同，保存内容hash供核对。

工业等距线是程序生成CAD式偏置曲线，不是实测工业数据。M16的合成K17..24超候选容量，做主结论时须另列共同K4..16。训练数据源和候选容量不同，不能将M16/M32这次对照称为纯容量消融。

本机GTX1070；Ours网络在CUDA运行，数值基线为CPU。完整方法时间单次测量，网络预热3次后10次测量；不并行运行两个benchmark以免互相争抢资源。单次时间噪声较大，论文性能结论应在空闲3090环境中多次重复。当前五文献方法均为仓库adaptation，不是已经核实的原生复现；公共额外可行性补点关闭。

network与完整时间是独立测量，不能相减得到refit时间；单次完整计时抖动时可能小于独立network计时，原始值保留而不人为调整。

## 复现

从项目根目录，已完成报告仅重新绘图：

```bash
python scripts/run_local_models_paper.py --plot-only
```

新目录不存在时运行全部测量和图版：

```bash
python scripts/run_local_models_paper.py --device cuda
```

已存在兼容结果且需继续中断测量，加 `--resume`。仅跑一个容量加 `--capacities 32` 或 `--capacities 16`。输入数据/代码指纹改变时应另建实验，不覆盖旧记录。

原始测量与控制点/节点向量JSON、NPZ在：

```text
outputs/comparisons/local_models_paper_20260922_m16/
outputs/comparisons/local_models_paper_20260922_m32/
```

本次只提供图版和测量，不据此填入尚无独立验证的优越性结论。精选案例、随机对照案例和全体统计三者分开。
