# CAD / CAGD 样条拟合论文选读与写作风格说明

本文档用于辅助阅读，不作为投稿正文的一部分。论文正文中的技术结论仍以原文为准。

## 1. 与本文最接近的曲线节点配置论文

| 论文 | 期刊 | 核心思路 | 建议重点阅读 |
|---|---|---|---|
| [Adaptive knot placement in B-spline curve approximation](https://doi.org/10.1016/j.cad.2004.09.008) | Computer-Aided Design, 2005 | 平滑离散曲率并据其积分自适应布置节点 | 摘要与引言如何从“稠密、含噪数据”引出节点配置困难 |
| [B-spline curve fitting based on adaptive curve refinement using dominant points](https://doi.org/10.1016/j.cad.2006.12.006) | Computer-Aided Design, 2007 | 将节点配置转化为主导点选择，并进行局部自适应细化 | 四阶段流水线的叙述方式：参数化、主导点、节点、最小二乘 |
| [Knot calculation for spline fitting via sparse optimization](https://doi.org/10.1016/j.cad.2014.08.022) | Computer-Aided Design, 2015 | 在过完备表示中用稀疏优化同时控制节点数量与位置 | 如何把“节点数 + 节点位置”写成统一优化目标 |
| [Learnt knot placement in B-spline curve approximation using support vector machines](https://doi.org/10.1016/j.cagd.2018.03.019) | Computer Aided Geometric Design, 2018 | 根据几何和微分几何特征学习逐位置节点质量分数 | 学习式节点配置如何与传统 B 样条求解器分工 |
| [Fast Automatic Knot Placement Method for Accurate B-spline Curve Fitting](https://doi.org/10.1016/j.cad.2020.102905) | Computer-Aided Design, 2020 | 用导数特征函数表征细节分布，并按特征含量配置节点 | 摘要的“问题—困难—核心机制—比较结果”结构，以及对适用条件的说明 |

## 2. 曲面拟合与参数化论文

| 论文 | 期刊 | 核心思路 | 与本文的联系 |
|---|---|---|---|
| [Surface fitting based on a feature sensitive parametrization](https://doi.org/10.1016/j.cad.2006.04.007) | Computer-Aided Design, 2006 | 给高曲率区域分配更多参数空间，使均匀节点形成特征感知的自由度分布 | 支持“参数化与模型容量不能完全分开”的论述 |
| [B-spline surface fitting by iterative geometric interpolation/approximation algorithms](https://doi.org/10.1016/j.cad.2012.02.011) | Computer-Aided Design, 2012 | 通过几何规则迭代更新曲面控制网格 | 展示标准样条结构与显式数值更新结合的写法 |
| [Spline surface fitting using normal data and norm-like functions](https://doi.org/10.1016/j.cagd.2018.06.005) | Computer Aided Geometric Design, 2018 | 同时拟合点和法向，以改善复合曲面的近似光顺性 | 说明不同几何约束应当在目标中明确区分，而不能只报告点残差 |

本文当前只处理有序曲线，不应把上述曲面结果描述为本文已经具备的能力。正文引用它们，是为了说明参数化、几何特征和表示容量之间的耦合也存在于曲面重建。

## 3. 近期学习式曲线拟合论文

| 论文 | 期刊 | 核心思路 |
|---|---|---|
| [The Deep Neural Network Solver for B-Spline Approximation](https://doi.org/10.1016/j.cad.2023.103668) | Computer-Aided Design, 2024 | 用逐实例神经求解器优化节点变量 |
| [Parameterization for Polynomial Curve Approximation via Residual Deep Neural Networks](https://doi.org/10.1016/j.cagd.2021.101977) | Computer Aided Geometric Design, 2021 | 以残差网络学习曲线参数化 |
| [SplineGen: Approximating Unorganized Points through Generative AI](https://doi.org/10.1016/j.cad.2024.103809) | Computer-Aided Design, 2025 | 自回归生成参数与变长度节点序列 |
| [DNN-based Parameterization for B-Spline Curve Approximation](https://doi.org/10.1016/j.cad.2025.103897) | Computer-Aided Design, 2025 | 逐实例优化 B 样条数据参数 |
| [Neural co-optimization of knots and parameterization for B-spline curve approximation](https://doi.org/10.1016/j.cad.2026.104059) | Computer-Aided Design, 2026 | 显式联合优化参数化和节点位置 |

## 4. 从 CAD / CAGD 论文中提炼的写作风格

1. **摘要先写几何问题，再写网络。** 先说明节点向量为什么影响精度和复杂度，再指出联合优化为何困难，随后只保留最核心的模型机制和量化结果。
2. **引言采用“工程背景—数学困难—方法谱系—明确缺口—本文方案”的顺序。** 不在开头连续堆叠模块名，也不把实验包装器写成网络贡献。
3. **相关工作围绕优化变量分类。** 本稿按特征驱动配置、连续/稀疏优化、曲面参数化、学习式参数与节点预测组织，而不是简单按年份罗列。
4. **明确传统求解器与学习模块的边界。** 网络预测参数和节点结构；控制顶点仍由标准最小二乘求解。这种分工更符合 CAD 论文对可检查几何输出的重视。
5. **结论使用有边界的比较措辞。** 写成“在给定样本和阈值下观察到的折中”，避免把仓库适配实验扩大为对原论文方法的普遍排名。
6. **曲线与曲面能力严格区分。** 曲面论文只用于支撑一般几何原则；当前框架并未实现二维参数域、张量积节点线或曲面接缝约束。

## 5. 本次正文调整

- 摘要压缩为“问题—困难—方法—实验—边界”五部分；
- 引言增加 CAD/CAGD 曲线与曲面拟合的联系，并强化真正的研究缺口；
- 相关工作新增“曲面拟合与几何感知参数化”小节；
- 删除与期刊风格不符的口语化辨析，改为按方法机制比较；
- 增加论文组织段，并保持所有已有实验数字和结论范围不变。
