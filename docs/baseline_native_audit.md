# 原论文基线复现审计（2026-09-21）

## 同日后续更新：实际运行、去附加修补、无水印

用户确认先运行现有实现。较长训练入口现默认 `adaptation`，五个对照均进入实际
计算，不因未通过原生资格核验而跳过。PNG不加诊断水印；实现来源与资格仍保存。

新默认路径已去掉：公共超阈值补点/满容量回退、Kang的1.25分簇系数、相对jump
截断、singleton额外区间搜索及degree+1长簇保留启发式、Luo无候选时强制最大jump。
Dung/Kang直接返回各自算法中的无强制端点最小二乘解，不再额外端点refit。
Dung超容量不再均匀丢边界，而明确报告capacity limit exceeded；这是共享预算失败，
不是原论文自身不可拟合的结论。部分历史扩展仅保留为显式调用选项。

回查原文后，另补入Kang实验性一般数据Algorithm1/3：插入区间中点、在固定初始
LS误差约束下重解稀疏问题、比较三个导数jump，仅对通过判定的区间做二分合并。
但四来源小样本中，该向量ADMM实现都在后续固定误差约束下变得不可行。
因此它仅是显式实验路径 `relocation_algorithm='general'`，不强行替换现有比较。
当前对比仍运行已有Algorithm4/5适配；原文仅推荐它用于清晰分簇，不能把其在
一般曲线上的退化归因于原版Kang。默认没有恢复degree+1保留、补点等额外修补。
Algorithm3的临时向量允许多一个中点，这是原文步骤，不是部署新增节点。

**仍不等于原版数值等价。** 区间稀疏求解沿用向量group-L1和有限ADMM，
不是作者scalar CVX。固定初始误差下，某些后续区间问题可能不可行；当前明确
报错，不放宽误差、不加回节点、不把未完成的算法记为成功。二分预算、截断、
重解次数均保存。Dung仍缺重数/角度分类，其余待全文核验部分不凭猜测删改。
以下详细表为本次清理前的审计快照，不代表上述已移除机制仍在默认执行。
当前状态亦见代码返回的 `baseline_provenance` 和逐方法诊断字段。

## 结论和使用边界

当前六方法中的五个文献对照都不是已经验证的原版实现。其状态统一为
`reproduction_unverified`，`native_comparison_available=false`。
关闭 `published_feasibility_safeguard` 只能得到**未补点的仓库适配版**，不能恢复
被省略的原论文阶段、原目标函数、原停止条件或原控制点解。

本次添加的是严格协议和逐方法来源审计，**没有完成五个原算法的重新实现**，也没有
通过移除保护代码、调低基线能力或增加修复来制造 Ours 的优势。原生对比未完成时，
不存在可用于论文的“六原版方法排名”。

## 一手材料与核验范围

| 方法 | 一手来源 | 本次实际核验范围 |
|---|---|---|
| Park & Lee (2007) | [出版社论文页](https://www.sciencedirect.com/science/article/pii/S0010448507000024)，DOI `10.1016/j.cad.2006.12.006` | 摘要及引言确认参数化、dominant points、参数平均布点、最小二乘四阶段；未取得可逐项验证的作者程序，未完成全文算法核验。 |
| Liang et al. (2017) | [原文 DOI](https://doi.org/10.1088/1361-6501/aa6a05) | 可访问摘要/预览说明密集参考曲线、解析特征积分和 IKI；本次出版社全文不可读，不能核实精确组合公式、常数及全部插点规则。 |
| Dung & Tjahjowidodo (2017) | [开放原文](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0173857)，[官方 S1 MATLAB 附件](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0173857.s006&type=supplementary) | 已读 §3.2.1--3.2.4 及下载的作者程序，未执行；当前 PATH 没有 MATLAB/Octave。附件提供 serial、parallel 和重数判定，不只是伪代码。 |
| Kang et al. (2015) | [出版社论文页](https://www.sciencedirect.com/science/article/pii/S0010448514001912)，仓库原有 `tmp/Kang2015.pdf` | 已提取并核对正文 §3.1--3.3、式(12)、Algorithms 1--5；渲染检查印刷页184--185，确认一般数据所需分支。未找到可运行的作者 CVX 程序。 |
| Luo, Kang & Yang (2022) | [出版社论文页](https://www.global-sci.com/JCM/article/view/12502)，DOI `10.4208/jcm.2012-m2020-0203` | 确认是混合范数稀疏选择 + DE 论文，不是 DNN 论文。全文下载地址在本次访问中返回 HTML 论文页，公开 PDF 仅为预览；不能把现有注释声称的细节当成独立验证。 |

原有 Kang PDF 的 SHA256 为
`c15103076ff84ab030749a641df6a5a6ff44b6a22f8c69f6d322a5bfd7a9cdbb`。
Dung 官方 ZIP 的 SHA256 为
`2fc1830966f71ce77f7c29e33ecf321b5fddca5f5bfca75458d667a1d1ce0399`。
后者本次下载至 `tmp/native_baseline_sources/dung2017_s006.zip`，仅供审计，没有运行
下载脚本，也没有把它冒充为当前 benchmark 已使用的求解器。

## 清理前对照代码的确切差异（审计快照）

下表把“在源码中可直接确认的变更/缺失”和“尚未验证的论文细节”分开；并不声称
已访问不到的完整论文中不存在某个步骤。代码路径均相对
`src/spline_fitting/evaluation/`。

| 方法 / 代码位置 | 当前执行路径 | 原版资格问题 |
|---|---|---|
| Park：`park_dominant_point.py::fit_park_dominant_points` | 从满足阶数的最小 dominant set 起，每次优先消耗一个 LCM seed；seed 不足时按形状指标补齐；所有扫描由共同 MSE 停止。`_knots_from_dominant_parameters` 对退化端点做 clamp。 | 本地模块已承认没有最大距离/迭代正交距离协议；没有参数更新。Menger 曲率、逐个消耗种子和补种规则没有与作者程序逐项比对。不能仅因采用平均布点和形状指标就称完整 DOM 原版。 |
| Liang：`liang_feature_iki.py::normalized_arc_curvature_feature`、`fit_liang_feature_iki` | 用采样折线的无符号转角近似累计曲率，以归一化弧长和转角的线性组合（默认0.5）构造 CDF；最坏对应点所在 knot span 的中点作为新节点。 | 这套具体特征公式、权重、离散积分和 IKI 规则未由原文验证。初始节点数、密集节点数和共同 MSE/端点约束是仓库设置。不能用后续文章的描述替代对 Liang 原文的完整核验。 |
| Dung：`dung_direct_knot.py::_relocate_simple_boundary` | 每个边界只优化一个 simple knot；容量溢出时先调用 `_uniform_capacity_indices` 均匀丢弃粗边界；单段短尾还有合并保护。 | 作者 `TwoPieceOptimalKnotSolver1.m` 对 `1..degree+1` 重数分别扫描/优化，然后结合 joining angle、`MinAngle`、`MaxSmooth` 选择重数。当前完全没有这条分类分支。只实现 serial 本身不是错误（它是论文正式变体）；但不能声称实现了 parallel split/join/shift。 |
| Dung：`fit_dung_direct_knots` 的最终解 | 先生成不约束端点的 `native_fit`，随后改为插值两端点的 `final_fit`，共同评测只使用后者。默认 `max_error=sqrt(mse_tolerance)`。 | 作者 `BsplineFitting.m` 使用 `ANmat\Ymat`，不强制端点等于观测。共同 refit 改变最终曲线；最大距离与平均平方距离也不等价。较小残差并不能弥补重数分支缺失。 |
| Kang：`sparse_knot_paper.py::_select_constrained_sparse_state` | 从标量问题扩展为二维 group-L1；通过有限 ADMM、惩罚参数搜索近似满足 MSE 约束。活跃判断还采用绝对/相对阈值。 | 原文式(12)是标量 constrained L1，数值实验使用 CVX。向量范数改变目标；未做与原凸问题/原例子的数值等价验证。 |
| Kang：`_algorithm4_clusters_are_clear`、`_relocate_clusters` | 用 `degree+1` 长度门槛决定是否重定位；长簇直接保留活跃节点。短簇用 Algorithms4/5-inspired 单/双节点分支，singleton 又执行额外局部区间搜索；分簇阈值为初始间距的1.25倍。 | 原文 Remark3.2 指一般数据优先 Algorithm1；其中 Algorithm3 会插入中点后重新求解凸问题以判断候选区间。当前 Algorithm1/3 不存在；“保留长簇”不是补齐它们。原文分簇阈值是初始间距；singleton 搜索和 `degree+1` 门槛为仓库规则。 |
| Kang：`published_baselines.py` 包装 | `feasibility_repair=False` 已关闭模块内贪心加回，但随后仍执行公共端点约束 refit；外部通用 safeguard 默认可再补点。 | 关闭任意一个补点开关都不能恢复缺失的 Algorithm1/3 或原标量目标。原文也不保证局部重定位后仍满足 sparse-stage 的同一误差值。 |
| Luo：`luo_linf_de.py::_select_regularization`、`_local_maximum_candidates` | 按共同 MSE 搜索正则强度；局部极值全部被筛掉时强制保留最大 jump，极短 jump 列表另有特殊分支。 | 这些是当前代码可确认的行为；因完整算法未核验，不将 fallback 认作原文规定，也不无依据断言所有分支都错误。 |
| Luo：`_differential_evolution`、`_ordered_with_gap` | 候选向量外的人口来自均匀随机位置，使用 best/2 型更新、排序/clamp/最小间距投影；每个候选适应度均由端点约束 LS 计算，最终再次同样 refit。 | 原始正则参数方案、ADMM 等价性、DE 初始化/更新/边界规则均未有作者程序对照；当前不是只改了“最终报告指标”，搜索过程也受共同 refit 影响。 |

共同 safeguard 位于 `published_baselines.py::_common_mse_feasibility_safeguard`：
原适配结果超阈值时，按残差从原候选和均匀网格补点，最终可能选取满容量均匀解。
它是明确的附加算法；在原版比较中既不能执行，也不能将其结果、节点数或耗时归给
被引论文。共同端点 refit 同样不是单纯的“读取指标”。若以后接入原版解，应直接
评估原版给出的曲线；需要共同 refit 的实验必须另列为适配/消融。

## 首次审计引入的协议（后续默认行为见页首更新）

公共入口 `run_published_baseline(..., baseline_protocol=...)` 支持：

- `adaptation`：历史数值路径，保留历史 safeguard 默认值，明确保存
  `reproduction_status`、`baseline_provenance`、`source_native_algorithm_executed=false`。
  `published_feasibility_safeguard=False` 只代表 unrepaired adaptation。
- `native`：拟合、参数化、重定位、修复和计时预热之前核验。
  当前会抛出 `NativeBaselineUnavailableError`，附带 `method` 与 `provenance`；
  不降级运行适配版、不执行额外步骤、不编造原版指标。

来源接口为 `baseline_provenance(method)`、
`validate_baseline_protocol(method, protocol)`，定义于 `native_protocol.py`。
旧 `native_*` / `comparison_feasibility_native_*` 字段为兼容保留，但其含义只是
“仓库适配在通用修复前”的结果，新 `legacy_native_fields_scope` 已明确说明。
Yeh 同样未获原版资格；`uniform_gradient_pruning` 标为 `repository_control`，
不能冒充被引论文的原版。

严格报告遇到不可用方法时，应保留全部计划样本记录及来源，几何、MSE、K、方法
运行时间为 null，图上显示 N/A。不可用不是“原论文拟合失败”，也不是“0%原版
通过率”；不得将它算作 Ours 战胜对照的证据。实现异常与已执行但超阈值的样本
应分别记录，不能选择性删掉差结果。

## 补全为原版还需要什么

1. Dung 可从官方源码开始，保留已声明的 serial/parallel 变体、所有重数/角度判定
   和原最终 LS；先复现附件自带 `KangSpline.mat`，再做参数曲线等价与容量行为测试。
   现成源码使这一步可推进，但下载成功不等于 Python adapter 已完成。
2. Kang 实验路径已补入 Algorithm1/3 控制流程，但仍需核验原 constrained solve 的数值等价性，
   明确标量与向量扩展是否属于同一对照任务；不能把 group-L1 改写或有限求解失败隐去。
3. Park、Liang、Luo 需取得完整合法来源/作者实现，建立公式、停止条件、参数化、
   数值容差、退化样本、最终输出及论文例子的逐项核验记录。
4. 只有完成验证且有真正 source-native 执行入口后才改原版资格；不得仅把注册表
   的布尔值改为 true。固定迭代预算、共同 K 上限若改变原停止准则，也须作为独立
   受限协议披露，而非偷偷截断或替换原结果。

本次没有更改训练、模型、Bash 启动器和各基线的数学路径。针对新协议的测试验证
所有未验证方法在任何数值工作前失败、关闭 safeguard 不会提升复现身份、来源记录
可序列化且不被调用方修改污染，以及历史默认参数保持不变。
