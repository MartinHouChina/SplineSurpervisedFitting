# M16/M32：数值耦合更新与论文输出

## 结果与边界

新M32 best是第5轮，122条测试77条通过（63.11%），旧Anchored为85条（69.67%）；M16为41条（33.61%），合成测试还包含超过其容量的K17..24。详见[结果审计](overnight_m16_m32_results_audit.md)。不能只靠延长训练解释或解决Joint退步。

这批测试已用于开发诊断，不能包装成未见过的最终论文测试。正式结论需要锁定配置、多训练种子及未参与调参的组级测试。下文是**较长训练与可追溯输出协议**，不是达标承诺或论文验收证书。

## 实际更新链路

原2/2/2特征加深保留，另加可选 `coupled_proposal_steps` 和 `coupled_subset_steps`，旧模型默认均0，新训练各2：

1. 初始编码生成采样参数t、候选U与特征。
2. 每个候选数值块双向交换点/节点特征，更新单调t，按同一参数映射搬移U，再做有界节点残差；下一块读取更新后的t/U/点特征。
3. 自适应概率质量/覆盖约束Top-K只生成**一次离散KeepMask**，没有在线循环贪心删除。
4. 固定该Mask，存活集数值块继续逐轮更新t/U；删除节点不能作存活解码器的Key/Value。
5. 最终做一次标准B样条refit。块内没有最小二乘、误差搜索或回退。

Teacher评估子集走同一个解码器，额外搜索仅在训练。新块零门控初始化；无固定弦长混合时可保持兼容旧模型的初始几何。候选更新不依赖解码器可训练变量，冻结候选阶段不会因解码器优化而暗中漂移。增深仍增加网络时间，未声称同速或已完成长程重训。

输出保留 `pre_coupling_params`、`pre_coupling_internal_knots` 与 `coupled_parameter_delta`，便于区别首轮解码和新增数值更新。历史 `ungated_subset_params`、`subset_parameter_trust`仍只描述首轮解码，不代表所有后续更新；`warped_proposal_internal_knots`始终指原始候选从初始参数映射到最终参数的位置。

## 弦长参考而非统一替代

配对诊断见 `outputs/diagnostics/m16_m32_parameterization_20260921_saved_inputs.json`。同输入、同Mask、节点随参数一致warp后refit；非仿射warp不是保持原样条形状的精确重参数化。

24条服务器保存外部案例中，M32弦长使UJI4/6、NaturalEarth6/6、USGS5/6条MSE下降，工业等距线仅1/6；各来源通过数未变。8条本机补充合成案例通过数5降到2，不支持“弦长全面更好”。

因此每个新参数更新头读取 `chord_t` 与 `current_t-chord_t`，学习有界参数间距更新，既有弦长反事实损失继续监督。默认固定混合系数0。`--parameter-chord-blend 0.5`是显式消融；1只固定初始Proposal参考，后续更新仍能改变t，不能称全程固定弦长。

## 训练配置

| 设置 | M16 | M32 |
|---|---:|---:|
| 内部候选上限 | 16 | 32 |
| 有标签合成源K | 4..16 | 4..24 |
| 初始化 | 已训练M16 best | 已训练M32 best |
| 总轮数 | 60 = 12 Proposal + 48 Joint | 同左 |
| Joint冻结候选校准 | 前8轮 | 前8轮 |
| 训练 / 合成验证 / Batch | 3000 / 500 / 32 | 同左 |
| 每曲线点数 | 192 | 192 |
| Joint LR / 候选比例 / 解码比例 | 1e-5 / 0.1 / 0.25 | 同左 |
| 额外数值更新 | Proposal2轮 + subset2轮 | 同左 |
| MSE阈值 / 峰值平方误差目标 | 5e-5 / 5e-4 | 同左 |

较小Joint学习率针对后期退步，不是已证实最优。保持混合合成训练；UJI、NaturalEarth、USGS各100条独立验证。IndustrialOffset是程序生成的工业等距线测试源，不冒称实测工业数据。两模型源K范围不同，不是纯容量消融；不按单条测试误差挑模型。

## 上传与Linux一条龙

Windows PowerShell上传：

```powershell
scp "E:\SelfSurpervisedSplineFitting\outputs\delivery\overnight_coupled_paper_linux_update.zip" feng@10.76.0.64:/home/feng/HouCode/
```

Linux激活已有PyTorch环境后执行（先备份服务器未提交的源码修改）：

```bash
cd /home/feng/HouCode/SplineFitting_1070_overnight
unzip /home/feng/HouCode/overnight_coupled_paper_linux_update.zip
bash scripts/run_v16_paper_training_linux.sh \
  --device cuda \
  --data-root /home/feng/HouCode/SplineSurpervisedFitting/data \
  --prepare-real-data \
  --run-prefix paper_coupled_3090_r1
```

要求保留 `outputs/checkpoints/universal_m16_m32_3090_r1_m16.pt` 和 `_m32.pt`；也可用 `--m16-checkpoint`、`--m32-checkpoint`明确指定。不能把旧Windows `.last.pt`当新架构resume。

先追加 `--dry-run`应显示两个独立流程；`--capacities 32`仅运行M32。默认各外部来源最多100条test记录、12条案例PNG，合成K4..24每K10条。**覆盖所有来源不等于遍历所有记录**，如需完整test split，加 `--all-real-test-samples`；测试可能很久，不承诺隔夜时限。

## 原生对照尚未完成：必须披露

默认 `--baseline-protocol native`拒绝未核实实现。当前五篇均未达到原版资格，结果为 `unavailable` / N/A，不是“拟合失败0%”，不会降级运行适配版。因此目前能完整保存Ours结果，**不能生成有效的六原版方法排名**。

见[原论文审计](baseline_native_audit.md)：Dung缺少重数/角度分类且最终求解不同；Kang缺少一般数据Algorithm1/3；其余有全文/公式与作者程序核验缺口。需完成原版实现与论文例子等价验证后才能正式比较。

若只想继续探索性比较，可显式加 `--baseline-protocol adaptation`。本新入口仍关闭公共误差修补，并标为**未修补适配版**，不是原版；旧入口默认行为不变。

## 测量保存与PNG

每模型在 `outputs/checkpoints`、`logs`、`comparisons`、`figures`下独立保存带 `_m16`/`_m32`名字的结果。每个评估样本/方法导出JSON+压缩NPZ，包括：原始/归一化输入和reference、归一化变换、采样参数、内部/完整节点、控制顶点、输入处拟合点、密集曲线、逐点平方误差、MSE/最大平方误差、network/完整耗时、协议和失败原因。SHA256把几何和测量记录关联；不可用方法不伪造几何或时间。

案例读取保存结果，不再运行一次拟合，避免图表对应不同结果。绘制全部已存案例：

```bash
python scripts/visualize_v16_six_methods.py \
  --benchmark-dir outputs/comparisons/paper_coupled_3090_r1_m32 \
  --output-dir outputs/figures/paper_coupled_3090_r1_m32/all_saved_cases \
  --max-cases-per-dataset 0
```

旧测量没有完整几何时拒绝补造控制点，应重测到新目录。时间图报告完整方法时间，另列network时间；误差不开方，原始与归一化单位分开。最大点残差不是Hausdorff或连续曲线误差证书。

本机USGS只有201条test而服务器记录1420；部分本机重建输入hash也不同。本地烟测用于检验实现/导出，不冒充服务器全量实验。最终应在服务器完整数据树上运行并保存manifest/hash。
