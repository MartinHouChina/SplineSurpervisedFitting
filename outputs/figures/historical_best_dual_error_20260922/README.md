# 历史最佳候选：六方法双阈值实验

选择权重：`outputs/checkpoints/overnight_stable_k32_3090_r1.pt`，epoch14。
105份载荷审计、40份兼容权重公共验证、前5名扩展验证；模型选择不使用最终61条测试数据。

共同限制：内部K≤32，MSE≤5e-5且最大单点平方误差≤5e-4，均不开方。

## 先看

- `dual_error_overall.png`：六行总体统计，原始→公共修复。
- `dual_error_summary.png` / `.md`：分数据源完整统计。
- `dual_error_five_metrics.png`：五来源×五指标，原始与修复对照。
- `six_methods_raw_6x5.png`：未修复的六方法随机案例。
- `six_methods_repaired_6x5.png`：同一批随机案例，公共插入修复后。
- `dual_error_metrics.json`：完整统计、选图ID和文件哈希。

Ours原始双通过率59.0%，修复后78.7%、平均K23.89。Dung修复后80.3%、K16.78；Luo修复后78.7%、K18.30。当前Ours有本机实测速度优势，但不能声称最少节点或所有曲线达标。

六方法均使用相同追加节点规则，包括Ours；追加后重新求控制点，不是保持曲线不变的纯Boehm插入。保留各方法原参数化与端点约定，不删除/移动原节点。达到32节点仍可能失败。

每方法61条：合成K4..24各1条，UJI/海岸线/等高线/工业等距线各10条。工业等距线为程序生成。失败保留在分母；Dung有1条原始分段超容量失败，误差/节点均值使用其60条有限拟合。

文献组为仓库适配实现；新增插入步骤为公共比较封装，不能称为作者原生方法。修复后的Ours也不是一次性网络部署。时间为GTX1070网络/CPU4线程本机预览，不能当作3090正式速度结论。

详细报告和复现命令：`docs/historical_best_dual_error_20260922.md`。
完整几何：`outputs/comparisons/historical_best_dual_error_20260922/`。
